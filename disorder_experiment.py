import argparse
import numpy as np
import sigpy as sp
from joint_estimation import JointEstimation
from transform import RigidTransform
from utils import compute_transform_grids, generate_motion_parameters
import sigpy.plot as pl
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def generate_disorder_mask(kspace_shape, num_shots, partition_size, partition_axis=1):
    """
    Generate a DISORDER sampling mask for k-space data with uniform coverage.

    Each shot randomly selects one k-space point per partition, without resampling across shots.

    Parameters:
    - kspace_shape: tuple (num_coils, kx, ky, kz)
    - num_shots: int
    - partition_size: tuple (p1, p2), block size along two selected axes
    - partition_axis: int, 1→(ky,kz), 2→(kx,kz), 3→(kx,ky)

    Returns:
    - mask: bool array (num_shots, kx, ky, kz)
    """
    num_coils, kx, ky, kz = kspace_shape
    mask = np.zeros((num_shots, kx, ky, kz), dtype=bool)

    # Maps for axes and their dimensions
    axes_map = {1: (1, 2),  # ky, kz
                2: (0, 2),  # kx, kz
                3: (0, 1)}  # kx, ky
    dims_map = {1: (ky, kz),
                2: (kx, kz),
                3: (kx, ky)}

    if partition_axis not in axes_map:
        raise ValueError("partition_axis must be 1, 2, or 3.")
    axis1, axis2 = axes_map[partition_axis]
    dim1, dim2 = dims_map[partition_axis]

    p1, p2 = partition_size
    grid1 = np.arange(0, dim1, p1)
    grid2 = np.arange(0, dim2, p2)

    sampled_points = set()
    rng = np.random.default_rng(32)

    for shot in range(num_shots):
        shot_mask = np.zeros((kx, ky, kz), dtype=bool)
        for g1 in grid1:
            for g2 in grid2:
                start1, end1 = g1, min(g1 + p1, dim1)
                start2, end2 = g2, min(g2 + p2, dim2)
                # Collect available points in this block
                avail = []
                for i in range(start1, end1):
                    for j in range(start2, end2):
                        if (i, j) not in sampled_points:
                            avail.append((i, j))
                if not avail:
                    continue
                sample1, sample2 = avail[rng.integers(len(avail))]
                sampled_points.add((sample1, sample2))

                # Build slicer for 3D mask (kx, ky, kz)
                idx = [slice(None)] * 3
                idx[axis1] = sample1
                idx[axis2] = sample2
                shot_mask[tuple(idx)] = True
        mask[shot] = shot_mask

    return mask

def compute_nrmse(gt, recon):
    """Normalized RMSE between ground truth and reconstruction."""
    return np.linalg.norm(recon - gt) / np.linalg.norm(gt)

def compute_ssim(gt, recon):
    """Compute Structural Similarity Index (SSIM)."""
    return ssim(gt, recon, data_range=gt.max() - gt.min())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint estimation for motion corruption')

    parser.add_argument('--max_iter', type=int, default=1000, help='Max number of joint estimation iterations.')
    parser.add_argument('--init_transforms_file', type=str, default=None, help='Inital guess transform params file.')
    parser.add_argument('--sense', type=bool, default=True, 
                        help='If set to true we do a quick sense recon and use that image as our initial img.')
    parser.add_argument('--rot_sample_range', type=float, default=4.0)
    parser.add_argument('--trans_sample_range', type=float, default=2.0)
    parser.add_argument('--accel_factor', type=int, default=2)
    parser.add_argument('--device', type=int, default=-1)

    parser.add_argument('ground_file', type=str, help='Ground Truth image.')
    parser.add_argument('mps_file', type=str, help='Coil maps file.')
    parser.add_argument('output_dir', type=str)
    parser.add_argument('num_corruption_shots', type=int)
    parser.add_argument('num_recon_shots', type=int)

    args = parser.parse_args()

    ground_truth = np.load(args.ground_file)
    mps = np.load(args.mps_file)
    
    num_shots = args.num_corruption_shots
    img_size = ground_truth.shape[0]
    exp_device = sp.Device(args.device)
    rotation_sample_range = args.rot_sample_range
    translation_sample_range = args.trans_sample_range
    num_recon_shots = args.num_recon_shots

    #Genearate random motion parameters to simulate continuous motion
    #So we transform essentially every lines scan in the kspace
    transforms = generate_motion_parameters(16*8, high_freq_var=2.0, seed=180)

    #Create a shot mask that samples for each shot and simulates and accel factor for aliasing
    shot_mask = generate_disorder_mask(mps.shape, 8, (4, 4), partition_axis=1)
    #pl.ImagePlot(shot_mask)
    corruption_mask = np.zeros((16*8,) + ground_truth.shape, dtype=bool)
    corr_bin_size = img_size // 16
    for s in range(8):
        for i in range(16):
            corruption_mask[s*16+i, :, i*corr_bin_size:(i+1)*corr_bin_size] = shot_mask[s, :, i*corr_bin_size:(i+1)*corr_bin_size]
    #pl.ImagePlot(corruption_mask) 
    #Generate corrupted kspace and save corrupted image
    kgrid, rkgrid = compute_transform_grids(ground_truth.shape, ground_truth.shape, [1.5, 1.875, 2])
    S = sp.mri.linop.Sense(mps)
    ksp = 0
    for s in range(corruption_mask.shape[0]):
        A = sp.linop.Multiply(S.oshape, corruption_mask[s])
        T = RigidTransform(ground_truth.shape, ground_truth.shape, transforms[s], kgrid, rkgrid)
        ksp += (A * S * T * ground_truth)
    
    corrupted_image = S.H * ksp
    #pl.ImagePlot(corrupted_image)
    #Set up constraints and preconditions
    xp = sp.get_array_module(mps)
    rss = sp.rss(mps, axes=(0,))
    M = rss > xp.max(rss) * 0.1
    P = sp.linop.Multiply(mps.shape[1:], 1 / (xp.sum(xp.abs(mps) ** 2, axis=0) + 1e-3))
    
    #Naive Sense recon for comparison and/or initialization
    sense_recon = sp.mri.app.SenseRecon(ksp, mps, device=exp_device).run()
    sense_recon = sense_recon.get() * M

    #Need to copy the sense img if we use as start because algo will modify this ref 
    init_image_guess = sense_recon.copy() if args.sense else None


    kgrid, rkgrid = compute_transform_grids(ground_truth.shape, ground_truth.shape, [1.5, 1.875, 2], device=exp_device)
    recon, estimates = JointEstimation(ksp, mps, shot_mask, kgrid, rkgrid, 
                            device=exp_device, P=P, constraint=M, img=init_image_guess, 
                            max_joint_iter=args.max_iter, tol=1e-12).run()
    
    recon = recon.get()
    estimates = estimates.get()
    #pl.ImagePlot(recon)
    experiment_name = f"disorder_recon"
    np.save(f"{args.output_dir}/{experiment_name}", recon)
    np.save(f"{args.output_dir}/sense_recon", sense_recon)

    nrmse_sense_recon = compute_nrmse(ground_truth, sense_recon)
    nrmse_joint_recon = compute_nrmse(ground_truth, recon)
    ssim_sense_recon = compute_ssim(np.abs(ground_truth), np.abs(sense_recon))
    ssim_joint_recon = compute_ssim(np.abs(ground_truth), np.abs(recon))
    nrmse_estimates = compute_nrmse(transforms, estimates) if num_recon_shots == num_shots else None

    # Create a unique log file name
    log_filename = f"{args.output_dir}/output_log_{experiment_name}.txt"

    # Write results to a unique log file
    with open(log_filename, "w") as log_file:
        log_file.write(f"### MRI Reconstruction Log (Job {experiment_name}) for {args.max_iter} iterations ###\n\n")
        log_file.write(f"NRMSE sense recon: {nrmse_sense_recon}\n")
        log_file.write(f"NRMSE joint recon: {nrmse_joint_recon}\n\n")
        log_file.write(f"SSIM sense recon: {ssim_sense_recon}\n")
        log_file.write(f"SSIM joint recon: {ssim_joint_recon}\n\n")
        log_file.write("Final transform estimates, rotations in degrees\n")
        estimates[:, 3:] = estimates[:, 3:] * 180 / np.pi
        for s in range(estimates.shape[0]): 
            log_file.write(f"Shot #{s+1}: ")
            log_file.write(",  ".join(f"{val:.3f}" for val in estimates[s]) + "\n")

    print(f"Log file saved as {log_filename}")