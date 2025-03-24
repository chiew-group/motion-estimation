import argparse
import numpy as np
import sigpy as sp
from joint_estimation import JointEstimation
from transform import RigidTransform
from utils import compute_transform_grids
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


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
    parser.add_argument('num_recon_shots', type=int)

    args = parser.parse_args()

    ground_truth = np.load(args.ground_file)
    mps = np.load(args.mps_file)
    bin_sizes = [3,7,5,10,6,8,4,9,11,3,12,6,8,7,5,6,8,10]
    num_shots = len(bin_sizes)
    img_size = ground_truth.shape[0]
    exp_device = sp.Device(args.device)
    rotation_sample_range = args.rot_sample_range
    translation_sample_range = args.trans_sample_range
    num_recon_shots = args.num_recon_shots

    #Generate a transform to apply, sample uniformly from a specified range
    rng = np.random.default_rng(123456789)
    rotations = rng.uniform(-(rotation_sample_range/2), (rotation_sample_range/2), (num_shots,3)) * np.pi / 180
    translations = rng.uniform(-(translation_sample_range/2), (translation_sample_range/2), (num_shots,3))
    transforms = np.hstack((translations, rotations))
    transforms -= np.mean(transforms, axis=0)

    #Create a shot mask that samples for each shot and simulates and accel factor for aliasing
    regular_mask = np.zeros(ground_truth.shape, dtype=bool)
    regular_mask[::args.accel_factor] = True

   
    uneven_mask = np.zeros([len(bin_sizes), *ground_truth.shape], dtype=bool)
    offset = 0
    for s,b in enumerate(bin_sizes):
        uneven_mask[s, offset: offset+b] = regular_mask[offset:offset+b]
        offset = offset + b

    #Generate corrupted kspace and save corrupted image
    kgrid, rkgrid = compute_transform_grids(ground_truth.shape)
    S = sp.mri.linop.Sense(mps)
    ksp = 0
    for s in range(len(bin_sizes)):
        A = sp.linop.Multiply(S.oshape, uneven_mask[s])
        T = RigidTransform(ground_truth.shape, ground_truth.shape, transforms[s], kgrid, rkgrid)
        ksp += (A * S * T * ground_truth)
    

    corrupted_image = S.H * ksp

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

    master_mask = (sp.rss(ksp, axes=(0,)) > 0).astype(bool)
    shot_size = img_size // num_recon_shots
    shot_mask = np.zeros((num_recon_shots, *ground_truth.shape), dtype=bool)
    for s in range(num_recon_shots):
        shot_mask[s, shot_size*s:(s+1)*shot_size] = master_mask[shot_size*s:(s+1)*shot_size]


    kgrid, rkgrid = compute_transform_grids(ground_truth.shape, device=exp_device)
    recon, estimates = JointEstimation(ksp, mps, shot_mask, kgrid, rkgrid, 
                            device=exp_device, P=P, constraint=M, img=init_image_guess, 
                            max_joint_iter=args.max_iter, tol=1e-12).run()
    
    recon = recon.get()
    estimates = estimates.get()
    experiment_name = f"{num_recon_shots}_shot_recon_for_uneven_bin_corruption"

    np.save(f"{args.output_dir}/recon_{experiment_name}", recon)
    np.save(f"{args.output_dir}/sense_recon_for_uneven_bin_corruption", sense_recon)
    
    nrmse_sense_recon = compute_nrmse(ground_truth, sense_recon)
    nrmse_joint_recon = compute_nrmse(ground_truth, recon)
    ssim_sense_recon = compute_ssim(np.abs(ground_truth), np.abs(sense_recon))
    ssim_joint_recon = compute_ssim(np.abs(ground_truth), np.abs(recon))
    nrmse_estimates = compute_nrmse(transforms, estimates) if num_recon_shots == num_shots else None

    # Get SLURM job ID (or use a timestamp if running locally)
    #job_id = os.getenv("SLURM_JOB_ID", "local")  # Default to "local" if not on SLURM

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
        for s in range(num_recon_shots): 
            log_file.write(f"Shot #{s+1}: ")
            log_file.write(",  ".join(f"{val:.3f}" for val in estimates[s]) + "\n")

    print(f"Log file saved as {log_filename}")