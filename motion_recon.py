from joint_estimation import JointEstimation
from image_estimation import ImageEstimation
import argparse
import numpy as np
import sigpy as sp
from utils import compute_transform_grids
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multiscale joint reconstruction with transform scaling')
    parser.add_argument('ksp_file', type=str, help='K-space file.')
    parser.add_argument('mps_file', type=str, help='Coil maps file.')
    parser.add_argument('out_dir', type=str)
    parser.add_argument('motion_states', type=int, help='Number of shots to group for motion estimate.')
    parser.add_argument('motion_axis', type=int, default=1, help='Axis along which motion is grouped.')
    parser.add_argument('voxel_size', type=float)
    parser.add_argument('--ds_factor', type=int, default=0)
    parser.add_argument('--max_iter', type=int, default=100)
    parser.add_argument('--x0', type=str)
    parser.add_argument('--t0', type=str)
    parser.add_argument('--gpu', type=bool, default=False, help='Prefix for output files')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    ksp = np.load(args.ksp_file)
    mps = np.load(args.mps_file)
    full_sampling_mask = (sp.rss(ksp, axes=(0,)) > 0).astype(bool)
    ds_factor = 2 ** args.ds_factor

    if args.x0 is not None:
        x0 = np.load(args.x0)
    else:
        x0 = None
    
    if args.t0 is not None:
        t0 = np.load(args.t0)
    else:
        t0 = None

    device = sp.Device(0) if  args.gpu else sp.Device(-1)

    print('-' * 50)
    print("Experiment Starting")
    print(f"Gpu usage: {args.gpu}")
    print(f"Downsampled by a factor of {ds_factor}")
    print(f"Number of motion states is: {args.motion_states} along axis {args.motion_axis}")
    print(f"Intial image: {args.x0}, Initial Transforms: {args.t0}, Max Iteations: {args.max_iter}, Voxel Size: {args.voxel_size}")
    img_shape = ksp.shape[1:]
    print(f"K-space intial shape -> {ksp.shape}, Mps intial shape -> {mps .shape}")
    num_channels = ksp.shape[0]
    desired_axis = args.motion_axis
    num_shots = args.motion_states

    downsample_shape = [img_shape[j] // ds_factor if j != desired_axis else img_shape[j] for j in range(3)]
    shape_with_chan = [num_channels] + downsample_shape
    print("Downsampling ... ")
    ksp = sp.resize(ksp, shape_with_chan)
    mps = sp.ifft(sp.resize(sp.fft(mps, axes=[-3,-2,-1]), shape_with_chan), axes=[-3,-2,-1])
    ds_mask = sp.resize(full_sampling_mask, downsample_shape)
    print(f"K-space ds shape -> {ksp.shape}, Mps ds shape -> {mps .shape}")
    shot_mask = np.zeros([num_shots] + downsample_shape, dtype=bool)
    shot_size = downsample_shape[desired_axis] // num_shots
    for shot in range(num_shots):
        sl = [slice(None)] * 3
        sl[desired_axis] = slice(shot * shot_size, (shot + 1) * shot_size)
        shot_mask[tuple([shot] + sl)] = ds_mask[tuple(sl)]

    voxel_size = [args.voxel_size] * 3
    kgrid, rkgrid = compute_transform_grids(img_shape, downsample_shape, voxel_size, device=device)

    # Preconditioner & constraints
    rss = sp.rss(mps, axes=(0,))
    M = rss > np.max(rss) * 0.3
    P = sp.linop.Multiply(mps.shape[1:], 1 / (np.sum(np.abs(mps) ** 2, axis=0) + 1e-3))

    # Recon
    recon, estimates = JointEstimation(ksp, mps, shot_mask, kgrid, rkgrid,
                                        device=device, P=P, constraint=M,
                                        img=x0, transforms=t0,
                                        max_joint_iter=args.max_iter, tol=1e-12).run()

    recon = sp.to_device(recon)
    estimates = sp.to_device(estimates)

    # Save this recon upsampled to full resolution for inspection
    recon_fullres = sp.ifft(sp.resize(sp.fft(recon), img_shape))
    
    np.save(os.path.join(args.out_dir, f"recon_{num_shots}shots_{ds_factor}dsfactor.npy"), recon_fullres)
    np.save(os.path.join(args.out_dir, f"transform_{num_shots}shots_{ds_factor}dsfactor.npy"), estimates)