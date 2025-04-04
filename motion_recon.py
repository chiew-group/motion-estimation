from joint_estimation import JointEstimation
from image_estimation import ImageEstimation
import argparse
import numpy as np
import sigpy as sp
from utils import compute_transform_grids
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multiscale joint reconstruction with transform scaling')
    parser.add_argument('ksp_file', type=str, help='K-space file.')
    parser.add_argument('mps_file', type=str, help='Coil maps file.')
    parser.add_argument('output_dir', type=str)
    parser.add_argument('motion_states', type=int, help='Number of shots to group for motion estimate.')
    parser.add_argument('motion_axis', type=int, default=1, help='Axis along which motion is grouped.')
    parser.add_argument('--joint_iters', nargs='+', type=int, default=[10000, 500, 10], help='Joint iterations for each pyramid level.')
    parser.add_argument('--pyramid_levels', nargs='+', type=int, default=[4, 2], help='Downsampling factors for each pyramid level.')
    parser.add_argument('--prefix', type=str, default='', help='Prefix for output files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    ksp = np.load(args.ksp_file)
    mps = np.load(args.mps_file)
    full_sampling_mask = (sp.rss(ksp, axes=(0,)) > 0).astype(bool)

    full_res_img_shape = ksp.shape[1:]
    num_channels = ksp.shape[0]
    desired_axis = args.motion_axis
    num_shots = args.motion_states
    gpu = sp.Device(0)

    # Naive Sense recon for initial guess
    sense_recon = sp.mri.app.SenseRecon(ksp, mps, device=gpu).run().get()
    np.save(os.path.join(args.output_dir, f"{args.prefix}_sense.npy"), sense_recon)

    # Multiscale setup
    pyramid_factors = args.pyramid_levels
    coarsest_shape = [full_res_img_shape[i] // pyramid_factors[0] if i != desired_axis else full_res_img_shape[i] for i in range(3)]
    prev_recon = sp.ifft(sp.resize(sp.fft(sense_recon), coarsest_shape))
    prev_transforms = None
    all_recons = []
    all_estimates = []

    for i, factor in enumerate(pyramid_factors):
        downsample_shape = [full_res_img_shape[j] // factor if j != desired_axis else full_res_img_shape[j] for j in range(3)]
        shape_with_chan = [num_channels] + downsample_shape

        ds_ksp = sp.resize(ksp, shape_with_chan)
        ds_mps = sp.ifft(sp.resize(sp.fft(mps, axes=[-3,-2,-1]), shape_with_chan), axes=[-3,-2,-1])
        ds_mask = sp.resize(full_sampling_mask, downsample_shape)
        print(ds_mask.shape, downsample_shape, full_sampling_mask.shape)
        # Bin into shots
        shot_mask = np.zeros([num_shots] + downsample_shape, dtype=bool)
        shot_size = downsample_shape[desired_axis] // num_shots
        for shot in range(num_shots):
            sl = [slice(None)] * 3
            sl[desired_axis] = slice(shot * shot_size, (shot + 1) * shot_size)
            print(sl)
            print(f"shot {shot}: expected shape {shot_mask[tuple([shot] + sl)].shape}, actual {ds_mask[tuple(sl)].shape}")
            shot_mask[tuple([shot] + sl)] = ds_mask[tuple(sl)]

        # Scale previous transforms to current resolution
        if prev_transforms is not None:
            scale = [2 if j != desired_axis else 1 for j in range(3)]
            print(scale)
            #scale = factor / pyramid_factors[i - 1]  # upscale translations
            scaled_transforms = prev_transforms.copy()
            scaled_transforms[:, 0] *= scale[0]
            scaled_transforms[:, 1] *= scale[1]
            scaled_transforms[:, 2] *= scale[2]
        else:
            scaled_transforms = None

        # Grids
        kgrid, rkgrid = compute_transform_grids(downsample_shape, device=gpu)

        # Preconditioner & constraints
        rss = sp.rss(ds_mps, axes=(0,))
        M = rss > np.max(rss) * 0.1
        P = sp.linop.Multiply(ds_mps.shape[1:], 1 / (np.sum(np.abs(ds_mps) ** 2, axis=0) + 1e-3))

        max_iter = args.joint_iters[i] if i < len(args.joint_iters) else args.joint_iters[-1]

        # Recon
        recon, estimates = JointEstimation(ds_ksp, ds_mps, shot_mask, kgrid, rkgrid,
                                           device=gpu, P=P, constraint=M,
                                           img=prev_recon, transforms=scaled_transforms,
                                           max_joint_iter=max_iter, tol=1e-12).run()

        recon = sp.to_device(recon)
        estimates = sp.to_device(estimates)

        # Save this recon upsampled to full resolution for inspection
        recon_fullres = sp.ifft(sp.resize(sp.fft(recon), full_res_img_shape))
        all_recons.append(recon_fullres)

        # Also save quick PNG slice
        plt.imsave(os.path.join(args.output_dir, f"{args.prefix}_slice_level_{i}.png"), np.abs(recon_fullres[:, :, recon_fullres.shape[2] // 2]), cmap='gray')

        # Update recon for next level using next resolution's shape if not final
        if i + 1 < len(pyramid_factors):
            next_factor = pyramid_factors[i + 1]
            next_shape = [full_res_img_shape[j] // next_factor if j != desired_axis else full_res_img_shape[j] for j in range(3)]
            print(f"Next shape: {next_shape}")
            prev_recon = sp.ifft(sp.resize(sp.fft(recon), next_shape))
            prev_transforms = estimates.copy()

        all_estimates.append(estimates)

    #Ensure the we are using the full res data for the final image estimation
    prev_recon = sp.ifft(sp.resize(sp.fft(recon), full_res_img_shape))
    scale = [2 if j != desired_axis else 1 for j in range(3)]
    #scale = factor / pyramid_factors[i - 1]  # upscale translations
    scaled_transforms = prev_transforms.copy()
    scaled_transforms[:, 0] *= scale[0]
    scaled_transforms[:, 1] *= scale[1]
    scaled_transforms[:, 2] *= scale[2]
    prev_transforms = scaled_transforms.copy()

    # Final image estimation with full res data and estimated motion
    rss = sp.rss(mps, axes=(0,))
    M = rss > np.max(rss) * 0.1
    P = sp.linop.Multiply(mps.shape[1:], 1 / (np.sum(np.abs(mps) ** 2, axis=0) + 1e-3))
    kgrid, rkgrid = compute_transform_grids(full_res_img_shape, device=sp.cpu_device)

    final_img = ImageEstimation(ksp, mps, full_sampling_mask, prev_transforms, kgrid, rkgrid,
                                x=prev_recon, P=P, constraint=M,
                                device=sp.cpu_device, max_iter=args.joint_iters[-1], tol=1e-12).run()

    final_img = sp.to_device(final_img)
    all_recons.append(final_img)

    # Save outputs
    np.save(os.path.join(args.output_dir, f"{args.prefix}_final_image.npy"), final_img)
    for i, r in enumerate(all_recons):
        np.save(os.path.join(args.output_dir, f"{args.prefix}_recon_level_{i}.npy"), r)
    for i, t in enumerate(all_estimates):
        np.save(os.path.join(args.output_dir, f"{args.prefix}_transforms_level_{i}.npy"), t)
