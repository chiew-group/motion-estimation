import argparse
import os
import numpy as np
import sigpy as sp
import sigpy.mri as mri

from joint_estimation import JointEstimation
from image_estimation import ImageEstimation
from utils import compute_transform_grids, generate_shot_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint estimatino in a multi resolution pymarid')
    parser.add_argument('ksp', type=str, help='Path to k-space data')
    parser.add_argument('mps', type=str, help='Path to sensitivity maps')
    parser.add_argument('outdir', type=str, help='Output directory')
    parser.add_argument('--num_bins', type=int, default=64, help='Number of bins')
    parser.add_argument('--iter_2', type=int, default=1000, help='Maximum number of iterations at resolution level 2')
    parser.add_argument('--iter_1', type=int, default=1000, help='Maximum number of iterations at resolution level 1')
    parser.add_argument('--iter_0', type=int, default=1000, help='Maximum number of iterations at resolution level 0')
    parser.add_argument('--device_id', type=int, default=0, help='Device to computer on. -1 is CPU, 0 is GPU')
    args = parser.parse_args()
    
    full_ksp = np.load(args.ksp)
    full_mps = np.load(args.mps)
    nc = full_ksp.shape[0]
    ns = args.num_bins
    img_shape = full_ksp.shape[1:]
    device = sp.Device(args.device_id)
    iter_2 = args.iter_2
    iter_1 = args.iter_1
    iter_0 = args.iter_0

    #Generate the sampling mask from numer of bins wanted
    desired_img_axis = 1
    bin_size = img_shape[desired_img_axis] // args.num_bins
    full_shot_mask = generate_shot_mask(bin_size, img_shape, desired_img_axis)

    print("Resolution level 2!")
    downsample_factor = 4
    downsample_shape = [nc, img_shape[0] // downsample_factor, img_shape[1], img_shape[2] // downsample_factor]
    ksp = sp.resize(full_ksp, downsample_shape)
    mps = sp.ifft(sp.resize(sp.fft(full_mps, axes=[-3,-2,-1]), downsample_shape), axes=[-3,-2,-1])
    shot_mask = sp.resize(full_shot_mask, [ns] + [*downsample_shape[1:]])
    print(f"ksp->{ksp.shape}, mps->{mps.shape}, mask->{shot_mask.shape}")

    rss = sp.rss(mps, axes=(0,))
    M = rss > np.max(rss) * 0.4
    P = sp.linop.Multiply(ksp.shape[1:], 1 / (np.sum(np.abs(mps) ** 2, axis=0) + 1e-3))
    
    kgrid, rkgrid = compute_transform_grids(img_shape, downsample_shape[1:], device=device)

    sense = mri.app.SenseRecon(ksp, mps, device=device).run()
    
    est_img, est_transforms = JointEstimation(ksp, mps, shot_mask, kgrid, rkgrid, 
                                              P=P, constraint=M, device=device,
                                              max_joint_iter=iter_2, max_nm_iter=0, max_cg_iter=100).run()
    
    np.save(os.path.join(args.outdir, 'recon_lvl2.npy'), est_img.get())
    np.save(os.path.join(args.outdir, 'sense_lvl2.npy'), sense.get())


    print("Resolution level 1!")
    downsample_factor = 2
    downsample_shape = [nc, img_shape[0] // downsample_factor, img_shape[1], img_shape[2] // downsample_factor]
    ksp = sp.resize(full_ksp, downsample_shape)
    mps = sp.ifft(sp.resize(sp.fft(full_mps, axes=[-3,-2,-1]), downsample_shape), axes=[-3,-2,-1])
    shot_mask = sp.resize(full_shot_mask, [ns] + [*downsample_shape[1:]])
    print(f"ksp->{ksp.shape}, mps->{mps.shape}, mask->{shot_mask.shape}")

    rss = sp.rss(mps, axes=(0,))
    M = rss > np.max(rss) * 0.4
    P = sp.linop.Multiply(ksp.shape[1:], 1 / (np.sum(np.abs(mps) ** 2, axis=0) + 1e-3))
    
    kgrid, rkgrid = compute_transform_grids(img_shape, downsample_shape[1:], device=device)

    sense = mri.app.SenseRecon(ksp, mps, device=device).run()

    est_img, est_transforms = JointEstimation(ksp, mps, shot_mask, kgrid, rkgrid, 
                                              P=P, constraint=M, transforms=est_transforms, 
                                              device=device, 
                                              max_joint_iter=iter_1).run()
    
    np.save(os.path.join(args.outdir, 'recon_lvl1.npy'), est_img.get())
    np.save(os.path.join(args.outdir, 'sense_lvl1.npy'), sense.get())

    print("Resolution level 0!")

    rss = sp.rss(full_mps, axes=(0,))
    M = rss > np.max(rss) * 0.4
    P = sp.linop.Multiply(full_ksp.shape[1:], 1 / (np.sum(np.abs(full_mps) ** 2, axis=0) + 1e-3))
    kgrid, rkgrid = compute_transform_grids(img_shape, device=device)
    sense = mri.app.SenseRecon(full_ksp, full_mps, device=device).run()
    
    est_img = ImageEstimation(full_ksp, full_mps, full_shot_mask, est_transforms, 
                              kgrid, rkgrid, P=P, constraint=M, device=device,
                              max_iter=iter_0).run()
    
    np.save(os.path.join(args.outdir, 'recon_lvl0.npy'), est_img.get())
    np.save(os.path.join(args.outdir, 'sense_lvl0.npy'), sense.get())