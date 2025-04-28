from joint_estimation import JointEstimation
from image_estimation import ImageEstimation
import argparse
import numpy as np
import sigpy as sp
from utils import compute_transform_grids
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint estimation for motion corruption')
    parser.add_argument('ksp_file', type=str, help='K-space file.')
    parser.add_argument('mps_file', type=str, help='Coil maps file.')
    parser.add_argument('output_file', type=str)
    parser.add_argument('motion_states', type=int, help='Number of line scanes to group for motion estiamte.')
    parser.add_argument('motion_axis', type=int, default=1, help='0: Par, 1: Lin, 2: Col, used for sampling of states along an axis.')

    args = parser.parse_args()

    ksp = np.load(args.ksp_file)
    mps = np.load(args.mps_file)
    full_sampling_mask = (sp.rss(ksp, axes=(0,)) > 0).astype(bool)
    

    full_res_ksp_shape = ksp.shape
    full_res_img_shape = ksp.shape[1:]
    num_channels = ksp.shape[0]
    desired_axis = args.motion_axis
    num_shots = args.motion_states
    gpu = sp.Device(0)

    #Naive Sense recon for initial guess
    sense_recon = sp.mri.app.SenseRecon(ksp, mps, device=gpu).run()
    sense_recon = sense_recon.get()

    #Downsample at corsest resolution which is a factor of 4
    #Only at this resolution do we downsample the sense recon as it will be our initial guess
    downsample_factor = 4 
    downsample_shape = []
    for axis in range(3):
        if not axis == desired_axis:
            downsample_shape.append(full_res_img_shape[axis] // downsample_factor) 
        else:
            downsample_shape.append(full_res_img_shape[axis])
    downsample_shape = [num_channels] + downsample_shape

    coarse_ksp = sp.resize(ksp, downsample_shape)
    coarse_mps = sp.ifft(sp.resize(sp.fft(mps, axes=[-3,-2,-1]), downsample_shape), axes=[-3,-2,-1])
    coarse_mask = sp.resize(full_sampling_mask, downsample_shape[1:])
    low_res_sense_recon = sp.ifft(sp.resize(sp.fft(sense_recon), downsample_shape[1:]))

    #We have to bin the mask now into number of motion states, each with a calculated size
    shot_mask = np.zeros([num_shots] + downsample_shape[1:], dtype=bool)
    shot_size = downsample_shape[1+desired_axis] // num_shots
    for shot in range(num_shots):
        indicies = [slice(None)] * len(downsample_shape[1:])
        indicies[desired_axis] = slice(shot * shot_size, (shot+1) * shot_size)
        idx = [shot] + indicies
        shot_mask[*idx] = coarse_mask[*indicies]
    
    print(f"ksp shape: {ksp.shape}")
    print(f"mps shape: {mps.shape}")
    print(f"msk shape: {shot_mask.shape}")
    print(f"Resolution downsample factor: {downsample_factor}")
    print(f"Number of motion states is {num_shots} each of size {shot_size}")
    
    kgrid, rkgrid = compute_transform_grids(full_res_img_shape, downsample_shape[1:], device=gpu)
    #Set up constraints and preconditions
    rss = sp.rss(coarse_mps, axes=(0,))
    M = rss > np.max(rss) * 0.1
    P = sp.linop.Multiply(coarse_mps.shape[1:], 1 / (np.sum(np.abs(coarse_mps) ** 2, axis=0) + 1e-3))

    recon, estimates = JointEstimation(coarse_ksp, coarse_mps, shot_mask, kgrid, rkgrid, 
                                    device=gpu, P=P, constraint=M, img=low_res_sense_recon,
                                    max_joint_iter=10000, tol=1e-12).run()

    low_res_recon = sp.to_device(recon)
    low_res_estimates = sp.to_device(estimates)

    ########################################################
    #Downsample at medium resolution which is a factor of 2
    #Only at this resolution do we downsample the sense recon as it will be our initial guess
    downsample_factor = 2
    downsample_shape = []
    for axis in range(3):
        if not axis == desired_axis:
            downsample_shape.append(full_res_img_shape[axis] // downsample_factor) 
        else:
            downsample_shape.append(full_res_img_shape[axis])
    downsample_shape = [num_channels] + downsample_shape

    coarse_ksp = sp.resize(ksp, downsample_shape)
    coarse_mps = sp.ifft(sp.resize(sp.fft(mps, axes=[-3,-2,-1]), downsample_shape), axes=[-3,-2,-1])
    coarse_mask = sp.resize(full_sampling_mask, downsample_shape[1:])

    #We have to bin the mask now into number of motion states, each with a calculated size
    shot_mask = np.zeros([num_shots] + downsample_shape[1:], dtype=bool)
    shot_size = downsample_shape[1+desired_axis] // num_shots
    for shot in range(num_shots):
        indicies = [slice(None)] * len(downsample_shape[1:])
        indicies[desired_axis] = slice(shot * shot_size, (shot+1) * shot_size)
        idx = [shot] + indicies
        shot_mask[*idx] = coarse_mask[*indicies]
    
    print(f"ksp shape: {ksp.shape}")
    print(f"mps shape: {mps.shape}")
    print(f"msk shape: {shot_mask.shape}")
    print(f"Resolution downsample factor: {downsample_factor}")
    print(f"Number of motion states is {num_shots} each of size {shot_size}")
    
    kgrid, rkgrid = compute_transform_grids(full_res_img_shape, downsample_shape[1:], device=gpu)
    #Set up constraints and preconditions
    rss = sp.rss(coarse_mps, axes=(0,))
    M = rss > np.max(rss) * 0.1
    P = sp.linop.Multiply(coarse_mps.shape[1:], 1 / (np.sum(np.abs(coarse_mps) ** 2, axis=0) + 1e-3))

    low_res_recon = sp.ifft(sp.resize(sp.fft(low_res_recon), downsample_shape[1:]))

    recon, estimates = JointEstimation(coarse_ksp, coarse_mps, shot_mask, kgrid, rkgrid, 
                                    device=gpu, P=P, constraint=M, 
                                    img=low_res_recon, transforms=low_res_estimates,
                                    max_joint_iter=500, tol=1e-12).run()

    med_res_recon = sp.to_device(recon)
    med_res_estimates = sp.to_device(estimates)
    med_res_recon = sp.ifft(sp.resize(sp.fft(med_res_recon), full_res_img_shape))

    rss = sp.rss(mps, axes=(0,))
    M = rss > np.max(rss) * 0.1
    P = sp.linop.Multiply(mps.shape[1:], 1 / (np.sum(np.abs(mps) ** 2, axis=0) + 1e-3))

    kgrid, rkgrid = compute_transform_grids(full_res_img_shape, device=sp.cpu_device)
    joint_recon = ImageEstimation(ksp, mps, full_sampling_mask, med_res_estimates, kgrid, rkgrid,
                            x=med_res_recon, P=P, constraint=M, 
                            device=sp.cpu_device, max_iter=5, tol=1e-12).run()

    joint_recon = sp.to_device(joint_recon)

    np.save(args.output_file, joint_recon)