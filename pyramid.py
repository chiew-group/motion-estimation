import numpy as np
import cupy as cp
import sigpy as sp
from utils import compute_transform_grids_voxel
from joint_recon import JointRecon
from image_estimation import estimate_image_cg
import sigpy.plot as pl
import matplotlib.pyplot as plt

def _plot_convergence(loss, title="", save=None):
    plt.figure(figsize=(8,8))
    plt.plot(np.log10(loss), marker='o', color='black')
    plt.ylabel('Log10 Loss')
    plt.xlabel('Iterations')
    plt.grid(True)
    plt.title(title)

    if save:
        plt.savefig(save)
    else:
        plt.show()


def pyramid_reconstruction(ksp, mps, A, nshots, n_spatial_levels=3, n_temporal_levels=3, n_joint_iters=100, save_path=None, sampling='disorder'):
    """
    Perform multi-resolution (pyramid) joint reconstruction for motion-corrupted k-space.

    Parameters
    - ksp (array-like): Input k-space array with shape (ncoils, H, W, [D]) or similar. Can be NumPy or CuPy array; the function will move data to CuPy for final processing.
    - mps (array-like): Sensitivity maps with shape compatible with `ksp` (typically (ncoils, H, W, [D])).
    - A (array-like): Sampling mask with shape (nshots, H, W, 1) (boolean-like). The first dimension indexes motion states/shots.
    - nshots (int): Number of motion states / shots represented in `A`.
    - n_spatial_levels (int, optional): Number of spatial pyramid levels (default: 3). Coarser levels run joint motion estimation.
    - n_temporal_levels (int, optional): Number of temporal aggregation levels (currently reserved/unused; default: 3).
    - n_joint_iters (int, optional): Base number of joint reconstruction iterations at full resolution. This value is scaled at coarser levels (default: 100).
    - save_path (path-like or None, optional): If provided, convergence plots are saved under this path.
    - sampling (str, optional): Sampling mode used when downsampling. Supported values: 'disorder' (default) or 'sequential'. Affects downsampled image shape handling.

    Returns
    - x (np.ndarray): Reconstructed image at full resolution (NumPy array). Complex-valued image returned in image-space and centered.
    - t_est (np.ndarray): Estimated transforms, shape (nshots, 6), dtype float. Each row contains [tx, ty, tz, rx, ry, rz] (rotations in radians).

    Notes
    - The function internally converts arrays to CuPy for GPU-accelerated operations and back to NumPy for return values.
    - At coarse pyramid levels, a `JointRecon` instance is used to jointly estimate image and transforms; final stage performs image-only estimation via `estimate_image_cg`.
    """

    #Initialize
    img_full_res = ksp.shape[1:]
    x_est = np.zeros(img_full_res, dtype=ksp.dtype)
    t_est = np.zeros((nshots, 6), dtype=float)
    ncoils = ksp.shape[0]

    #Multi-resolution loop
    for sl in range(n_spatial_levels-1):
        down_factor = 2 ** (n_spatial_levels - sl - 1)
        if sampling == 'disorder':
            ds_img_shape = (int(img_full_res[0] // down_factor), int(img_full_res[1]//down_factor), img_full_res[2])
        elif sampling == 'sequential':
            #When using a seq sampling scheme we have to crop only across the axis that contains all motion states info
            ds_img_shape = (int(img_full_res[0] // down_factor), img_full_res[1], img_full_res[2])
        print(f"Spatial Level {sl+1}/{n_spatial_levels} Downsampling by a factor of {down_factor}")
        #Downsample image, maps, ksp, sampling mask and transform grids

        mps_res = sp.ifft(sp.resize(sp.fft(mps, axes=(-3,-2,-1)), (ncoils, *ds_img_shape)), axes=(-3,-2,-1))
        ksp_res = sp.resize(ksp, (ncoils, *ds_img_shape))
        A_res   = sp.resize(A, (nshots, *ds_img_shape[:2], 1))

        print(f"Resizing ksapce from {ksp.shape} -> {ksp_res.shape}")
        print(f"Resizing sense maps from {mps.shape} -> {mps_res.shape}")
        print(f"Resizing sampling mask from {A.shape} -> {A_res.shape}")
        kgrid, rkgrid = compute_transform_grids_voxel(img_full_res, [1,1,1], ds_img_shape, xp=cp)
        app = JointRecon(ksp_res, mps_res, A_res, kgrid, rkgrid, t0=t_est, max_joint_iter=int(n_joint_iters * down_factor), xp=cp)
        recon, t_est = app.run()
    
        _plot_convergence(app.objective_history, 
                          f"Objective loss at [1/{down_factor}] resolution", 
                          save=save_path / f"loss_{down_factor}_factor.png")

        #TODO Temporal levels, this is experimental and needs to be further tested
        """
        t_levels = n_temporal_levels if (sl == n_spatial_levels - 1) else 1
        for t_lvl in range(t_levels):
            print(f"Temporal Level {t_lvl+1}/{t_levels}")

            #Aggregate motion states only at coarser temporal levels
            if t_lvl < t_levels - 1:
                temporal_factor = 2 ** (t_levels - t_lvl - 1)
                A_eff = aggregate_sampling_masks(A_res, temporal_factor)
                T_eff = aggregate_transforms(t_est, temporal_factor)
            else:
                A_eff = A_res
                T_eff = t_est
            print(f"Temporal Factor:: {A_eff.shape}, {T_eff.shape}")
        """
    print("Final level of the pyramid - full resollution, image estimation only!")

    #Image recon only at the highest spatial level
    ksp = cp.fft.ifftshift(cp.array(ksp),axes=(-3,-2,-1)).astype(cp.complex64)
    mps = cp.fft.ifftshift(cp.array(mps), axes=(-3,-2,-1)).astype(cp.complex64)
    A = cp.fft.ifftshift(cp.array(A), axes=(-3,-2,-1)).astype(bool)
    max_norm = cp.max(cp.abs(cp.fft.ifftn(ksp, axes=(-3,-2,-1), norm='ortho')))
    ksp /= max_norm
    x = cp.zeros(img_full_res, dtype=ksp.dtype)
    kgrid, rkgrid = compute_transform_grids_voxel(img_full_res, [1,1,1], None, xp=cp)

    P = 1 / (cp.sum(cp.abs(mps) ** 2, axis=0) + 1e-3)
    #self.P = cp.ones(self.img_shape)
    M = (cp.sum(cp.abs(mps) ** 2, axis=0) > 0.1)
    #self.M = cp.ones(self.img_shape)

    x = estimate_image_cg(ksp, mps, A, t_est, kgrid, rkgrid, P, M, x0=x)

    x = cp.asnumpy(cp.fft.fftshift(x * max_norm))
    t_test = cp.asnumpy(t_est)

    print("Pyramid Recon Complete!")
    return x, t_test

def aggregate_sampling_masks(A, bin_factor):
    nshots = A.shape[0]
    nbins = nshots // bin_factor
    return A.reshape((nbins, bin_factor, *A.shape[1:])).sum(1)

def aggregate_transforms(T, bin_factor):
    nshots = T.shape[0]
    nbins = nshots // bin_factor
    return T.reshape((nbins, bin_factor, 6)).sum(1)