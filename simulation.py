import argparse
from pathlib import Path
from math import prod

import numpy as np
import cupy as cp
import sigpy as sp
import matplotlib.pyplot as plt
import nibabel as nib
import sigpy.plot as pl


from image_estimation import estimate_image_cg
from transform import RigidTransformCudaOptimzied
from utils import compute_transform_grids_voxel, show_mid_slices
from pyramid import pyramid_reconstruction

def sample_disorder(K, U, R=1, seed=42):
    ky = np.arange(K[0])
    kx = np.arange(K[1])
    Ky, Kx = np.meshgrid(ky, kx, indexing='ij')
    k = np.stack([Ky.ravel(), Kx.ravel()], axis=-1).reshape(-1,2)  # shape (NP,2)
    rng = np.random.default_rng(seed)

    # 2) Choose your DISORDER parameters:
    nrow_tiles = K[0] // U[0]
    ncol_tiles = K[1] // U[1]
    tr = (k[:, 0] // U[0]).astype(int)
    tc = (k[:, 1] // U[1]).astype(int)
    ntotal_tiles = nrow_tiles * ncol_tiles
    tile_ids = tr * ncol_tiles + tc

    shot_ids = -np.ones(np.prod(K), dtype=int)
    nshots = np.prod(U) // R
    nsamples = np.prod(K) // R
    for t in range(ntotal_tiles):
        ids = np.flatnonzero(tile_ids == t)
        ids = rng.permutation(ids)[::R]
        
        if R < 1:
            bucket = np.arange(nshots, dtype=int)
            bucket = np.pad(bucket, (0, nshots), 'constant', constant_values=(-1,-1))
            shot_ids[ids] = bucket
        else:
            shot_ids[ids] = np.arange(nshots, dtype=int)

    temporal = []
    for s in range(nshots):
        temporal.append(np.flatnonzero(shot_ids == s))
    temporal = np.concatenate(temporal)

    y_order, x_order = np.unravel_index(temporal, K)
    y_order = y_order.astype(int)
    x_order = x_order.astype(int)


    return shot_ids, temporal, y_order, x_order

def generate_corrupted_kspace(gt_img, mps, A, transforms):
    xp = cp.get_array_module(gt_img)
    nshots = transforms.shape[0]

    #We are uncentering our inputs here because the rigid transform
    #operation works under those assumptions for now (may change in future)
    gt_img =    xp.fft.ifftshift(gt_img)
    mps =       xp.fft.ifftshift(mps, axes=(-3,-2,-1))
    A =         xp.fft.ifftshift(A, axes=(-3, -2 ,-1))

    kgrid, rkgrid = compute_transform_grids_voxel(gt_img.shape, [1,1,1], xp=xp)
    ksp = 0
    for s in range(nshots):
        temp = RigidTransformCudaOptimzied(transforms[s], kgrid, rkgrid).apply(gt_img)
        temp = temp * mps
        temp = xp.fft.fftn(temp, axes=(-3,-2,-1), norm='ortho')
        temp *= A[s]
        ksp += temp
    ksp = xp.fft.fftshift(ksp, axes=(-3,-2,-1))

    return ksp

def gradient_entropy(vol):
    """Simple gradient-entropy proxy (lower is sharper)."""
    xp = cp.get_array_module(vol)

    gx = xp.diff(vol, axis=0, append=vol[-1:, :, :])
    gy = xp.diff(vol, axis=1, append=vol[:, -1:, :])
    gz = xp.diff(vol, axis=2, append=vol[:, :, -1:])
    gmag = xp.sqrt(gx*gx + gy*gy + gz*gz)
    hist, _ = xp.histogram(gmag, bins=256, range=(0, gmag.max() + 1e-8), density=True)
    hist = hist + 1e-12
    return float(-xp.sum(hist * xp.log2(hist)))

def generate_complex_gaussian_noise(shape, std=1.0, seed=42, xp=np):
    rng = xp.random.default_rng(seed)
    n = std * rng.standard_normal(shape, dtype=xp.float32) + 1j * std * rng.standard_normal(shape, dtype=xp.float32)
    return n.astype(xp.complex64)

def generate_motion_parameters_new(
    num_states,
    low_freq_var=0.1,      # controls overall drift + wobble energy (bigger => more motion)
    high_freq_var=5.0,     # controls bump amplitudes (smaller than before; we use it gently)
    spike_prob=0.02,       # repurposed as "expected bumps / num_states"
    seed=123456789,
    rot_limit_deg=20.0,    # soft limit for rotations
):
    """
    Generate smooth, near-linear rigid-body motion with a gentle wobble and 1–2 small bumps.

    - First 3 cols: translations (mm)
    - Last 3 cols: rotations (radians)
    """
    rng = np.random.default_rng(seed)
    T = num_states
    t = np.linspace(0.0, 1.0, T)

    # ---- 1) Linear drift (dominant "more linear" look) -----------------------
    # Draw an end-point for each DOF and linearly interpolate 0 -> end.
    # Translations a bit larger than rotations.
    drift_end_trans = rng.normal(0, np.sqrt(low_freq_var), size=3)          # mm at t=1
    drift_end_rot_deg = rng.normal(0, 0.6*np.sqrt(low_freq_var), size=3)    # deg at t=1
    drift_trans = t[:, None] * drift_end_trans[None, :]
    drift_rot_deg = t[:, None] * drift_end_rot_deg[None, :]

    # ---- 2) Wobble: very smooth AR(1) (band-limited-ish) ---------------------
    # Choose strong correlation for smoothness.
    rho = 0.98
    def ar1(T, rho, target_var, size):
        # Select innovation std so stationary var ~= target_var
        innov_std = np.sqrt(max(1e-12, target_var) * (1 - rho**2))
        x = np.zeros((T, size))
        for i in range(1, T):
            x[i] = rho * x[i - 1] + rng.normal(0, innov_std, size=size)
        return x

    wobble_trans = ar1(T, rho, 0.3 * low_freq_var, size=3)          # mm
    wobble_rot_deg = ar1(T, rho, 0.2 * low_freq_var, size=3)        # deg

    # ---- 3) One or two "small bumps" (Gaussian pulses), not spikes ----------
    # Use spike_prob to set expected count, but clamp to {1,2}.
    expected = max(1, int(round(spike_prob * T)))
    n_bumps = int(np.clip(expected, 1, 2))
    centers = rng.uniform(0.15, 0.85, size=n_bumps)                 # avoid edges
    widths = rng.uniform(0.06, 0.14, size=n_bumps) * T              # ~6–14% of length

    # amplitudes: modest, scaled off high_freq_var (smaller than your spikes)
    amp_trans = rng.normal(0, 0.15*np.sqrt(high_freq_var), size=(n_bumps, 3))   # mm
    amp_rot_deg = rng.normal(0, 0.08*np.sqrt(high_freq_var), size=(n_bumps, 3)) # deg

    bumps_trans = np.zeros((T, 3))
    bumps_rot_deg = np.zeros((T, 3))
    grid = np.arange(T)
    for k in range(n_bumps):
        g = np.exp(-0.5 * ((grid - centers[k]*T) / (widths[k] + 1e-9))**2)[:, None]
        bumps_trans += g * amp_trans[k][None, :]
        bumps_rot_deg += g * amp_rot_deg[k][None, :]

    # ---- Combine components --------------------------------------------------
    trans_mm = drift_trans + wobble_trans + bumps_trans
    rot_deg = drift_rot_deg + wobble_rot_deg + bumps_rot_deg

    # Soft clip rotations to ±rot_limit_deg, then convert to radians
    limit = float(rot_limit_deg)
    rot_deg = np.clip(rot_deg, -limit, limit)
    rot_rad = np.deg2rad(rot_deg)

    # Stack (mm, mm, mm, rad, rad, rad)
    motion_parameters = np.hstack([trans_mm, rot_rad])
    return motion_parameters

def plot_transforms(transforms):
    fig, axs = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    fig.suptitle("Transforms")
    
    axs[0].plot(transforms[:,:3])
    axs[0].set_ylabel("Unit Vox")
    axs[0].grid(True)
    axs[0].legend(["x-axis", "y-axis", "z-axis"])

    axs[1].plot(transforms[:, 3:] * 180 / np.pi)
    axs[1].set_ylabel("Degrees")
    axs[1].set_xlabel("Motion States")
    axs[1].grid(True)
    axs[1].legend(["x-axis", "y-axis", "z-axis"])
    plt.show()
    

def main():
    ap = argparse.ArgumentParser(description="DISORDER 3D synthetic motion experiment sweep")    
    ap.add_argument("--ground", required=True, help="Ground-truth")
    ap.add_argument("--mps", required=True, help="Sensitivity maps")
    ap.add_argument("--out_dir", required=True, type=str, help="Output dir for all the results")
    ap.add_argument("--ord", default='disorder', type=str)
    ap.add_argument("--tile_size", type=int, nargs='+', help="Comma list of tile shape, needs to match shot count")
    ap.add_argument("--accel", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--no", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)

    #Load the input data for experiment
    gt  = np.load(args.ground).astype(np.complex64)
    mps = np.load(args.mps).astype(np.complex64)
    assert gt.ndim == 3
    assert mps.ndim == 4  and mps.shape[1:] == gt.shape

    tile_size = tuple(args.tile_size)
    assert len(tile_size) == 2

    nshots = prod(tile_size) // args.accel
    nshots_corruption = gt.shape[0] // args.accel
    print(f"Number of corruptions shots: {nshots_corruption}")
    print(f"Number of shots: {nshots} with tiles of size {tile_size} and acceleration factor {args.accel}")
    K = gt.shape[:2]

    if args.ord == 'disorder':

        #Sample coordinates using DISORDER then take up to the acceleration amount
        #R=2 means first half and R=4 means first quarter
        shot_ids, temporal, y, x = sample_disorder(K, tile_size, R=1, seed=args.seed)        
        y = y[:len(y)//args.accel]
        x = x[:len(x)//args.accel]

        #Create the reconstruction mask by binning a certain amount of samples per motion state
        A = np.zeros((nshots, *K), dtype=bool)
        samples_per_shot = y.shape[0] // nshots
        for s in range(nshots):
            start = s*samples_per_shot
            end   = (s+1)*samples_per_shot
            A[s, y[start:end], x[start:end]] = True
        
        #Likewise we create the corruption mask but it will be done at a higher number of states
        corruption_A = np.zeros((nshots_corruption, *A.shape[1:]), dtype=bool)
        samples_per_shot = len(y) // nshots_corruption
        for s in range(nshots_corruption):
            start = s*samples_per_shot
            end   = (s+1)*samples_per_shot
            corruption_A[s, y[start:end], x[start:end]] = True

    elif args.ord == 'sequential':

        #For the sequential we create the coordinates and for acceleration factors we skip lines
        y = np.repeat(np.arange(0,K[0],args.accel), K[1])
        x = np.tile(np.arange(K[1]), K[0]//args.accel)

        #Create the reconstruction mask like in the disorder case
        A = np.zeros((nshots, *K), dtype=bool)
        samples_per_shot = y.shape[0] // nshots
        for s in range(nshots):
            start = s*samples_per_shot
            end   = (s+1)*samples_per_shot
            A[s, y[start:end], x[start:end]] = True


        #Create the corruption mask
        corruption_A = np.zeros((nshots_corruption, *A.shape[1:]), dtype=bool)
        samples_per_shot = len(y) // nshots_corruption
        for s in range(nshots_corruption):
            start = s*samples_per_shot
            end   = (s+1)*samples_per_shot
            corruption_A[s, y[start:end], x[start:end]] = True

    #We have to add this extra dimension at the end so broadcasting can work in the recon algorithm
    #Not having data in the third dimension saves us some space and lets numpy do the work
    A = A[..., None]
    corruption_A = corruption_A[..., None]

    #pl.ImagePlot(A)
    #pl.ImagePlot(corruption_A)
    #pl.ImagePlot(np.sum(corruption_A, axis=0))

    corruption_transforms = generate_motion_parameters_new(gt.shape[0], low_freq_var=1, high_freq_var=20.0)

    #plot_transforms(corruption_transforms)

    gt  = cp.array(gt)
    mps = cp.array(mps)
    A   = cp.array(A)
    corruption_A = cp.array(corruption_A)
    #Here we make sure that our generated motion curve is at the highest number of shots for each simulation
    #Only from there we can chop it up to the amount of acceleration we want this way the same curve is used
    #each time
    corruption_transforms = cp.array(corruption_transforms[:nshots_corruption])

    #Create the motion corrupted kspace using the corruption mask
    #Add some complex gaussian noise if required
    ksp = generate_corrupted_kspace(gt, mps, corruption_A, corruption_transforms)
    if args.no > 0:
        std = cp.max(ksp) * 0.0001 * args.no
        noise = generate_complex_gaussian_noise(ksp.shape, std=std, xp=cp)
        ksp += noise

    #with_noise = cp.sum(mps.conj() * sp.ifft(ksp + noise, axes=(-3,-2,-1)), axis=0)
    #no_noise   = cp.sum(mps.conj() * sp.ifft(ksp, axes=(-3,-2,-1)), axis=0)
    
    #show_mid_slices(with_noise.get())
    #show_mid_slices(no_noise.get())


    ksp = cp.array(ksp)
    #We must uncenter the inputs to use image estimation since that is required form of input
    ksp = cp.fft.ifftshift(ksp, axes=(-3,-2,-1))
    mps = cp.fft.ifftshift(mps, axes=(-3,-2,-1))
    A   = cp.fft.ifftshift(A,   axes=(-3,-2,-1))

    kgrid, rkgrid = compute_transform_grids_voxel(gt.shape, [1,1,1], xp=cp)

    P = 1 / (cp.sum(cp.abs(mps) ** 2, axis=0) + 1e-6)
    M = (cp.sum(cp.abs(mps) ** 2, axis=0) > 0.1)
    uncorrected = estimate_image_cg(ksp, mps, A, cp.zeros((nshots, 6)), kgrid, rkgrid, P, M)
    uncorrected = cp.fft.fftshift(uncorrected)
    uncorrected = cp.asnumpy(uncorrected)
    gt = cp.asnumpy(gt)

    #We center the inputs again becuase the full pyramid recon handles the shifting for us
    ksp = cp.fft.fftshift(ksp, axes=(-3,-2,-1))
    mps = cp.fft.fftshift(mps, axes=(-3,-2,-1))
    A   = cp.fft.fftshift(A,   axes=(-3,-2,-1))


    corrected, t_estimates = pyramid_reconstruction(ksp, mps, A, nshots, 
                                                    n_joint_iters=args.iters, 
                                                    save_path=out)
    
    #The corrected recon and transform estimates will have be sent back to the cpu already

    #pl.ImagePlot(corrected)
    #pl.ImagePlot(uncorrected)
    
    err_volume = np.real(np.abs(corrected - gt))
    err_img = np.linalg.norm(err_volume)
    #err_t = np.linalg.norm(corruption_transforms-t_estimates)

    #We save the corrected and uncorrected as compressed nifti files to aquires QA scores
    nib.save(nib.Nifti1Image(np.abs(corrected), np.eye(4)), out / "corrected.nii.gz")
    nib.save(nib.Nifti1Image(np.abs(uncorrected), np.eye(4)), out / "uncorrected.nii.gz")

    #Save the midslices as a sneak peek png file
    show_mid_slices(uncorrected, save_path=out/ "uncorrected_slices.png")
    show_mid_slices(corrected, save_path=out / "corrected_slices.png")

    #Save the important simulation results and errors
    np.savez(out / "results", 
             corrected=corrected, 
             uncorrected=uncorrected,
             t_estimates=t_estimates,
             err_volume=err_volume, 
             err_img=err_img)


if __name__ == '__main__':
    main()