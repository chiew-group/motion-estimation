import argparse
from pathlib import Path
from math import prod

import numpy as np
import cupy as cp
import sigpy as sp
import sigpy.mri
import matplotlib.pyplot as plt
import nibabel as nib
import sigpy.plot as pl


from image_estimation import estimate_image_cg
from transform import RigidTransformCudaOptimzied
from utils import compute_transform_grids_voxel, show_mid_slices
from pyramid import pyramid_reconstruction

from dataclasses import dataclass
from math import ceil
import random
from typing import List, Tuple

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

@dataclass
class DisorderConfig:
    max_line_number: int           # original (0-based) max line index
    max_partition_number: int      # original (0-based) max partition index
    pat_lines_to_measure: int      # lines actually measured per partition
    block_lin: int                 # lBlockLin
    block_par: int                 # lBlockPar
    seed: int = 123456             # match std::mt19937(123456)

def make_disorder_order(cfg: DisorderConfig) -> List[Tuple[int, int]]:
    """
    Returns a list of (line, partition) pairs of length NoOfReorderIndices
    that replicates the tiling + within-tile shuffle + global interleave logic.
    """
    # 1) Pad to block multiples
    L = ceil((cfg.max_line_number + 1) / cfg.block_lin) * cfg.block_lin  # total lines after padding
    P = ceil((cfg.max_partition_number + 1) / cfg.block_par) * cfg.block_par  # total partitions after padding
    max_line_number = L - 1
    max_partition_number = P - 1

    # 2) Count indices (like setNoOfReorderIndices(getPATLinesToMeasure() * (MaxPartition+1)))
    no_of_reorder_indices = cfg.pat_lines_to_measure * (max_partition_number + 1)

    # 3) Groups/tiles
    max_indices_per_group = cfg.block_lin * cfg.block_par
    num_lin_groups = L // cfg.block_lin
    num_par_groups = P // cfg.block_par
    num_groups = num_lin_groups * num_par_groups
    indices_per_group = no_of_reorder_indices // num_groups  # integer division, like in the C++

    # 4) Prepare the shuffled local indices for a tile
    rng = random.Random(cfg.seed)
    base_idx = list(range(max_indices_per_group))  # 0..(block_lin*block_par-1)

    # 5) Allocate output (line, partition) per reorder index
    out = [None] * (indices_per_group * num_groups)

    # Interleave by j (the within-tile draw index), then by (linG, parG)
    for linG in range(num_lin_groups):
        for parG in range(num_par_groups):
            idx = base_idx[:]              # copy
            rng.shuffle(idx)               # shuffle anew for each group, same seed stream as mt19937
            for j in range(indices_per_group):
                local = idx[j]
                local_lin = local % cfg.block_lin
                local_par = local // cfg.block_lin

                line =  local_lin + (linG % cfg.block_par)*(max_line_number+1)//cfg.block_par + (linG//cfg.block_par) * cfg.block_lin
                part =  local_par + parG * cfg.block_par 

                out_pos = j * num_groups + linG * num_par_groups + parG
                out[out_pos] = (line, part)

    return out, (L,P)




def main():
    """
    Main entry for DISORDER 3D synthetic motion experiment sweep.

    Command-line arguments (via argparse):

    --ground (str, required): Path to the ground-truth `.npy` file containing a 3D complex image.
    --mps (str, required): Path to the sensitivity maps `.npy` file (expected shape: coils x H x W x D).
    --out_dir (str, required): Output directory to save results (nifti files, slices, and results .npz).
    --ord (str, default='disorder'): Sampling order to use. Supported: 'disorder' or 'sequential'.
    --tile_size (int int, required): Two integers specifying tile shape (Partitions Lines). Must match shot count.
    --accel (int, default=2): Acceleration (undersampling) factor applied to k-space sampling.
    --seed (int, default=42): Random seed for sampling/order generation.
    --iters (int, default=200): Number of joint reconstruction iterations passed to `pyramid_reconstruction`.
    --no (int, default=0): Noise multiplier; if >0 complex Gaussian noise (scaled from k-space max) is added.
    --mask (str, default=None): Optional path to a mask `.npy` file to override generated sampling masks.
    --transforms (str, default=None): Optional path to transforms `.npy` file (not required for default flow).
    --low_freq_var (int, default=1): Controls low-frequency variation magnitude when generating motion.
    --continuous (flag): If set, uses a continuous corruption length for corruption mask (affects nshots_corruption).

    Notes:
    - `--tile_size` should be provided as two integers (e.g. `--tile_size 8 8`).
    - File paths are expected to be NumPy `.npy` files compatible with the script's loaders.
    """
    ap = argparse.ArgumentParser(description="DISORDER 3D synthetic motion experiment sweep")    
    ap.add_argument(
        "--ground",
        required=True,
        help="Path to ground-truth .npy file containing a 3D complex image",
    )
    ap.add_argument(
        "--mps",
        required=True,
        help="Path to sensitivity maps .npy file (coils x H x W [, D]).",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        type=str,
        help="Output directory to save results (nifti, slices, and results .npz).",
    )
    ap.add_argument(
        "--ord",
        default='disorder',
        type=str,
        help="Sampling order: 'disorder' (default) or 'sequential' (line-wise).",
    )
    ap.add_argument(
        "--tile_size",
        type=int,
        nargs='+',
        help="Tile shape as two integers: (Partitions Lines) (e.g. --tile_size 8 8).",
    )
    ap.add_argument(
        "--accel",
        type=int,
        default=2,
        help="Acceleration (undersampling) factor applied to k-space sampling (default: 2).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling/order generation (default: 42).",
    )
    ap.add_argument(
        "--iters",
        type=int,
        default=200,
        help="Number of joint reconstruction iterations passed to pyramid_reconstruction (default: 200).",
    )
    ap.add_argument(
        "--no",
        type=int,
        default=0,
        help=(
            "Noise multiplier; if >0 adds complex Gaussian noise scaled from k-space max. "
            "Used as std = max(ksp) * 0.0001 * <no>.")
    )
    ap.add_argument(
        "--mask",
        type=str,
        default=None,
        help=(
            "Optional path to a mask .npy file to override generated sampling masks. "
            "Mask will be broadcast/expanded to match expected dimensions."),
    )
    ap.add_argument(
        "--transforms",
        type=str,
        default=None,
        help="Optional path to a transforms .npy file with precomputed motion parameters.",
    )
    ap.add_argument(
        "--low_freq_var",
        type=int,
        default=1,
        help="Low-frequency variance for motion generation (controls drift/wobble magnitude).",
    )
    ap.add_argument(
        "--continuous",
        action='store_true',
        help=(
            "If set, uses a continuous corruption length for corruption masks (affects nshots_corruption)."
        ),
    )
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

    assert gt.shape[0] % tile_size[0] == 0
    assert gt.shape[1] % tile_size[1] == 0
    nshots = prod(tile_size) // args.accel
    nshots_corruption = gt.shape[1] // args.accel if args.continuous else nshots
    print(f"Number of corruptions shots: {nshots_corruption}")
    print(f"Number of shots: {nshots} with tiles of size {tile_size} and acceleration factor {args.accel}")
    K = gt.shape[:2]
    #If we want to use our own mask choose it here
    if args.mask is not None:
        A = np.load(args.mask)
        A = A[..., None]
        corruption_A = A
    else:
        if args.ord == 'disorder':
            # Example usage:
            cfg = DisorderConfig(
                max_line_number=gt.shape[1]-1,         # e.g. 224 lines (0..223) before padding
                max_partition_number=gt.shape[0]-1,     # e.g. 64 partitions (0..63) before padding
                pat_lines_to_measure=gt.shape[1]//args.accel,    #PAT lines aquired, correct config calculation is lines/acceleration factor
                block_lin=tile_size[1],                 # tile lines
                block_par=tile_size[0],                 # tile partitions
                seed=123456
            )
            order, (L,P) = make_disorder_order(cfg)
            order = np.array(order, dtype=int)

            #Create the reconstruction mask by binning a certain amount of samples per motion state
            A = np.zeros((nshots, *K), dtype=bool)
            samples_per_shot = order.shape[0] // nshots
            for s in range(nshots):
                start = s*samples_per_shot
                end   = (s+1)*samples_per_shot
                A[s, order[start:end, 1], order[start:end, 0]] = True
            
            #Likewise we create the corruption mask but it will be done at a higher number of states
            corruption_A = np.zeros((nshots_corruption, *A.shape[1:]), dtype=bool)
            samples_per_shot = len(order) // nshots_corruption
            for s in range(nshots_corruption):
                start = s*samples_per_shot
                end   = (s+1)*samples_per_shot
                corruption_A[s, order[start:end, 1], order[start:end, 0]] = True

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

    corruption_transforms = generate_motion_parameters_new(nshots_corruption, low_freq_var=args.low_freq_var, high_freq_var=20.0)

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

    ksp = cp.array(ksp)

    p = 1 / (cp.sum(cp.abs(mps) ** 2, axis=0) + 1e-6) #preconditioner
    M = (cp.sum(cp.abs(mps) ** 2, axis=0) > 0.1) #mask to zero out anything outside the brain
    P = sp.linop.Multiply(ksp.shape[1:], p)
    uncorrected = sp.mri.app.SenseRecon(ksp, mps, P=P, device=sp.Device(0), tol=1e-12).run()
    uncorrected = cp.asnumpy(uncorrected)
    gt = cp.asnumpy(gt)

    #Pyramid reconstruction will internally handle all the shifting and unshifting required for recon simply pass arguments in
    corrected, t_estimates = pyramid_reconstruction(ksp, mps, A, nshots, 
                                                    n_joint_iters=args.iters, 
                                                    save_path=out)
    
    #The corrected recon and transform estimates will have be sent back to the cpu already

    err_volume = np.real(np.abs(corrected - gt))
    err_img = np.linalg.norm(err_volume)

    #We save the corrected and uncorrected as compressed nifti files to aquires QA scores
    nib.save(nib.Nifti1Image(np.abs(corrected).astype(np.float64), np.eye(4)), out / "corrected.nii.gz")
    nib.save(nib.Nifti1Image(np.abs(uncorrected).astype(np.float64), np.eye(4)), out / "uncorrected.nii.gz")

    #Save the midslices as a sneak peek png file
    show_mid_slices(uncorrected, save_path=out/ "uncorrected_slices.png")
    show_mid_slices(corrected, save_path=out / "corrected_slices.png")

    #Save the important simulation results and errors
    np.savez(out / "results", 
             corrected=np.abs(corrected), 
             uncorrected=np.abs(uncorrected),
             t_estimates=t_estimates,
             err_volume=err_volume, 
             err_img=err_img)
    

if __name__ == '__main__':
    main()