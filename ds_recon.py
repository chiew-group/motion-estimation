from utils import compute_transform_grids_voxel
from joint_recon import JointRecon

import numpy as np
import sigpy as sp
import cupy as cp
import matplotlib.pyplot as plt

import argparse
from pathlib import Path

def pad_to_square(img3d):
    x, y, z = img3d.shape
    target_size = max(x, y, z)

    pad_x = (target_size - x) // 2
    pad_y = (target_size - y) // 2
    pad_z = (target_size - z) // 2

    padded = np.pad(
        img3d,
        ((pad_x, target_size - x - pad_x),
         (pad_y, target_size - y - pad_y),
         (pad_z, target_size - z - pad_z)),
        mode='constant'
    )
    return padded

def show_mid_slices(img3d, save_path=None):
    img3d = np.abs(img3d)
    img3d = pad_to_square(img3d)

    x, y, z = img3d.shape
    mid_x, mid_y, mid_z = x // 2, y // 2, z // 2

    # Extract slices
    sagittal = np.rot90(np.rot90(np.flipud(img3d[mid_x+3, :, :])))     # Rotate 180 and flip vertically
    coronal  = np.rot90(np.rot90(img3d[:, mid_y, :]))      # Rotate 180 degrees
    axial    = img3d[:, :, mid_z]                          # No rotation

    slices = [sagittal, coronal, axial]

    # Plot without gaps
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=300)
    for ax, slc in zip(axs, slices):
        ax.imshow(slc.T, cmap='gray', origin='lower')
        ax.axis('off')

    # Remove white space between images
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved recon image midsections to {save_path}")
    else:
        plt.show()


def plot_joint_recon_summary(loss_list, t_list, shot_idx=0, save_path=None):
    """
    Plot joint recon objective and transform evolution for a fixed shot.

    Args:
        loss_list: list of scalar losses per iteration
        t_list: list of (num_shots, 6) arrays per iteration
        shot_idx: which shot's transforms to track
        save_path: optional path to save the figure
    """
    t_shot = t_list[:, shot_idx, :]  # shape = (iters, 6)
    iters = np.arange(len(loss_list))

    labels_mm = ['Tx (mm)', 'Ty (mm)', 'Tz (mm)']
    labels_rad = ['Rx (rad)', 'Ry (rad)', 'Rz (rad)']

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # 1. Objective function loss
    axs[0].plot(iters, np.log10(loss_list), marker='o', color='black')
    axs[0].set_ylabel("Objective")
    axs[0].set_title("Objective Function Over Iterations")
    axs[0].grid(True)

    # 2. Translation params
    for i in range(3):
        axs[1].plot(iters, t_shot[:, i], label=labels_mm[i])
        axs[1].text(
            iters[-1] + 0.5, t_shot[-1, i],
            f"{t_shot[-1, i]:.4f}", va='center', fontsize=9
        )
    axs[1].set_ylabel("Displacement (mm)")
    axs[1].set_title(f"Translation Parameters (Shot {shot_idx})")
    axs[1].legend()
    axs[1].grid(True)

    # 3. Rotation params
    for i in range(3, 6):
        axs[2].plot(iters, t_shot[:, i], label=labels_rad[i - 3])
        axs[2].text(
            iters[-1] + 0.5, t_shot[-1, i],
            f"{t_shot[-1, i]:.4f}", va='center', fontsize=9
        )
    axs[2].set_ylabel("Rotation (rad)")
    axs[2].set_title(f"Rotation Parameters (Shot {shot_idx})")
    axs[2].set_xlabel("Iteration")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved joint recon summary image of shot idx {shot_idx} to {save_path}")
    else:
        plt.show()

def crop_roi(ksp, mps):
    ksp = np.fft.ifftshift(ksp, axes=(-1,-3))
    ksp = np.fft.ifftn(ksp, axes=(-1,-3), norm='ortho')
    ksp = np.fft.fftshift(ksp, axes=(-1,-3))
    ksp = ksp[:, 25:-25, :, 50:250]

    ksp = np.fft.ifftshift(ksp, axes=(-1,-3))
    ksp = np.fft.fftn(ksp, axes=(-1,-3), norm='ortho')
    ksp = np.fft.fftshift(ksp, axes=(-1,-3))

    mps = mps[:, 25:-25, :, 50:250]
    return ksp, mps

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Coil compression for k-space and maps.")
    parser.add_argument('--ksp', type=str, required=True, help='Path to k-space .npy file')
    parser.add_argument('--mps', type=str, required=True, help='Path to sensitivty maps .npy file')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory')
    parser.add_argument('--shots', type=int, required=True, help='Number of motion states/shots to recon for')
    parser.add_argument('--t0', type=str, default=None, help='Initial transform parameters .npy file')
    parser.add_argument('--lowres', type=float, default=1.0, help='Downsample factor')
    parser.add_argument('--cg_iter', type=int, default=5, help='Max iterations for CG estimation')
    parser.add_argument('--nm_iter', type=int, default=1, help='Max iterations for NM estimation')
    parser.add_argument('--max_iter', type=int, default=100, help='Max iterations for joint recon')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ksp = np.load(args.ksp)
    mps = np.load(args.mps)
    t0 = np.load(args.t0) if args.t0 else None
    
    ksp, mps = crop_roi(ksp, mps)

    full_res = ksp.shape[1:]
    ncoils = ksp.shape[0]
    num_shots = args.shots
    shot_size = ksp.shape[2] // num_shots

    #Create sequential shot mask based on indexing axis 1
    #This can be modified later on as a todo
    #Also we can change this so as to input your own mask as a npy array
    rss_ksp = np.sum(np.abs(ksp)**2, axis=0, dtype=bool)
    shot_mask = np.zeros([num_shots, *ksp.shape[1:]], dtype=bool)
    for s in range(num_shots):
        shot_mask[s, :, s*shot_size:(s+1)*shot_size] = rss_ksp[:, s*shot_size:(s+1)*shot_size, :]
    voxel = [0.8,0.8,0.8]
    ds_shape = None

    if args.lowres > 1:
        #Resizing to downsample factor, if 1 things will be unchanged
        ds_shape = [int(full_res[0]//args.lowres), full_res[1], int(full_res[2]//args.lowres)]
        print(ds_shape)
        ksp = sp.resize(ksp, [ncoils] + ds_shape)
        mps = sp.ifft(sp.resize(sp.fft(mps, axes=(-3,-2,-1)), [ncoils] + ds_shape), axes=(-3,-2,-1))
        shot_mask = sp.resize(shot_mask, [num_shots] + ds_shape)
        print(f"ksp: {ksp.shape}, mps: {mps.shape}, mask: {shot_mask.shape}")
        voxel = [0.8*args.lowres,0.8, 0.8*args.lowres]

    #RECON
    kgrid, rkgrid = compute_transform_grids_voxel(full_res, voxel, ds_shape, xp=cp)
    app = JointRecon(ksp, mps, shot_mask, kgrid, rkgrid, t0=t0, max_cg_iter=args.cg_iter, max_nm_iter=args.nm_iter, max_joint_iter=args.max_iter, xp=cp)
    recon, t = app.run()

    #Resize image to full resoultion and move to cpu device
    if args.lowres > 1:
        recon = sp.ifft(sp.resize(sp.fft(recon, axes=(-3,-2,-1)), full_res), axes=(-3,-2,-1))
    recon = recon.get()
    t = t.get()

    #Plot summaries and save images
    img_file = f"recon_sh{num_shots}_ds{int(args.lowres)}.npy"
    viz_file = f"recon_sh{num_shots}_ds{int(args.lowres)}.png"
    tform_file = f"transforms_sh{num_shots}_ds{int(args.lowres)}.npy"
    summary_file = f"summary_sh{num_shots}_ds{int(args.lowres)}.png"

    plot_joint_recon_summary(app.objective_history, app.transform_history, num_shots//2, save_path=outdir / summary_file)
    show_mid_slices(pad_to_square(recon), outdir / viz_file)
    np.save(outdir / img_file, recon)
    np.save(outdir / tform_file, t)  # shape: (shots, iters, 6)