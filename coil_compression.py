import argparse
import numpy as np
import cupy as cp
from pathlib import Path

def coil_compression(mps, ksp=None, perc=0.95, reg=1e-6):
    
    #Choose what device we are on
    xp = cp.get_array_module(mps)


    num_coils = mps.shape[0]
    img_shape = mps.shape[1:]

    mps_conj = mps.conj()
    norm = 1 / (np.sum(mps * mps.conj(), axis=0) + reg)

    # Compute RSS normalization
    #rss = xp.sum(xp.abs(mps)**2, axis=0)
    #norm = 1 / (rss + reg)

    #mps = xp.reshape(mps, (num_coils, -1)) * norm.flatten()
    #mps_conj = xp.reshape(mps.conj(), (num_coils, -1)).T
    mps = mps.reshape((num_coils, -1))
    mps_conj = mps.conj().reshape((num_coils, -1))
    P = (mps * norm.flatten()) @ mps_conj.T
    #P = mps @ mps_conj
    #Threshold the singular values and vectors
    U, S, _ = xp.linalg.svd(P)
    #singular_total = np.sum(S)
    #num_reduced_coils = num_coils - ((np.cumsum(S) / singular_total) >= perc).sum()

    energy = xp.cumsum(S) / xp.sum(S)
    num_reduced_coils = xp.argmax(energy >= perc)
    print(f'Reduced {num_coils} coils -> {num_reduced_coils} coils')

    #Make the compression matrix
    A = U.conj().T[:num_reduced_coils]
    cc_mps = (A @ mps).reshape(num_reduced_coils, *img_shape)
    #A = xp.reshape(num_reduced_coils, *img_shape)
    if ksp is not None:
        cc_ksp = (A @ ksp.reshape(num_coils, -1)).reshape(num_reduced_coils, *img_shape)
    else:
        cc_ksp = None

    return cc_mps, cc_ksp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Coil compression for k-space and maps.")
    parser.add_argument('--mps', type=str, required=True, help='Path to sensitivty maps .npy file')
    parser.add_argument('--ksp', type=str, default=None, help='Optional path to k-space .npy file')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory')
    parser.add_argument('--perc', type=float, default=0.95, help='Variance capture threshold (default: 0.95)')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    mps = np.load(args.mps)
    ksp = np.load(args.ksp) if args.ksp else None

    # Run compression
    compressed_mps, compressed_ksp = coil_compression(mps, ksp=ksp, perc=args.perc)
    import sigpy.plot as pl
    pl.ImagePlot(compressed_mps)
    # Save outputs
    np.save(outdir / 'cc_mps.npy', compressed_mps)
    if compressed_ksp is not None:
        np.save(outdir / 'cc_ksp.npy', compressed_ksp)

    print(f"Saved compressed maps to {outdir/'cc_mps.npy'}")
    if compressed_ksp is not None:
        print(f"Saved compressed k-space to {outdir/'cc_ksp.npy'}")