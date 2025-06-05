import argparse
import numpy as np
from pathlib import Path

def noise_whiten(ksp, noise, reg=1e-6):
    num_coils = ksp.shape[0]
    noise = noise.reshape(num_coils, -1)
    cov = (noise @ noise.conj().T) / noise.shape[1]
    cov += reg * np.eye(num_coils)
    L = np.linalg.cholesky(cov)
    W = np.linalg.inv(L).conj().T  # whitening matrix
    ksp_flat = ksp.reshape(num_coils, -1)
    ksp_white = (W @ ksp_flat).reshape(ksp.shape)
    return ksp_white, W

def main():
    parser = argparse.ArgumentParser(description="Apply noise whitening to k-space data.")
    parser.add_argument('--ksp', type=str, required=True, help='Path to input k-space .npy')
    parser.add_argument('--noise', type=str, required=True, help='Path to noise .npy file')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory to store whitened k-space')
    parser.add_argument('--reg', type=float, default=1e-6, help='Regularization for noise covariance')

    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    ksp = np.load(args.ksp)
    noise = np.load(args.noise)

    # Whiten
    ksp_white, W = noise_whiten(ksp, noise, reg=args.reg)

    # Save results
    np.save(outdir / 'w_ksp.npy', ksp_white)
    np.save(outdir / 'whitening_matrix.npy', W)
    print(f"Saved whitened k-space to {outdir/'w_ksp.npy'}")
    print(f"Saved whitening matrix to {outdir/'whitening_matrix.npy'}")

if __name__ == '__main__':
    main()
