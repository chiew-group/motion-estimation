import argparse
import twixtools
import nibabel as nib
from pathlib import Path
import numpy as np
import sigpy as sp
import sigpy.mri
import sigpy.plot as pl
from espirit import rx_espirit_3d
from coil_compression import coil_compression
from utils import compute_transform_grids_voxel, show_mid_slices
import cupy as cp
from joint_recon import JointRecon
from pyramid import pyramid_reconstruction

def noise_whiten(ksp, noise, reg=1e-6):
    num_coils = ksp.shape[0]
    #noise = noise.reshape(num_coils, -1)
    cov = (noise @ noise.conj().T) / noise.shape[1]
    cov += reg * np.eye(num_coils)
    L = np.linalg.cholesky(cov)
    W = np.linalg.inv(L).conj().T  # whitening matrix
    white_ksp = (W @ ksp.reshape(num_coils, -1)).reshape(ksp.shape)
    return white_ksp, W

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Joint reconstruction for motion correction")
    parser.add_argument("--dat", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--nshots", required=True)
    parser.add_argument("--iters", default=1000)
    args = parser.parse_args()

    preprocess_path = Path(args.out) / "pre"
    preprocess_path.mkdir(exist_ok=True)
    
    recon_path = Path(args.out) / "recon"
    recon_path.mkdir(exist_ok=True)

    # Load and extract from .dat
    mapped = twixtools.map_twix(twixtools.read_twix(args.dat))
    print(f"Loaded {args.dat}")


    # Extract primary k-space data
    #Data comes from the data file in the following shape
    #[par, lin, channel, col]
    try:
        data = mapped[-1]['image']  # shape: (kx, coils, ky, kz, ...)
        data.flags['remove_os'] = True
        lines = [mdb.cLin for mdb in data.mdb_list]
        partitions = [mdb.cPar for mdb in data.mdb_list]
        ksp = data[:].squeeze().astype(np.complex64)
        ksp = np.transpose(ksp, (2, 0, 1, 3))
        print(f"kspace shape - {ksp.shape}")
        np.save(preprocess_path / "ksp.npy", ksp)
        #del ksp
        print(f"Saved ksp to {preprocess_path / 'ksp.npy'}")
    except Exception as e:
        print(f"Failed to extract k-space: {e}")

    # Extract primary reference data
    try:
        data = mapped[-1]['refscan'] # shape: (kx, coils, ky, kz, ...)
        data.flags['remove_os'] = True
        data.flags['skip_empty_lead'] = True #Usually we get the first half just all black
        ref = data[:].squeeze().astype(np.complex64)
        ref = np.transpose(ref, (2, 0, 1, 3))
        print(f"reference shape - {ref.shape}")
        np.save(preprocess_path / "ref.npy", ref)
        print(f"Saved reference scan to {preprocess_path / 'ref.npy'}")
    except Exception as e:
        print(f"Failed to extract reference scan: {e}")

    # Extract noise data
    try:
        data = mapped[-1]['noise']  # shape: (coils, columns)
        data.flags['remove_os'] = True
        noise = data[:].squeeze().astype(np.complex64)
        #noise = np.transpose(noise, (2, 0, 1, 3))
        print(f"noise shape - {noise.shape}")
        np.save(preprocess_path / "noise.npy", noise)
        #del noise
        print(f"Saved noise to {preprocess_path / 'noise.npy'}")
    except Exception as e:
        print(f"Failed to extract noise data: {e}")


    ksp, white_matrix = noise_whiten(ksp, noise)
    np.save(preprocess_path / "white_ksp.npy", ksp)
    del noise

    n_coils = ksp.shape[0]
    ref = sp.resize(ref, [n_coils, 24,24,24])
    ref = sp.to_device(ref, sp.Device(0))
    imsize = ksp.shape[1:]
    #TODO right now the esprit takes kspace in a odd shape so tranposes are required
    mps = rx_espirit_3d(ref.transpose(1,2,3,0), imsize,
                        kernel_size=(6,6,6),
                        eig_thresh=0.02,
                        mask_thresh=0.95)
    mps = mps.transpose((3,0,1,2))
    mps = sp.to_device(mps)
    np.save(preprocess_path / "mps.npy", mps)
    del ref


    mps, ksp = coil_compression(mps, ksp=ksp, perc=0.95)
    np.save(preprocess_path / "cc_mps.npy", mps)
    np.save(preprocess_path / "cc_ksp.npy", ksp)

    sense = sp.mri.app.SenseRecon(ksp, mps, device=sp.Device(0)).run()

    np.save(recon_path / "sense.npy", sense)
    print(f"Save sense recon to {recon_path / 'sense.npy'}")
    
    n_shots = args.nshots
    motion_mask = np.zeros((n_shots, ksp.shape[1], ksp.shape[2]), dtype=bool)
    samples_per_shot = len(lines) // n_shots
    for s in range(n_shots):
        start = s * samples_per_shot
        end = (s+1) * samples_per_shot
        motion_mask[s, partitions[start:end], lines[start:end]] = True
    motion_mask = motion_mask[..., None]

    recon = pyramid_reconstruction(ksp, mps, motion_mask, n_shots, n_joint_iters=args.iters, save_path=recon_path)

    np.save(recon_path / "joint_recon.npy", recon)
    show_mid_slices(recon, save_path=recon_path / "image_slices.png")
    nib.save(nib.Nifti1Image(recon, np.eye(4)), recon_path / "joint_recon.nii.gz")

