import argparse
import numpy as np
from pathlib import Path
import twixtools

def main():
    parser = argparse.ArgumentParser(description="Extract k-space, ref, and noise from Siemens .dat file.")
    parser.add_argument('--dat', type=str, required=True, help='Path to Siemens raw .dat file')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory to store ksp/ref/noise')

    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load and extract from .dat
    multi_twix = twixtools.read_twix(args.dat, parse_pmu=False)
    mapped = twixtools.map_twix(multi_twix)
    print(f"Loaded {args.dat}")

    # Extract primary k-space data
    try:
        data = mapped[-1]['image']  # shape: (kx, coils, ky, kz, ...)
        data.flags['remove_os'] = True
        data.flags['zf_missing_lines'] = True
        ksp = data[:].squeeze().astype(np.complex64)
        ksp = np.transpose(ksp, (2, 0, 1, 3))
        np.save(outdir / 'ksp.npy', ksp)
        del ksp
        print(f"Saved ksp to {outdir/'ksp.npy'}")
    except Exception as e:
        print(f"Failed to extract k-space: {e}")

    # Extract primary reference data
    try:
        data = mapped[-1]['refscan'] # shape: (kx, coils, ky, kz, ...)
        data.flags['remove_os'] = True
        data.flags['zf_missing_lines'] = True
        ref = data[:].squeeze().astype(np.complex64)
        ref = np.transpose(ref, (2, 0, 1, 3))
        np.save(outdir / 'ref.npy', ref)
        del ref
        print(f"Saved reference scan to {outdir/'ref.npy'}")
    except Exception as e:
        print(f"Failed to extract reference scan: {e}")

    # Extract noise data
    try:
        data = mapped[-1]['noise']  # shape: (kx, coils, ky, kz, ...)
        data.flags['remove_os'] = True
        data.flags['zf_missing_lines'] = True
        noise = data[:].squeeze().astype(np.complex64)
        noise = np.transpose(noise, (2, 0, 1, 3))
        np.save(outdir / 'noise.npy', noise)
        del noise
        print(f"Saved noise to {outdir/'noise.npy'}")
    except Exception as e:
        print(f"Failed to extract noise data: {e}")

if __name__ == '__main__':
    main()
