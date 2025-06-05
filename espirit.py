import argparse
import numpy as np
import cupy as cp
import sigpy as sp
import cupyx.scipy.ndimage as ndi
from tqdm import tqdm
from math import prod #Cupy prod doesn't work on tuples like numpy
from pathlib import Path


def hankel_fwd(x, kernel_size, dims_in):
    xp = cp.get_array_module(x)
    dims = list(dims_in) # Make a mutable copy

    if x.ndim == 3: # Or len(dims) == 3
        # Add a 4th dimension for coils if not present (single coil case)
        x = x[..., xp.newaxis]
        dims.append(1)
    
    Nx, Ny, Nz, N1 = dims[0], dims[1], dims[2], dims[3]
    kx, ky, kz = kernel_size[0], kernel_size[1], kernel_size[2]

    num_blocks_x = Nx - kx + 1
    num_blocks_y = Ny - ky + 1
    num_blocks_z = Nz - kz + 1
    total_blocks = num_blocks_x * num_blocks_y * num_blocks_z
    h_matrix = xp.zeros((kx * ky * kz, total_blocks, N1), dtype=x.dtype)
    idx = 0
    # Order of loops matches MATLAB: kx varies fastest, then ky, then kz
    for kz_start in range(num_blocks_z): # 0 to Nz - kz
        for ky_start in range(num_blocks_y): # 0 to Ny - ky
            for kx_start in range(num_blocks_x): # 0 to Nx - kx
                block = x[kx_start : kx_start + kx,
                          ky_start : ky_start + ky,
                          kz_start : kz_start + kz,
                          :] # Shape: (kx, ky, kz, N1)

                # Reshape block column-wise (Fortran order) to match MATLAB
                h_matrix[:, idx, :] = block.reshape((kx * ky * kz, N1), order='F')
                idx += 1
    return h_matrix

def fold_rx(x_in):
    xp = cp.get_array_module(x_in)
    x_permuted = xp.transpose(x_in, (0, 2, 1))
    return x_permuted.reshape((-1, x_permuted.shape[2]), order='F')

def rx_espirit_3d(calib, imsize, kernel_size=None, eig_thresh=0.02, mask_thresh=0.99):

    """

    Python implementation of 3D ESPIRiT receive sensitivity mapping.

    Args:

        calib (xp.ndarray): Complex Rx k-space [NKx, NKy, NKz, NRx].

        imsize (tuple): Image dimensions for output [Nx, Ny, Nz].

        kernel_size (tuple, optional): Kernel size [kx, ky, kz]. Defaults to (5,5,5).

        eig_thresh (float, optional): Scalar threshold for eigenvalues.

                                      If < 1, interpreted as s <= s(1)*eig_thresh.

                                      If >= 1, interpreted as number of eigenvalues r <= eig_thresh.

                                      Defaults to 0.02.

        mask_thresh (float, optional): Scalar threshold for sensitivity masking.

                                       Defaults to 0.99.

    Returns:

        xp.ndarray: Rx sensitivity maps [Nx, Ny, Nz, NRx].

    """
    xp = cp.get_array_module(calib)
    # Default params

    if kernel_size is None:

        kernel_size = (5, 5, 5)

    # Get dimensions

    calib_dims = calib.shape

    NRx = calib_dims[3]

    # Initialise outputs

    # Ensure imsize is a list or tuple that can be concatenated

    sens = xp.zeros(list(imsize) + [NRx], dtype=calib.dtype)

    # Generate calibration matrix

    # Hankel_fwd expects kernel_size as a list/tuple, and calib_dims

    H_hankel = hankel_fwd(calib, kernel_size, calib_dims)

    H = fold_rx(H_hankel) # H shape: (kx*ky*kz*NRx, num_blocks)

    # Get left singular (kernel x coil) vectors, above singular value threshold

    # 'econ' equivalent is full_matrices=False

    U_svd, S_diag, _ = xp.linalg.svd(H, full_matrices=False)

    # S_diag are singular values, U_svd columns are left singular vectors

    if eig_thresh < 1:

        # Threshold based on ratio to the largest singular value

        if S_diag.size > 0 and S_diag[0] > 0: # Avoid division by zero or issues with empty S_diag

            U_svd = U_svd[:, S_diag > S_diag[0] * eig_thresh]

        else: # If no significant singular values, U becomes empty (or handle as error)

            U_svd = xp.zeros((U_svd.shape[0], 0), dtype=U_svd.dtype)

    else:

        # Threshold based on number of singular values

        num_svals_to_keep = int(round(eig_thresh))

        U_svd = U_svd[:, :num_svals_to_keep]

    

    num_components = U_svd.shape[1]

    if num_components == 0:

        print("Warning: No significant singular components found. Returning zero sensitivity maps.")

        # Rotate phase (optional, as sens is zeros, but for completeness if it were non-zero)

        # sens = sens * xp.exp(-1j * xp.angle(sens[..., 0:1]))

        return sens

    # Reshape U_svd to (kx, ky, kz, NRx, num_components) using Fortran order

    # Each column of U_svd (length kx*ky*kz*NRx) is a component

    U_reshaped = U_svd.reshape((kernel_size[0], kernel_size[1], kernel_size[2], NRx, num_components), order='F')

    # Zero-pad U_reshaped in the spatial dimensions (first 3 dims of U_reshaped) to match imsize

    # Calculate padding for z-dimension (axis 2 of U_reshaped)

    pad_z_pre = int(xp.ceil((imsize[2] - kernel_size[2]) / 2))

    pad_z_post = int(xp.floor((imsize[2] - kernel_size[2]) / 2))

    # Padding specs for xp.pad: ((before_0, after_0), (before_1, after_1), ...)

    padding_z_specs = [(0,0), (0,0), (pad_z_pre, pad_z_post), (0,0), (0,0)]

    U_padded_z = xp.pad(U_reshaped, padding_z_specs, mode='constant', constant_values=0)

    # FFT along the z-dimension (axis 2)

    # fftshift before, ifft, ifftshift after

    U_fft_z = xp.fft.fftshift(xp.fft.ifft(xp.fft.ifftshift(U_padded_z, axes=2), axis=2), axes=2)

    # Loop over z-locations (for memory, as in MATLAB script)

    for z_idx in tqdm(range(imsize[2])):

        #print(f"Processing z-slice: {z_idx + 1}/{imsize[2]}") # Match 1-based display

        # Extract the current z-slice from U_fft_z

        # U_fft_z shape: (kx, ky, imsize_z, NRx, num_components)

        # slice_U_z shape: (kx, ky, NRx, num_components)

        current_z_slice_kernels = U_fft_z[:, :, z_idx, :, :]

        # Pad in x and y dimensions for this z-slice

        pad_x_pre = int(xp.ceil((imsize[0] - kernel_size[0]) / 2))

        pad_x_post = int(xp.floor((imsize[0] - kernel_size[0]) / 2))

        pad_y_pre = int(xp.ceil((imsize[1] - kernel_size[1]) / 2))

        pad_y_post = int(xp.floor((imsize[1] - kernel_size[1]) / 2))

        padding_xy_specs = [

            (pad_x_pre, pad_x_post), (pad_y_pre, pad_y_post), (0,0), (0,0)

        ]

        tmp_padded_xy = xp.pad(current_z_slice_kernels, padding_xy_specs, mode='constant', constant_values=0)

        

        # FFT along x (axis 0) and y (axis 1)

        tmp_fft_xy = xp.fft.ifftn(xp.fft.ifftshift(tmp_padded_xy, axes=(0, 1)), axes=(0, 1))
        tmp_fft_xy = xp.fft.fftshift(tmp_fft_xy, axes=(0, 1))


        # tmp_fft_xy shape: (imsize_x, imsize_y, NRx, num_components)

        # Perform Eigendecomposition on coil x component matrices, voxelwise

        # Keep the first component (eigenvector corresponding to largest eigenvalue)

        M = tmp_fft_xy.reshape(-1, NRx, num_components)  # (Nx*Ny, NRx, num_comp)
        C = M @ M.transpose(0, 2, 1).conj()              # (Nx*Ny, NRx, NRx)
        eigvals, eigvecs = xp.linalg.eigh(C)            # batched!

        idx_max = xp.argmax(eigvals.real, axis=1)                        # (Nx*Ny,)
        principal_vecs = xp.take_along_axis(eigvecs, idx_max[:, None, None], axis=2)[:, :, 0]  # (Nx*Ny, NRx)

        norm_factor = prod(imsize) / xp.sqrt(prod(kernel_size))
        mask = (xp.sqrt(eigvals[:, -1]) * norm_factor > mask_thresh)     # (Nx*Ny,)
        sens_slice = xp.zeros((imsize[0]*imsize[1], NRx), dtype=calib.dtype)
        sens_slice[mask] = principal_vecs[mask]
        sens[:, :, z_idx] = sens_slice.reshape(imsize[0], imsize[1], NRx)

    # Rotate phase relative to first coil

    # sens[..., 0] is the first coil. sens[..., 0:1] keeps the dimension for broadcasting.

    phase_ref = xp.angle(sens[..., 0:1])

    sens = sens * xp.exp(-1j * phase_ref)

    return sens

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Estimate ESPIRiT maps from k-space data.")
    parser.add_argument('--ksp', type=str, required=True, help='Path to input k-space reference scan .npy file')
    parser.add_argument('--out', type=str, required=True, help='Path to output maps .npy file')
    parser.add_argument('--kernel', type=int, nargs=3, default=[5, 5, 5], help='ESPIRiT kernel size')
    parser.add_argument('--eig_thresh', type=float, default=0.02, help='Eigenvalue threshold')
    parser.add_argument('--mask_thresh', type=float, default=0.99, help='Sensitivity masking threshold')
    parser.add_argument('--use-cpu', action='store_true', help='Force use of CPU (numpy) instead of GPU (cupy)')

    args = parser.parse_args()

    # Load k-space
    ksp = np.load(args.ksp)
    nc = ksp.shape[0]
    full_imsize = ksp.shape[1:]

    xp = np if args.use_cpu else cp
    ksp = sp.resize(ksp, [nc, 24, 24, 24])
    ksp = xp.asarray(ksp)
    # Run ESPIRiT
    #imsize = [128,128,128]
    imsize = full_imsize # assume shape is (nCoils, Nx, Ny, Nz)
    mps = rx_espirit_3d(ksp.transpose(1, 2, 3, 0), imsize, 
                         kernel_size=tuple(args.kernel), 
                         eig_thresh=args.eig_thresh, 
                         mask_thresh=args.mask_thresh)
    mps = mps.transpose((3,0,1,2))
    #factor = [full / ds for full, ds in zip(full_imsize, imsize)]
    #factor = [1] + factor
    #mps = ndi.zoom(mps, factor)
    
    import sigpy.plot as pl
    pl.ImagePlot(mps)
    # Save maps
    #maps = cp.asnumpy(maps)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    xp.save(args.out, mps)
    print(f"Saved maps to {args.out}")