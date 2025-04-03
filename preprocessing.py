import numpy as np
import sigpy as sp
import sigpy.mri
import argparse
import twixtools
import os

def coil_compression(mps, ksp=None, perc=0.95, reg=1e-6):
    
    #Compute P value
    #mps are [nc, kx, ky, kz]
    num_coils = mps.shape[0]
    img_shape = mps.shape[1:]

    mps_conj = mps.conj()
    norm = 1 / (np.sum(mps * mps.conj(), axis=0) + reg)

    mps = mps.reshape((num_coils, -1))
    mps_conj = mps.conj().reshape((num_coils, -1))
    P = (mps * norm.flatten()) @ mps_conj.T
    
    #Threshold the singular values and vectors
    U, S, _ = np.linalg.svd(P)
    print(S)
    singular_total = np.sum(S)
    num_reduced_coils = num_coils - ((np.cumsum(S) / singular_total) >= perc).sum()
    print(f'Reducing from {num_coils} -> {num_reduced_coils} coils')

    #Make the compression matrix
    A = U.conj().T[:num_reduced_coils]
    compressed_mps = (A @ mps).reshape(num_reduced_coils, *img_shape)

    if ksp is not None:
        compressed_ksp = (A @ ksp.reshape(num_coils, -1)).reshape(num_reduced_coils, *img_shape) 

    return compressed_mps, compressed_ksp

def get_kspaces(filename):
    multi_twix = twixtools.read_twix(filename, parse_pmu=False)
    
    # map the twix data to twix_array objects
    mapped = twixtools.map_twix(multi_twix)
    mapped_img_data = mapped[-1]['image']
    mapped_refscan_data = mapped[-1]['refscan']
    
    # make sure that we later squeeze the right dimensions:
    print(f'Img Data non singleton dims : {mapped_img_data.non_singleton_dims}')
    print(f'RefScan Data non singleton dims : {mapped_refscan_data.non_singleton_dims}')
    
    # remove 2x oversampling and zero pad to ensure same shape
    mapped_img_data.flags['remove_os'] = True
    mapped_img_data.flags['zf_missing_lines'] = True
    mapped_refscan_data.flags['remove_os'] = True
    mapped_refscan_data.flags['zf_missing_lines'] = True
    
    
    image_ksp = mapped_img_data[:].squeeze()
    refscan_ksp = mapped_refscan_data[:].squeeze()
    print(f'Dimensions of image k-space is {image_ksp.shape}')
    print(f'Dimensions of refscan k-space is {refscan_ksp.shape}')
    
    #Rearrange so that shape follows format of [nc, par, line, col]
    image_ksp = np.transpose(image_ksp, (2, 0, 1 ,3))
    refscan_ksp = np.transpose(refscan_ksp, (2, 0, 1, 3))
    print(f'New dimenions are [nc, par, line, col] -> {refscan_ksp.shape}')

    return image_ksp, refscan_ksp

def sampling_mask(bins, img_shape, img_axis=1):
    num_shots = img_shape[img_axis] // bins
    sampling_mask = np.zeros((num_shots, *img_shape), dtype=bool)
    
    for shot in range(num_shots):
        idx = [shot] + [slice(None)] * len(img_shape)
        idx[1+img_axis] = slice(shot * bins, (shot+1) * bins)
        sampling_mask[tuple(idx)] = 1
    return sampling_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Estimate maps and compress both maps and kspace from raw dat file'
    )
    parser.add_argument("dat_file", help="Dat file of RAW data")
    parser.add_argument("out_dir", help="Save all outputs to a file, creates it if not made already")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base_name = os.path.basename(args.dat_file).replace('.dat', '')

    ksp, ref_ksp = get_kspaces(args.dat_file)

    print(ksp.dtype, ref_ksp.dtype)
    ref_img = sp.ifft(ref_ksp, axes=(1,2,3))
    magnitude = np.sum(np.abs(ref_img)**2, axis=(0,1,2))
    threshold = np.percentile(magnitude, 5)
    desired_indicies = np.argwhere(magnitude>threshold).flatten()

    refscan_partial_space = sp.fft(ref_img, axes=[1,2])
    device = sp.Device(0)
    print('estimating mps...')
    from tqdm import tqdm
    pbar = tqdm(total=len(desired_indicies), desc='Estimating mps', leave=True)
    with device:
        xp = device.xp 
        mps = xp.zeros(refscan_partial_space.shape, dtype=np.complex64)
        for e, i in enumerate(desired_indicies):
            pbar.update()
            mps[:,:,:,i] =  sigpy.mri.app.EspiritCalib(refscan_partial_space[:,:,:,i], device=device, show_pbar=False).run()
    pbar.close()
    mps = sp.to_device(mps)

    mps, ksp = coil_compression(mps, ksp)

    norm = np.linalg.norm(sp.ifft(ksp, axes=[-3,-2,-1]))
    recon = sigpy.mri.app.SenseRecon(ksp / norm, mps, device=device).run()
    recon *= norm
    
    ksp = sp.to_device(ksp)
    mps = sp.to_device(mps)
    recon = sp.to_device(recon)

    np.save(os.path.join(args.out_dir, f"{base_name}_ksp"), ksp)
    print(f"Kspace saved in directory {args.out_dir}!")
    np.save(os.path.join(args.out_dir, f"{base_name}_mps"), mps)
    print(f"Maps saved in directory {args.out_dir}!")
    np.save(os.path.join(args.out_dir, f"{base_name}_sense"), recon)
    print(f"Naive Sense recon saved in directory {args.out_dir}!")  