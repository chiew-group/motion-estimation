import sigpy as sp
import numpy as np
import matplotlib.pyplot as plt

def generate_motion_parameters(num_states, low_freq_var=0.1, high_freq_var=5.0, spike_prob=0.02, seed=123456789):
    """
    Generate rigid body head motion parameters with low-frequency noise and high-frequency spikes.
    Rotations are constrained to radian units within ±20 degrees.

    Parameters:
    - num_states (int): Number of motion states (time points).
    - low_freq_var (float): Variance of the low-frequency noise.
    - high_freq_var (float): Variance of the high-frequency spikes.
    - spike_prob (float): Probability of a spike occurring at each time point.

    Returns:
    - np.ndarray: An array of shape (num_states, 6) representing the motion parameters
                  (3 translations and 3 rotations in radians).
    """
    rng = np.random.default_rng(seed)

    # Generate low-frequency noise using a cumulative sum of Gaussian noise
    low_freq_noise = np.cumsum(rng.normal(0, np.sqrt(low_freq_var), size=(num_states, 6)), axis=0)
    
    # Generate sparse high-frequency spikes
    spikes = rng.choice([0, 1], size=(num_states, 6), p=[1 - spike_prob, spike_prob]) \
             * rng.normal(0, np.sqrt(high_freq_var), size=(num_states, 6))
    
    # Combine low-frequency noise and spikes
    motion_parameters = low_freq_noise + spikes

    # Constrain rotation parameters to radians within ±20 degrees (±0.35 radians)
    #motion_parameters[:, 3:] = np.clip(motion_parameters[:, 3:], -np.deg2rad(20), np.deg2rad(20))
    motion_parameters[:, 3:] *= np.pi  / 180 # Convert degrees to radians

    return motion_parameters

def compute_spatial_coord(resolution, subsample_resolution=None, device=sp.cpu_device):
    xp = device.xp
    if subsample_resolution is None:
        subsample_resolution = resolution
    r1 = xp.linspace(-(resolution[0]//2),  resolution[0]//2, subsample_resolution[0], endpoint=False).reshape((-1,1,1))
    r2 = xp.linspace(-(resolution[1]//2),  resolution[1]//2, subsample_resolution[1], endpoint=False).reshape((1,-1,1))
    r3 = xp.linspace(-(resolution[2]//2),  resolution[2]//2, subsample_resolution[2], endpoint=False).reshape((1,1,-1))
    return r1, r2, r3

def compute_spectral_coord(resolution, device=sp.cpu_device):
    xp = device.xp
    k1 = xp.linspace(-xp.pi,  xp.pi, resolution[0], endpoint=False).reshape((-1,1,1)) #* (1/subsample_resolution[0])
    k2 = xp.linspace(-xp.pi,  xp.pi, resolution[1], endpoint=False).reshape((1,-1,1)) #* (1/subsample_resolution[1])
    k3 = xp.linspace(-xp.pi,  xp.pi, resolution[2], endpoint=False).reshape((1,1,-1)) #* (1/subsample_resolution[2])
    return k1, k2, k3

def compute_transform_grids(resolution, downsample_resolution, voxel_size, device=sp.cpu_device):
    #if subsample_resolution is None:
    #    subsample_resolution = resolution
    #r1, r2, r3 = compute_spatial_coord(resolution, subsample_resolution, device=device)
    #k1, k2, k3 = compute_spectral_coord(subsample_resolution, device=device)

    nx, ny, nz = resolution
    vx, vy, vz = voxel_size
    ds_x, ds_y, ds_z = downsample_resolution
    xp = device.xp
    factor = [resolution[i]//downsample_resolution[i] for i in range(3)]

    r1 = (xp.linspace(-(nx//2), nx//2, ds_x, endpoint=False) * vx).reshape((-1,1,1))
    r2 = (xp.linspace(-(ny//2), ny//2, ds_y, endpoint=False) * vy).reshape((1,-1,1))
    r3 = (xp.linspace(-(nz//2), nz//2, ds_z, endpoint=False) * vz).reshape((1,1,-1))
    
    k1 = (xp.linspace(-xp.pi,  xp.pi, ds_x, endpoint=False).reshape((-1,1,1)))
    k2 = (xp.linspace(-xp.pi,  xp.pi, ds_y, endpoint=False).reshape((1,-1,1)))
    k3 = (xp.linspace(-xp.pi,  xp.pi, ds_z, endpoint=False).reshape((1,1,-1)))

    rkgrid = [[k2 * r3, k3 * r1, k1 * r2], [k3 * r2, k1 * r3, k2 * r1]]
    return [k1, k2, k3], rkgrid

def generate_shot_mask(bins, img_shape, img_axis=1):
    num_shots = img_shape[img_axis] // bins
    sampling_mask = np.zeros((num_shots, *img_shape), dtype=bool)
    
    for shot in range(num_shots):
        idx = [shot] + [slice(None)] * len(img_shape)
        idx[1+img_axis] = slice(shot * bins, (shot+1) * bins)
        sampling_mask[tuple(idx)] = 1
    return sampling_mask

def generate_motion_corruption(img, mps, bins, bin_axis=1, transforms=None, device=sp.cpu_device):
    from transform import RigidTransform
    bin_size = img.shape[bin_axis] // bins
    shot_mask = generate_shot_mask(bin_size, img.shape, img_axis=bin_axis)
    kgrid, rkgrid = compute_transform_grids(img.shape, device=device)
    
    if transforms is None:
        transforms = generate_motion_parameters(len(shot_mask))
    else:
        transforms = sp.to_device(transforms, device)
    
    S = sp.linop.Multiply(img.shape, mps)
    F = sp.linop.FFT(S.oshape, axes=[-3,-2,-1])
    ksp = 0
    for shot_idx in range(len(transforms)):
        A = sp.linop.Multiply(F.oshape, shot_mask[shot_idx])
        T = RigidTransform(img.shape, img.shape, transforms[shot_idx], kgrid, rkgrid)
        ksp += (A * F * S * T * img)
    return ksp, transforms, shot_mask

def compute_transform_grids_voxel(shape, voxel_size, downsample_res=None, xp=np):
    nx, ny, nz = shape
    vx, vy, vz = voxel_size
    if downsample_res is None:
        downsample_res = shape
    ds_x, ds_y, ds_z = downsample_res

    rx = xp.fft.fftshift((xp.linspace(-nx//2, nx//2, ds_x, endpoint=False) * vx).reshape(-1,1,1))
    ry = xp.fft.fftshift((xp.linspace(-ny//2, ny//2, ds_y, endpoint=False) * vy).reshape(1,-1,1))
    rz = xp.fft.fftshift((xp.linspace(-nz//2, nz//2, ds_z, endpoint=False) * vz).reshape(1,1,-1))

    kx =  xp.fft.fftshift(xp.linspace(-xp.pi,  xp.pi, ds_x, endpoint=False).reshape((-1,1,1)))
    ky =  xp.fft.fftshift(xp.linspace(-xp.pi,  xp.pi, ds_y, endpoint=False).reshape((1,-1,1)))
    kz =  xp.fft.fftshift(xp.linspace(-xp.pi,  xp.pi, ds_z, endpoint=False).reshape((1,1,-1)))

    kgrid = {'x': kx, 'y': ky, 'z': kz }
                     
    rkgrid = {
        'tan': {
            'x': rz * ky, # rotation around x-axis (tan-shears: y↔z typically)
            'y': rx * kz, # rotation around y-axis (tan-shears: x↔z)
            'z': ry * kx  # rotation around z-axis (tan-shears: x↔y)
        },
        'sin': {
            'x': ry * kz, # rotation around x-axis (sin-shears: y↔z typically)
            'y': rz * kx, # rotation around y-axis (sin-shears: x↔z)
            'z': rx * ky  # rotation around z-axis (sin-shears: x↔y)
        }
    }

    return kgrid, rkgrid

def _pad_to_square(img3d):
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
    img3d = _pad_to_square(img3d)

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
        print(f"Saved image montage to {save_path}")
    else:
        plt.show()