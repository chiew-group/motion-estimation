import numpy as np
import sigpy as sp
import sigpy.mri
from tqdm import tqdm
import sigpy.plot as pl
from transform import RigidTransform
from utils import compute_transform_grids, generate_shot_mask

img = np.load('./sample/gen/ground_truth.npy')
mps = np.load('./sample/gen/mps.npy')
img_shape = img.shape

num_bins = 6 #300 slices and we want to bin 50 motion states
bin_axis = 1
bin_size = img_shape[bin_axis] // num_bins 
print(f"Num of bins are {num_bins} each with size {bin_size}")

shot_mask = np.zeros((num_bins, *img_shape), dtype=bool)
for shot in range(num_bins):
    shot_mask[shot, :, shot*bin_size:(shot+1)*bin_size, :] = True

theta = 2
transforms = np.zeros((num_bins, 6), dtype=float)
#transforms[:, 3:] = np.pi * 2 * (np.random.rand(num_shots, 3)-0.5) / 180
transforms[:, 3] = np.pi * (np.linspace(-2, 2, num_bins, endpoint=True, dtype=float)) / 180
transforms[:, 4] = np.pi * (np.linspace(-3, 2, num_bins, endpoint=True, dtype=float)) / 180
transforms[:, 5] = np.pi * (np.linspace(-4, 2, num_bins, endpoint=True, dtype=float)) / 180
#transforms[:, 3:] = np.pi * theta * (np.random.rand(num_bins, 3) - 0.5) / 180
#print(transforms)
#pl.ImagePlot(shot_mask)

device = sp.Device(0)
kgrid, rkgrid = compute_transform_grids(img_shape, device=device)

ksp = 0
transforms = sp.to_device(transforms, device=device)
img = sp.to_device(img, device=device)
for shot in tqdm(range(num_bins)):
    S = sp.mri.linop.Sense(mps, weights=shot_mask[shot], coil_batch_size=None)
    T = RigidTransform(img_shape, img_shape, transforms[shot], kgrid, rkgrid)
    ksp += (S * T * img)
np.save('./sample/other/lin_corr_ksp_6bins', ksp)
pl.ImagePlot((mps.conj() * sp.ifft(ksp.get(), axes=[-3, -2, -1])).sum(axis=0))
