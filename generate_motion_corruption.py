import numpy as np
import sigpy as sp
import sigpy.plot as pl
from transform import RigidTransform
from utils import compute_transform_grids, generate_shot_mask

#img = np.load('./test_sample/low_res_img.npy')
#mps = np.load('./test_sample/low_res_mps.npy')
img = np.load('./test_sample/sense_recon.npy')
mps = np.load('./test_sample/mps.npy')
img_shape = img.shape

num_shots = 2
bin_axis = 1
bin_size = img_shape[bin_axis] // num_shots
print(bin_size)

shot_mask = generate_shot_mask(bin_size, img_shape, bin_axis)

#transforms = generate_motion_parameters(num_shots)
#transforms = np.array([[-1, -2, -1, (-5*np.pi/180), (-4*np.pi/180), (-2*np.pi/180)], [1, 2, 1, (5*np.pi/180), (4*np.pi/180), (2*np.pi/180)]])
transforms = np.array([[0,0,-1,(2.5)*np.pi/180,0,0],[0,0,1,(-2.5)*np.pi/180,0,0]])
#transforms[:, 3:] *= np.pi/180
kgrid, rkgrid = compute_transform_grids(img_shape)

ksp = 0
S = sp.linop.Multiply(img_shape, mps)
F = sp.linop.FFT(S.oshape, axes=[-3,-2,-1])
for shot_idx in range(len(transforms)):
    A = sp.linop.Multiply(F.oshape, shot_mask[shot_idx])
    T = RigidTransform(img_shape, img_shape, transforms[shot_idx], kgrid, rkgrid)
    ksp += (A * F * S * T * img)
pl.ImagePlot(S.H * F.H * ksp)
np.save('./test_sample/corr_ksp', ksp)
