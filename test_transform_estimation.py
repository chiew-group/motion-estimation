import unittest

import numpy as np
import numpy.testing as npt

import sigpy as sp
import sigpy.mri.sim as sim
from scipy.ndimage import gaussian_filter
from transform import RigidTransform
from transform_estimation import TransformEstimation
from utils import *

if __name__ == "__main__":
    unittest.main()

class TestTransformEstimation(unittest.TestCase):
    def shepp_logan_setup(self):
        size = 64
        nc = 10
        img_shape = [size] * 3
        mps_shape = [nc] + img_shape

        #img_shape = [32, 32, 32]
        #mps_shape = [10, 32, 32, 32]

        img = sp.shepp_logan(img_shape)
        img = gaussian_filter(img, 1)
        mps = sim.birdcage_maps(mps_shape)

        num_shots = 4
        bin_size = size // num_shots
        #num_bins = 64 // bin_size
        #shot_mask = generate_shot_mask(num_bins, img_shape)
        shot_mask = np.zeros((num_shots, *img_shape), dtype=bool)
        for s in range(num_shots):
            shot_mask[s, s*bin_size:(s+1)*bin_size] = True
        #transforms = generate_motion_parameters(num_shots)
        #transforms = np.array([[-1, -2, -1, (-5*np.pi/180), (-4*np.pi/180), (-2*np.pi/180)], [1, 2, 1, (5*np.pi/180), (4*np.pi/180), (2*np.pi/180)]])
        #transforms[:, 3:] *= np.pi/180
        transforms = np.zeros((num_shots, 6), dtype=float)
        #transforms[:, 3:] = np.pi * 2 * (np.random.rand(num_shots, 3)-0.5) / 180
        transforms[:, 3] = np.pi * (np.linspace(-2, 2, num_shots, endpoint=True, dtype=float)) / 180
        transforms[:, 4] = np.pi * (np.linspace(-3, 3, num_shots, endpoint=True, dtype=float)) / 180
        transforms[:, 5] = np.pi * (np.linspace(-4, 4, num_shots, endpoint=True, dtype=float)) / 180
        
        kgrid, rkgrid = compute_transform_grids(img_shape)

        ksp = 0
        S = sp.linop.Multiply(img_shape, mps)
        F = sp.linop.FFT(S.oshape, axes=[-3,-2,-1])
        for shot_idx in range(len(transforms)):
            A = sp.linop.Multiply(F.oshape, shot_mask[shot_idx])
            T = RigidTransform(img_shape, img_shape, transforms[shot_idx], kgrid, rkgrid)
            ksp += (A * F * S * T * img)

        return ksp, mps, shot_mask, transforms, img
    
    def test_transform_estimation(self):
        ksp, mps, shot_mask, transforms, img = self.shepp_logan_setup()
        #ksp = sp.to_device(ksp, sp.Device(0))
        #mps = sp.to_device(mps, sp.Device(0))
        #shot_mask = sp.to_device(shot_mask, sp.Device(0))
        #transforms = sp.to_device(transforms, sp.Device(0))
        #img = sp.to_device(img, sp.Device(0))
        M = sp.rss(img) > 1e-3                
        gpu_device = sp.Device(0)
        xp = gpu_device.xp

        init = xp.zeros_like(transforms)
        damp = xp.ones(len(transforms))
        convergence_flags = xp.zeros(len(transforms), dtype=bool)
        convergence_reset_count = xp.zeros(len(transforms), dtype=int)
        kgrid, rkgrid = compute_transform_grids(img.shape, device=gpu_device)
        TransformEstimation(ksp, mps, shot_mask, init, img, 
                            kgrid, rkgrid, damp, 
                            convergence_flags, convergence_reset_count, 
                            constraint=M, tol=1e-3, max_iter=10000, device=gpu_device).run()

        npt.assert_allclose(transforms, init.get(), atol=1e-3, rtol=1e-3)