import unittest

import numpy as np
import numpy.testing as npt

import sigpy as sp
import sigpy.mri.sim as sim
from scipy.ndimage import gaussian_filter
from transform import RigidTransform
from image_estimation import ImageEstimation
from utils import *

if __name__ == "__main__":
    unittest.main()

class TestImageEstimation(unittest.TestCase):
    def shepp_logan_setup(self):
        img_shape = [32, 32, 32]
        mps_shape = [10, 32, 32, 32]

        img = sp.shepp_logan(img_shape)
        img = gaussian_filter(img, 1)
        mps = sim.birdcage_maps(mps_shape)

        num_shots = 2
        bin_size = 16
        num_bins = 64 // bin_size
        #shot_mask = generate_shot_mask(num_bins, img_shape)
        shot_mask = np.zeros((num_shots, *img_shape), dtype=bool)
        for s in range(num_shots):
            shot_mask[s, s*bin_size:(s+1)*bin_size] = True
        #transforms = generate_motion_parameters(num_shots)
        transforms = np.array([[-1, -2, -1, (-5*np.pi/180), (-4*np.pi/180), (-2*np.pi/180)], [1, 2, 1, (5*np.pi/180), (4*np.pi/180), (2*np.pi/180)]])
        #transforms[:, 3:] *= np.pi/180
        kgrid, rkgrid = compute_transform_grids(img_shape)

        ksp = 0
        S = sp.linop.Multiply(img_shape, mps)
        F = sp.linop.FFT(S.oshape, axes=[-3,-2,-1])
        for shot_idx in range(len(transforms)):
            A = sp.linop.Multiply(F.oshape, shot_mask[shot_idx])
            T = RigidTransform(img_shape, img_shape, transforms[shot_idx], kgrid, rkgrid)
            ksp += (A * F * S * T * img)

        return ksp, mps, shot_mask, transforms, img
    
    def test_image_estimation(self):
        ksp, mps, shot_mask, transforms, img = self.shepp_logan_setup()
        M = (sp.rss(img) > 1e-3)
        P = sp.linop.Multiply(ksp.shape[1:], 1 / (np.sum(np.abs(mps) ** 2, axis=0) + 1e-3))
        ksp = sp.to_device(ksp, sp.Device(0))
        mps = sp.to_device(mps, sp.Device(0))
        shot_mask = sp.to_device(shot_mask, sp.Device(0))
        transforms = sp.to_device(transforms, sp.Device(0))
        img = sp.to_device(img, sp.Device(0))
        M = sp.to_device(M, sp.Device(0))
                        
        gpu_device = sp.Device(0)
        xp = gpu_device.xp
        kgrid, rkgrid = compute_transform_grids(img.shape, device=gpu_device)

    
        recon = ImageEstimation(ksp, mps, shot_mask, transforms, kgrid, rkgrid, P=P, constraint=M, max_iter=1000, tol=1e-12, device=gpu_device).run()
        import sigpy.plot as pl
        pl.ImagePlot(recon)
        npt.assert_allclose(img.get(), recon.get(), atol=1e-3, rtol=1e-3)
    
    def test_image_estimation_fidelity(self):
        ksp, mps, shot_mask, transforms, img = self.shepp_logan_setup()
        img_shape = [64, 64, 64]
        mps_shape = [10, 64, 64, 64]
        img = sp.shepp_logan(img_shape)
        img = gaussian_filter(img, 1)
        mps = sim.birdcage_maps(mps_shape)

        ksp = sp.fft(mps * img, axes=[-3,-2,-1])
        kgrid, rkgrid = compute_transform_grids(img_shape, device=sp.Device(0))
        transforms = np.zeros((1,6))
        shot_mask = np.ones(img_shape, dtype=bool)

        recon = ImageEstimation(ksp, mps, shot_mask, transforms, kgrid, rkgrid, 
                                max_iter=100, tol=0, device=sp.Device(0)).run()
        sense = sp.mri.app.SenseRecon(ksp, mps, tol=0, device=sp.Device(0)).run()

        import sigpy.plot as pl
        pl.ImagePlot(np.concatenate([recon.get(), sense.get()], axis=-1))
        npt.assert_allclose(recon.get(), sense.get(), atol=1e-3, rtol=1e-3)
