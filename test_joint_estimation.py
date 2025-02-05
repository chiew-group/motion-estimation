import unittest

import numpy as np
import numpy.testing as npt

import sigpy as sp
import sigpy.mri.sim as sim
from scipy.ndimage import gaussian_filter
from transform import RigidTransform
from joint_estimation import JointEstimation
from image_estimation import ImageEstimation
from utils import *

if __name__ == "__main__":
    unittest.main()

class TestJointEstimation(unittest.TestCase):
    def shepp_logan_setup(self):
        size = 32
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
    
    def test_joint_estimation(self):
        ksp, mps, shot_mask, transforms, img = self.shepp_logan_setup()
        M = (sp.rss(img) > 1e-3)
        #P = sp.linop.Multiply(ksp.shape[1:], 1 / (np.sum(np.abs(mps) ** 2, axis=0) + 1e-3))                        
        gpu_device = sp.Device(0)
        kgrid, rkgrid = compute_transform_grids(img.shape, device=gpu_device)
        est_img, est_transforms = JointEstimation(ksp, mps, shot_mask, kgrid, rkgrid,
                                                  P=None, constraint=M, 
                                                  device=gpu_device, max_joint_iter=5000).run()
        import sigpy.plot as pl
        #pl.ImagePlot(np.concatenate([est_img.get(), img], axis=-1))
        pl.ImagePlot(est_img.get())
        #npt.assert_allclose(est_img.get(), img, atol=1e-3, rtol=1e-3)
        npt.assert_allclose(est_transforms.get(), transforms, atol=1e-3, rtol=1e-3)

    def test_joint_estimation_image_recon(self):
        ksp, mps, shot_mask, transforms, img = self.shepp_logan_setup()
        img_shape = [64, 64, 64]
        mps_shape = [10, 64, 64, 64]
        img = sp.shepp_logan(img_shape)
        img = gaussian_filter(img, 1)
        mps = sim.birdcage_maps(mps_shape)

        ksp = sp.fft(mps * img, axes=[-3,-2,-1])
        kgrid, rkgrid = compute_transform_grids(img_shape, device=sp.Device(0))
        transforms = np.zeros((1,6))
        shot_mask = np.ones((1, *img_shape), dtype=bool)

        recon, estimates = JointEstimation(ksp, mps, shot_mask, kgrid, rkgrid, 
                                           device=sp.Device(0), 
                                           max_joint_iter=1, max_nm_iter=0, max_cg_iter=100).run()
        #recon = ImageEstimation(ksp, mps, shot_mask, transforms, kgrid, rkgrid, 
        #                        max_iter=100, tol=0, device=sp.Device(0)).run()
        sense = sp.mri.app.SenseRecon(ksp, mps, tol=0, device=sp.Device(0)).run()

        import sigpy.plot as pl
        pl.ImagePlot(np.concatenate([recon.get(), sense.get()], axis=-1))
        npt.assert_allclose(recon.get(), sense.get(), atol=1e-3, rtol=1e-3)

    def test_joint_estimation_transform_estimation(self):
        ksp, mps, shot_mask, transforms, img = self.shepp_logan_setup()
        #M = (sp.rss(img) > 1e-3)
        #P = sp.linop.Multiply(ksp.shape[1:], 1 / (np.sum(np.abs(mps) ** 2, axis=0) + 1e-3))                        
        gpu_device = sp.Device(0)
        kgrid, rkgrid = compute_transform_grids(img.shape, device=gpu_device)
        est_img, est_transforms = JointEstimation(ksp, mps, shot_mask, kgrid, rkgrid, img=img,
                                                  device=gpu_device, max_joint_iter=1000,
                                                  max_cg_iter=3, max_nm_iter=1).run()
        import sigpy.plot as pl
        pl.ImagePlot(np.concatenate([est_img.get(), img], axis=-1))
        #npt.assert_allclose(est_img.get(), img, atol=1e-3, rtol=1e-3)
        npt.assert_allclose(est_transforms.get(), transforms, atol=1e-3, rtol=1e-3)

    def test_joint_estimation_image_est_with_correct_transforms(self):
        ksp, mps, shot_mask, transforms, img = self.shepp_logan_setup()
        #norm = np.max(np.abs(sp.ifft(ksp, axes=[-3,-2,-1])))
        M = (sp.rss(img) > 1e-3)
        #P = sp.linop.Multiply(ksp.shape[1:], 1 / (np.sum(np.abs(mps) ** 2, axis=0) + 1e-3))                        
        gpu_device = sp.Device(0)
        kgrid, rkgrid = compute_transform_grids(img.shape, device=gpu_device)
        est_img, est_transforms = JointEstimation(ksp, mps, shot_mask, kgrid, rkgrid, transforms=transforms,
                                                  device=gpu_device, max_joint_iter=100, constraint=M).run()
        est_img = est_img.get()
        import sigpy.plot as pl
        #pl.ImagePlot(np.concatenate([est_img, img], axis=-1))
        pl.ImagePlot(est_img)
        npt.assert_allclose(np.abs(est_img), np.abs(img), atol=1e-3, rtol=1e-3)
        #npt.assert_allclose(est_img, img, atol=1e-3, rtol=1e-3)
        #npt.assert_allclose(est_transforms.get(), transforms, atol=1e-3, rtol=1e-3)     