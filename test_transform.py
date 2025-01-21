import unittest

import numpy as np
import numpy.testing as npt

import sigpy as sp
import sigpy.mri.sim as sim
from scipy.ndimage import gaussian_filter
from transform import RigidTransform
from utils import *

if __name__ == "__main__":
    unittest.main()

class TestTransform(unittest.TestCase):
    def shepp_logan_setup(self):
        img_shape = [64, 64, 64]
        mps_shape = [10, 64, 64, 64]

        img = sp.shepp_logan(img_shape)
        img = gaussian_filter(img, 1)
        mps = sim.birdcage_maps(mps_shape)

        num_shots = 8
        bin_size = 8
        num_bins = 64 // bin_size
        shot_mask = generate_shot_mask(num_bins, img_shape)
        transforms = generate_motion_parameters(num_shots)

        kgrid, rkgrid = compute_transform_grids(img_shape)

        ksp = 0
        S = sp.linop.Multiply(img_shape, mps)
        F = sp.linop.FFT(S.oshape, axes=[-3,-2,-1])
        for shot_idx in range(len(transforms)):
            A = sp.linop.Multiply(F.oshape, shot_mask[shot_idx])
            T = RigidTransform(img_shape, img_shape, transforms[shot_idx], kgrid, rkgrid)
            ksp += (A * F * S * T * img)

        return ksp, mps, shot_mask, transforms, img