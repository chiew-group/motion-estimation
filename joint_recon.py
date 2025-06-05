import numpy as np
import cupy as cp
import sigpy as sp
from image_estimation import estimate_image_cg
from transform_estimation import estimate_transform
from transform import RigidTransformCudaOptimzied
from tqdm import tqdm


class JointRecon:
    def __init__(
        self, 
        ksp, 
        mps, 
        shot_mask,
        kgrid,
        rkgrid,
        t0=None,
        max_cg_iter=3,
        max_nm_iter=1,
        max_joint_iter=100,
        xp=np
    ):
        
        #Fixed parameters
        self.kgrid = kgrid
        self.rkgrid = rkgrid
        self.max_cg_iter = max_cg_iter
        self.max_nm_iter = max_nm_iter
        self.max_joint_iter = max_joint_iter

        self.num_shots = len(shot_mask)
        self.img_shape = ksp.shape[1:]

        #Data needs to be uncentered and then moved onto the device
        self.ksp = cp.fft.fftshift(cp.array(ksp))
        self.mps = cp.fft.fftshift(cp.array(mps))
        self.shot_mask = cp.fft.fftshift(cp.array(shot_mask))
    
        self.x = cp.zeros(self.img_shape, dtype=ksp.dtype)
        if t0 is None:
            self.t = cp.zeros((self.num_shots,6), dtype=float)
        else:
            self.t = cp.asarray(t0)

        #Preallocate buffers for the transform estimation calculations
        self.residual_buf = xp.zeros_like(self.ksp)
        self.partial_buf = xp.zeros((6,) + self.ksp.shape, dtype=self.ksp.dtype)
        self.gradient_buf = xp.zeros(6, dtype=float)
        self.hessian_buf = xp.zeros((6,6), dtype=float)

        self.damp = xp.ones((self.num_shots))
        self.convergence_flags = xp.zeros((self.num_shots), dtype=bool)


    def run(self):
        self.transform_history = []
        self.objective_history = []

        for iter in tqdm(range(self.max_joint_iter)):
            self.x = estimate_image_cg(self.ksp, self.mps, self.shot_mask, self.t, self.kgrid, self.rkgrid, 
                                    x0=self.x, max_iter=self.max_cg_iter)
            estimate_transform(self.ksp, self.mps, self.shot_mask, self.x, self.kgrid, self.rkgrid,
                                        self.damp, self.convergence_flags,
                                        self.residual_buf, self.partial_buf, self.gradient_buf, self.hessian_buf,
                                        transforms=self.t, max_iter=self.max_nm_iter)
            self.transform_history.append(cp.asnumpy(self.t.copy()))
            self.objective_history.append(self._compute_objective())

        self.transform_history = np.stack(self.transform_history)
        self.objective_history = np.array(self.objective_history)
        return cp.fft.fftshift(self.x), self.t
            
    def _compute_objective(self):
        total_loss = 0.0
        for s in range(self.num_shots):
            T = RigidTransformCudaOptimzied(self.t[s], self.kgrid, self.rkgrid)
            Ex = cp.fft.fftn(self.mps * T.apply(self.x), axes=(-3,-2,-1))
            resid = self.shot_mask[s] * (Ex - self.ksp)
            total_loss += (cp.vdot(resid, resid).real / resid.size)
        return total_loss.item()
