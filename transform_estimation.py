import sigpy as sp
from transform import RigidTransform, RigidTransformDerivative, RigidTransformDerivativeCuda, RigidTransformCudaOptimzied
import cupy as cp
from cupyx.scipy.fft import fftn as gpu_fftn, ifftn as gpu_ifftn
import numpy as np 

class LMAlgorithm(sp.alg.Alg):
    def __init__(self, ksp, mps, shot_mask, 
                 transforms, img, kgrid, rkgrid, 
                 damp, convergence_flags, convergence_reset_count, max_iter=100, tol=1e-4):
        #Constant Algorithm Parameters
        self.ksp = ksp
        self.mps = mps
        self.shot_mask = shot_mask
        self.kgrid = kgrid
        self.rkgrid = rkgrid
        self.tol = tol

        #Mutable INPUT values that will be updated by the algorithm
        self.transforms = transforms
        self.img = img
        self.damp = damp
        self.convergence_flags = convergence_flags
        self.convergence_reset_count = convergence_reset_count

        #Algorithm State
        self.num_shots = len(transforms)
        self.img_shape = self.img.shape

        xp = sp.get_array_module(self.img)
        self.partial_buf = xp.zeros((6,) + self.ksp.shape, dtype=img.dtype)
        self.grad_buf = xp.zeros(6, dtype=float)
        self.hess_buf = xp.zeros((6,6), dtype=float)

        super().__init__(max_iter)

    def _update(self):
        xp = sp.get_array_module(self.img)
        S = sp.linop.Multiply(self.img_shape, self.mps)
        F = sp.linop.FFT(S.oshape, axes=[-3,-2,-1])
        
        for shot_idx in range(self.num_shots):
            if self.convergence_flags[shot_idx]:
                self.convergence_reset_count[shot_idx] += 1
                if self.convergence_reset_count[shot_idx] == 500:
                    self.convergence_reset_count[shot_idx] = 0
                    self.convergence_flags[shot_idx] = False
                continue 
            A = sp.linop.Multiply(F.oshape, self.shot_mask[shot_idx])
            T = RigidTransform(self.img_shape, self.img_shape, self.transforms[shot_idx], self.kgrid, self.rkgrid)

            #Calculate the resid error or energy
            resid = (A * F * S * T * self.img) - (A*self.ksp)
            current_error = xp.linalg.norm(resid)

            #Compute the partial derive kspaces
            #partials = []
            #gradient = [] 
            for p_idx in range(6):
                T = RigidTransformDerivative(self.img_shape, self.img_shape, p_idx, self.transforms[shot_idx], self.kgrid, self.rkgrid)
                #partials.append(A * F * S * T * self.img)
                self.partial_buf[p_idx] = A * F * S * T * self.img
                self.grad_buf[p_idx] = xp.vdot(resid, self.partial_buf[p_idx]).real
                #Compute the gradient of the motion state while we are at it
                #gradient.append(xp.sum(xp.real(resid.conj() * partials[-1])))
            #gradient = xp.array(gradient)

            #Approximate the Hessian matrix from partials
            """
            hessian = xp.zeros((6,6))
            for row in range(6):
                for col in range(row, 6):
                    hessian[row, col] = xp.sum(xp.real(partials[row].conj() * partials[col]))
            i_lower = xp.tril_indices(6, -1)
            hessian[i_lower] = hessian.T[i_lower]
            """
            for i in range(6):
                pi = self.partial_buf[i]
                for j in range(i, 6):
                    val = xp.vdot(pi, self.partial_buf[j]).real
                    if i == j:
                        self.hess_buf[i,i] = val + self.damp[shot_idx]
                    else:
                        self.hess_buf[i, j] = val
                        self.hess_buf[j, i] = val
            
            #hessian += self.damp[shot_idx] * xp.eye(6)
            #delta = xp.linalg.lstsq(hessian, gradient, rcond=None)[0]
            #self.hess_buf 
            delta = xp.linalg.solve(self.hess_buf, self.grad_buf)
            next_transform = self.transforms[shot_idx] - delta
            
            #Calculate the resid for the new transform
            T = RigidTransform(self.img_shape, self.img_shape, next_transform, self.kgrid, self.rkgrid)
            resid = (A * F * S * T * self.img) - (A*self.ksp)
            next_error = xp.linalg.norm(resid)

            #Check candidate for update and if so check partial convergence for this motion state
            if next_error < current_error:
                if (self.transforms[shot_idx] - next_transform < self.tol).all():
                    self.convergence_flags[shot_idx] = True
                self.transforms[shot_idx] = next_transform.copy()
                self.damp[shot_idx] = xp.maximum(self.damp[shot_idx] / 3, 1e-4)
            else:
                self.damp[shot_idx] = xp.minimum(self.damp[shot_idx] * 1.5, 1e16)

class TransformEstimation(sp.app.App):
    def __init__(self, ksp, mps, shot_mask, 
                 transforms, img, kgrid, rkgrid, 
                 damp, convergence_flags, convergence_reset_count, constraint=None, max_iter=100, tol=1e-6, device=sp.cpu_device, show_pbar=True, leave_pbar=True):
        self.transforms = sp.to_device(transforms, device)
        self.kgrid = kgrid
        self.rkgrid = rkgrid
        self.img = sp.to_device(img, device)
        self.img_shape = img.shape
        self.constraint = constraint
        self.device = device

        ksp_gpu = sp.to_device(ksp, device)
        alg = LMAlgorithm(ksp_gpu, mps, shot_mask, 
                          self.transforms, self.img, kgrid, rkgrid, 
                          damp, convergence_flags, convergence_reset_count, max_iter=max_iter, tol=tol)
        
        super().__init__(alg, show_pbar=show_pbar, leave_pbar=leave_pbar)
    
    def _post_update(self):
        mean_transform = self.transforms.mean(axis=0)
        T = RigidTransform(self.img_shape, self.img_shape, mean_transform, self.kgrid, self.rkgrid)
        next_img = (T * self.img)
        
        if self.constraint is not None:
            self.constraint = sp.to_device(self.constraint, self.device)
            next_img *= self.constraint
        
        sp.copyto(self.img, next_img)
        self.transforms -= mean_transform

def estimate_transform(ksp, mps, shot_mask, image, kgrid, rkgrid, damp, convergence_flags,
                       residual_buf, partial_buf, gradient_buf, hessian_buf,
                       transforms=None, max_iter=10, tol=1e-6, use_ga=False
                       ):
    
    xp = cp.get_array_module(ksp)
    nshots = shot_mask.shape[0]

    for s in range(nshots):
        #First projection to get current error
        T = RigidTransformCudaOptimzied(transforms[s], kgrid, rkgrid)
        xp.multiply(mps, T.apply(image), out=residual_buf)
        residual_buf = gpu_fftn(residual_buf, axes=(-3,-2,-1), norm='ortho', overwrite_x=True)
        xp.subtract(residual_buf, ksp, out=residual_buf)
        xp.multiply(residual_buf, shot_mask[s], out=residual_buf)
        #resid = A * F * S * resid - (A*ksp)
        e0 = xp.vdot(residual_buf, residual_buf).real

        #For each of the 6 parameters we calculate the partial derivative and cache
        #correct passed in buffer that way allocation of these only happens once in the joint class
        dT = RigidTransformDerivativeCuda(image.shape, transforms[s], kgrid, rkgrid)
        for p_idx in range(6):
            #T = RigidTransformDerivative(self.img_shape, self.img_shape, p_idx, self.transforms[shot_idx], self.kgrid, self.rkgrid)
            #partials.append(A * F * S * T * self.img)
            
            #partial_buf[p_idx] = A * F * S * (dT.apply(self.img, p_idx))
            xp.multiply(mps, dT.apply(image, p_idx), out=partial_buf[p_idx])
            partial_buf[p_idx] = gpu_fftn(partial_buf[p_idx], axes=(-3,-2,-1), norm='ortho', overwrite_x=True)
            xp.multiply(shot_mask[s], partial_buf[p_idx], out=partial_buf[p_idx])
            
            gradient_buf[p_idx] = xp.vdot(residual_buf, partial_buf[p_idx]).real
        
        
        for i in range(6):
            pi = partial_buf[i]
            for j in range(i, 6):
                val = xp.vdot(pi, partial_buf[j]).real
                if i == j:
                    hessian_buf[i,j] = val + damp[s]
                else:
                    hessian_buf[i, j] = val
                    hessian_buf[j, i] = val

        #Solve for delta step and estimate the Hessian matrix from Jacobians, then compute next error
        delta = xp.linalg.solve(hessian_buf, gradient_buf)

        # --- Geodesic acceleration (GA) ---
        #Still experimental and needs testing!
        if use_ga:
            jv_buf = xp.zeros_like(residual_buf)
            eps = 1e-4  # you can tune; 1e-3..1e-5 typical

            v = -delta  # actual increment since you do transforms[s] - delta

            # 1) Compute Jv = sum_p v[p] * J_p using existing partial_buf[p]
            #    jv_buf is an extra complex buffer passed in (same shape as residual_buf)
            jv_buf.fill(0)
            for p_idx in range(6):
                # jv_buf += v[p_idx] * partial_buf[p_idx]
                xp.add(jv_buf, v[p_idx] * partial_buf[p_idx], out=jv_buf)

            # 2) Compute J_eps v at theta_eps = theta + eps*v WITHOUT storing all J_eps columns.
            theta_eps = transforms[s] + eps * v
            dT_eps = RigidTransformDerivativeCuda(image.shape, theta_eps, kgrid, rkgrid)

            # reuse residual_buf as accumulator for jv_eps
            residual_buf.fill(0)
            for p_idx in range(6):
                # temp = A*F*S*(dT_eps.apply(image,p))
                tmp = residual_buf  # just a name; we overwrite it each loop then add into accumulator
                xp.multiply(mps, dT_eps.apply(image, p_idx), out=tmp)
                tmp = gpu_fftn(tmp, axes=(-3,-2,-1), norm='ortho', overwrite_x=True)
                xp.multiply(shot_mask[s], tmp, out=tmp)

                # residual_buf_accum += v[p_idx] * tmp
                xp.add(residual_buf, v[p_idx] * tmp, out=residual_buf)

            # 3) Directional derivative: J'v ≈ (J_eps v - Jv)/eps (stored in residual_buf)
            xp.subtract(residual_buf, jv_buf, out=residual_buf)
            residual_buf /= eps  # now residual_buf holds Jprime_v

            # 4) Compute rhs = -J^T (J'v)
            #    Using inner products with existing J columns in partial_buf
            #    (this is cheap: 6 dot products)
            for p_idx in range(6):
                gradient_buf[p_idx] = -xp.vdot(residual_buf, partial_buf[p_idx]).real

            # 5) Solve (J^T J + λI) a = rhs
            a = xp.linalg.solve(hessian_buf, gradient_buf)

            # 6) Accept only if acceleration is not crazy (stability gate)
            if xp.linalg.norm(a) <= 1.0 * xp.linalg.norm(v):
                v_ga = v + 0.5 * a
            else:
                v_ga = v

            next_transform = transforms[s] + v_ga
        else:
            next_transform = transforms[s] - delta
        # --- end GA ---

        #next_transform = transforms[s] - delta
        
        #Calculate the resid for the new transform
        #T = RigidTransform(self.img_shape, self.img_shape, next_transform, self.kgrid, self.rkgrid)
        T = RigidTransformCudaOptimzied(next_transform, kgrid, rkgrid)
        #resid = (A * F * S * T * self.img) - (A*self.ksp)
        xp.multiply(mps, T.apply(image), out=residual_buf)
        residual_buf = gpu_fftn(residual_buf, axes=(-3,-2,-1), norm='ortho', overwrite_x=True)
        xp.subtract(residual_buf, ksp, out=residual_buf)
        xp.multiply(residual_buf, shot_mask[s], out=residual_buf)
        
        #next_error = xp.linalg.norm(resid)
        e1 = xp.vdot(residual_buf, residual_buf).real

        #Check candidate for update and if so check partial convergence for this motion state
        if e1 < e0:
            if xp.all(xp.abs(transforms[s] - next_transform) < tol):
                convergence_flags[s] = True
            transforms[s] = next_transform.copy()
            damp[s] = xp.maximum(damp[s] / 5, 1e-4)
        else:
            damp[s] = xp.minimum(damp[s] * 1.5, 1e16)

    #Post transform update we subtract the mean to prevent drifting of the image
    mean_transform = xp.mean(transforms, axis=0)
    T = RigidTransformCudaOptimzied(mean_transform, kgrid, rkgrid)
    out = T.apply(image)
    xp.subtract(transforms, mean_transform, out=transforms)

    return out