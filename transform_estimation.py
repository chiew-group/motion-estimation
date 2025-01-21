import sigpy as sp
from transform import RigidTransform, RigidTransformDerivative

class LMAlgorithm(sp.alg.Alg):
    def __init__(self, ksp, mps, shot_mask, 
                 transforms, img, kgrid, rkgrid, 
                 damp, convergence_flags, max_iter=100, tol=1e-4):
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
        

        #Algorithm State
        self.num_shots = len(transforms)
        self.img_shape = self.img.shape
        
        super().__init__(max_iter)

    def _update(self):
        xp = sp.get_array_module(self.img)
        S = sp.linop.Multiply(self.img_shape, self.mps)
        F = sp.linop.FFT(S.oshape, axes=[-3,-2,-1])
        
        for shot_idx in range(self.num_shots):
            A = sp.linop.Multiply(F.oshape, self.shot_mask[shot_idx])
            T = RigidTransform(self.img_shape, self.img_shape, self.transforms[shot_idx], self.kgrid, self.rkgrid)

            #Calculate the resid error or energy
            resid = (A * F * S * T * self.img) - self.ksp
            current_error = xp.linalg.norm(resid)

            #Compute the partial derive kspaces
            partials = []
            gradient = [] 
            for p_idx in range(6):
                T = RigidTransformDerivative(self.img_shape, self.img_shape, p_idx, self.transforms[shot_idx], self.kgrid, self.rkgrid)
                partials.append(A * F * S * T * self.img)
                
                #Compute the gradient of the motion state while we are at it
                gradient.append(xp.sum(xp.real(resid.conj() * partials[-1])))
            gradient = xp.array(gradient)

            #Approximate the Hessian matrix from partials
            hessian = xp.zeros((6,6))
            for row in range(6):
                for col in range(row, 6):
                    hessian[row, col] = xp.sum(xp.real(partials[row].conj() * partials[col]))
            i_lower = xp.tril_indices(6, -1)
            hessian[i_lower] = hessian.T[i_lower]
    
            hessian += self.damp[shot_idx] * xp.eye(6)
            delta = xp.linalg.lstsq(hessian, gradient, rcond=None)[0]
            next_transform = self.transforms[shot_idx] - delta
            
            #Calculate the resid for the new transform
            T = RigidTransform(self.img_shape, self.img_shape, next_transform, self.kgrid, self.rkgrid)
            resid = (A * F * S * T * self.img) - self.ksp
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
                 damp, convergence_flags, constraint, max_iter=100, tol=1e-6, show_pbar=True, leave_pbar=True):
        self.transforms = transforms
        self.kgrid = kgrid
        self.rkgrid = rkgrid
        self.img = img
        self.img_shape = img.shape
        self.constraint = constraint
        alg = LMAlgorithm(ksp, mps, shot_mask, 
                          transforms, img, kgrid, rkgrid, 
                          damp, convergence_flags, max_iter=max_iter, tol=tol)
        super().__init__(alg, show_pbar=show_pbar, leave_pbar=leave_pbar)
    
    def _post_update(self):
        mean_trasnform = self.transforms.mean(axis=0)
        T = RigidTransform(self.img_shape, self.img_shape, mean_trasnform, self.kgrid, self.rkgrid)
        next_img = (T * self.img) * self.constraint
        sp.copyto(self.img, next_img)
        self.transforms -= mean_trasnform