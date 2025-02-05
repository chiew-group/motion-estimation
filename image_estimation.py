import sigpy as sp
from transform import RigidTransform

class ImageEstimation(sp.app.App):
    def __init__(        
        self,
        ksp,
        mps,
        shot_mask,
        transforms,
        kgrid,
        rkgrid,
        x=None,
        P=None,
        constraint=None,
        max_iter=10,
        tol=0,
        device=sp.cpu_device,
        comm=None,
        show_pbar=True,
        leave_pbar=True,
        ):
        
        #ksp = sp.to_device(ksp, device)
        
        self.constraint = constraint
        if self.constraint is not None:
            self.constraint = sp.to_device(constraint, device)
        self.transforms = sp.to_device(transforms, device)
        E = self.encode(self.transforms, mps, shot_mask, kgrid, rkgrid, comm)

        if x is None:
            with device:
                self.x = device.xp.zeros(ksp.shape[1:], dtype=ksp.dtype)
        else:
            self.x = sp.to_device(x, device)
        ksp_adj = E.H * sp.to_device(ksp, device)
        alg = sp.alg.ConjugateGradient(E.N, ksp_adj, self.x, P=P, max_iter=max_iter, tol=tol)

        super().__init__(alg, show_pbar=show_pbar, leave_pbar=leave_pbar)
    
    def _post_update(self):
        if self.constraint is not None:
            self.alg.x *= self.constraint

    def _output(self):
        return self.alg.x

    def _summarize(self):
        if self.show_pbar:
            self.pbar.set_postfix(
                    obj="{0:.2E}".format(self.alg.resid)
                )
            
    def encode(self, transforms, mps, shot_mask, kgrid, rkgrid, comm=None):
        img_shape = mps.shape[1:]
        #S = sp.linop.Multiply(img_shape, mps)
        #F = sp.linop.FFT(S.oshape, axes=[-3,-2,-1])
        list_of_ops = []
        for shot_idx in range(len(transforms)):
            S = sp.mri.linop.Sense(mps, weights=shot_mask[shot_idx])
            #A = sp.linop.Multiply(F.oshape, shot_mask[shot_idx])
            T = RigidTransform(img_shape, img_shape, transforms[shot_idx], kgrid, rkgrid)
            list_of_ops.append(S * T) 
        E = sp.linop.Add(list_of_ops)
        if comm is not None:
            C = sp.linop.AllReduce(S.oshape, comm, in_place=True)
            E = C * E
        E.repr_str = "Fwd Encoding"
        return E