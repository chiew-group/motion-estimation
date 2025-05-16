import sigpy as sp
import sigpy.mri
from transform import RigidTransform, RigidTransformCudaOptimzied
from cupyx.scipy.fft import fftn as gpu_fftn, ifftn as gpu_ifftn
import cupy as cp
from tqdm import tqdm

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

def estimate_image_cg(
    ksp,
    mps,
    shot_mask,
    transforms,
    kgrid,
    rkgrid,
    x0=None,
    max_iter=10,
    tol=1e-6,
):
    """
    Estimate the image x by solving A x = b with Conjugate Gradient,
    where A applies Sense, RigidTransform, shot masking, and FFTs.

    Parameters:
    -----------
    ksp: array_like, shape (nshots, ncoils, *img_shape)
        Measured k-space data.
    mps: array_like, shape (nshots, ncoils, *img_shape)
        Coil sensitivity maps per shot.
    shot_mask: array_like, shape (nshots, *img_shape)
        Binary or weighting mask per shot in k-space.
    transforms: list of parameters
        Parameters for rigid_transform(image, params, kgrid, rkgrid).
    kgrid, rkgrid: arrays
        Grids for rigid_transform (unused here until you implement it).
    rigid_transform: callable
        Function(image, params, kgrid, rkgrid) -> transformed image.
    x0: array_like, optional
        Initial guess for image. Defaults to zeros.
    max_iter: int, tolerance for CG iterations.
    tol: float, stopping tolerance on residual norm.

    Returns:
    --------
    x: array, shape img_shape
        Reconstructed image.
    """
    # Choose array module
    #xp = sp.Device(0).xp  # Use GPU if available, otherwise CPU
    xp = cp.get_array_module(ksp)

    nshots = transforms.shape[0]
    ncoils = ksp.shape[0]
    img_shape = ksp.shape[1:]

    # Pre-allocate solution and scratch buffers
    x = xp.zeros(img_shape, dtype=ksp.dtype) if x0 is None else xp.array(x0)
    b = xp.zeros_like(x)
    r = xp.empty_like(x)
    p = xp.empty_like(x)
    Ap = xp.empty_like(x)

    Tops = [RigidTransformCudaOptimzied(p, kgrid, rkgrid) for p in transforms]

    temp_buf = xp.empty_like(ksp)
    axes3 = tuple(range(-3, 0))

    # Precompute right-hand side b = Sum_shots S^H ksp
    for i in range(nshots):
        xp.multiply(shot_mask[i], ksp, out=temp_buf) 
        # IFFT over spatial axes for each coil
        if xp is cp:
                temp_buf = gpu_ifftn(
                    temp_buf,
                    axes=tuple(range(-3, 0)),
                    overwrite_x=True, norm='ortho'
                )
        b += Tops[i].adjoint(xp.sum(xp.conj(mps) * temp_buf, axis=0))
    
    # Define linear operator A(x) using out=Ap
    def A_func(x_img, out):
        out.fill(0)
        for i in range(nshots):
            # 1) Rigid transformation on image
            T = Tops[i]
            Ex = T.apply(x_img)
            # 2) Multiply by coil sensitivities
            Ex = mps * Ex  # shape (ncoils, *img_shape)
            # 3) FFT + shot mask
            Ex = gpu_fftn(
                Ex,
                axes=tuple(range(-len(img_shape), 0)),
                overwrite_x=True, norm='ortho'
            )
            Ex *= shot_mask[i]
            # 4) Inverse FFT
            Ex = gpu_ifftn(
                Ex,
                axes=tuple(range(-len(img_shape), 0)),
                overwrite_x=True, norm='ortho'
            )
            # 5) Adjoint: sum conj(mps) * img_pred
            xp.add(out, T.adjoint(xp.sum(xp.conj(mps) * Ex, axis=0)), out=out) # Apply adjoint of rigid transform

    # --- Conjugate Gradient ---
    # Initial residual r = b - A(x)
    # No preconditioning in this at the moment
    A_func(x, out=Ap)
    r[:] = b - Ap
    p[:] = r
    rsold = xp.real(xp.vdot(r, r))
    resid = rsold.item() ** 0.5 
    temp = xp.empty_like(x)
    for it in tqdm(range(max_iter)):
        A_func(p, out=Ap)
        pAp = xp.real(xp.vdot(p, Ap)).item()
        alpha = rsold / pAp
        
        # x = x + alpha * p
        xp.multiply(p, alpha, out=temp)
        xp.add(x, temp, out=x)
        
        # r = r - alpha * Ap
        xp.multiply(Ap, alpha, out=temp) # temp = alpha * Ap
        xp.subtract(r, temp, out=r)

        rsnew = xp.real(xp.vdot(r, r))
        if xp.sqrt(xp.abs(rsnew)) < tol:
            break

        #p = r + (rsnew / rsold) * p
        beta = rsnew / rsold
        xp.multiply(p, beta, out=temp) # temp = beta * p
        xp.add(r, temp, out=p) 
        rsold = rsnew
        resid = rsold.item() ** 0.5

    print(resid)
    return x