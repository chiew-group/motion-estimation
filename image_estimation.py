import sigpy as sp
import sigpy.mri
from transform import RigidTransformCudaOptimzied
from cupyx.scipy.fft import fftn as gpu_fftn, ifftn as gpu_ifftn
from utils import compute_transform_grids_voxel
import cupy as cp
from tqdm import tqdm
import numpy as np

def estimate_image_cg(
    ksp,
    mps,
    shot_mask,
    transforms,
    kgrid,
    rkgrid,
    P,
    M,
    x0=None,
    max_iter=100,
    tol=1e-6,
    show_pbar=True
):
    """
    Reconstruct an image via Conjugate Gradient solving A x = b.

    The linear operator A applies, for each shot: rigid transform ->
    multiply by coil sensitivity maps -> forward FFT -> shot mask.
    The adjoint operator is used to build the right-hand side b and to
    apply A in each CG iteration.

    Parameters
    ----------
    ksp : array-like
        Measured k-space data. Shape expected (ncoils, H, W[, D]).
    mps : array-like
        Coil sensitivity maps with shape compatible with `ksp` (ncoils, H, W[, D]).
    shot_mask : array-like
        Per-shot k-space sampling masks. Shape (nshots, H, W[, D]) or (nshots, H, W, 1).
    transforms : array-like
        Rigid transform parameters with shape (nshots, 6). Each row is [tx,ty,tz,rx,ry,rz]
        where rotations are in radians.
    kgrid, rkgrid : array-like
        Grids required by the rigid transform operators (from compute_transform_grids_voxel).
    P : array-like
        Diagonal preconditioner (image-space) applied as z = P * r in CG. Should match image shape.
    M : array-like
        Image-space mask/support (boolean-like). Applied to b and to A(x) outputs.
    x0 : array-like, optional
        Initial image guess. If None, zeros are used.
    max_iter : int, optional
        Maximum number of CG iterations (default: 100).
    tol : float, optional
        Stopping tolerance on the preconditioned residual (default: 1e-6).
    show_pbar : bool, optional
        Whether to display a progress bar during CG (default: True).

    Returns
    -------
    x : array-like
        Reconstructed image of shape `img_shape`. Returned array lives on the same
        array module (NumPy or CuPy) as the input `ksp`.

    Notes
    -----
    - All inputs to this function are expected to be in a uncentered fftshift format.
    - Uses `RigidTransformCudaOptimzied` for fast per-shot transform operations.
    - The function automatically chooses NumPy or CuPy based on `ksp` with
      `cp.get_array_module(ksp)`, and uses GPU FFTs when available.
    - Ensure `P` and `M` have shapes matching the image-space (H, W[, D]).
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

    Tops = [RigidTransformCudaOptimzied(tr, kgrid, rkgrid) for tr in transforms]

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
        b *= M

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
                axes=(-3,-2,-1),
                overwrite_x=True, norm='ortho'
            )
            Ex *= shot_mask[i]
            # 4) Inverse FFT
            Ex = gpu_ifftn(
                Ex,
                axes=(-3,-2,-1),
                overwrite_x=True, norm='ortho'
            )
            # 5) Adjoint: sum conj(mps) * img_pred
            xp.add(out, T.adjoint(xp.sum(xp.conj(mps) * Ex, axis=0)), out=out) # Apply adjoint of rigid transform
        out *= M
    # --- Conjugate Gradient ---
    # Initial residual r = b - A(x)
    # No preconditioning in this at the moment
    A_func(x, out=Ap)
    r[:] = b - Ap
    z = P * r
    p = z.copy()
    rsold = xp.real(xp.vdot(r, z))
    resid = rsold.item() ** 0.5 
    temp = xp.empty_like(x)
    pbar = tqdm(range(max_iter), desc="CG-Sense", disable=not show_pbar)
    for it in pbar:
        A_func(p, out=Ap)
        pAp = xp.real(xp.vdot(p, Ap)).item()
        alpha = rsold / pAp
        
        # x = x + alpha * p
        xp.multiply(p, alpha, out=temp)
        xp.add(x, temp, out=x)

        # r = r - alpha * Ap
        xp.multiply(Ap, -alpha, out=temp) # temp = alpha * Ap
        xp.add(r, temp, out=r)
        z = P * r
        rsnew = xp.real(xp.vdot(r, z))
        pbar.set_postfix(res=f"{resid:.3e}")
        if xp.sqrt(xp.abs(rsnew)) < tol:
            break

        #p = r + (rsnew / rsold) * p
        beta = rsnew / rsold
        xp.multiply(p, beta, out=temp) # temp = beta * p
        xp.add(z, temp, out=p) 
        rsold = rsnew
        resid = rsold.item() ** 0.5

    return x