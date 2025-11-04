import sigpy as sp
import sigpy.mri
from transform import RigidTransform, RigidTransformCudaOptimzied
from cupyx.scipy.fft import fftn as gpu_fftn, ifftn as gpu_ifftn
from utils import compute_transform_grids_voxel
import cupy as cp
from tqdm import tqdm
import numpy as np
import nibabel as nib
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

def pad_to_square(img3d):
    x, y, z = img3d.shape
    target_size = max(x, y, z)

    pad_x = (target_size - x) // 2
    pad_y = (target_size - y) // 2
    pad_z = (target_size - z) // 2

    padded = np.pad(
        img3d,
        ((pad_x, target_size - x - pad_x),
         (pad_y, target_size - y - pad_y),
         (pad_z, target_size - z - pad_z)),
        mode='constant'
    )
    return padded

def show_mid_slices(img3d, save_path=None):
    img3d = np.abs(img3d)
    img3d = pad_to_square(img3d)

    x, y, z = img3d.shape
    mid_x, mid_y, mid_z = x // 2, y // 2, z // 2

    # Extract slices
    sagittal = np.rot90(np.rot90(np.flipud(img3d[mid_x+3, :, :])))     # Rotate 180 and flip vertically
    coronal  = np.rot90(np.rot90(img3d[:, mid_y, :]))      # Rotate 180 degrees
    axial    = img3d[:, :, mid_z]                          # No rotation

    slices = [sagittal, coronal, axial]

    # Plot without gaps
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=300)
    for ax, slc in zip(axs, slices):
        ax.imshow(slc.T, cmap='gray', origin='lower')
        ax.axis('off')

    # Remove white space between images
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved recon image midsections to {save_path}")
    else:
        plt.show()


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
    P,
    M,
    x0=None,
    max_iter=100,
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
    for it in range(max_iter):
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
        if xp.sqrt(xp.abs(rsnew)) < tol:
            break

        #p = r + (rsnew / rsold) * p
        beta = rsnew / rsold
        xp.multiply(p, beta, out=temp) # temp = beta * p
        xp.add(z, temp, out=p) 
        rsold = rsnew
        resid = rsold.item() ** 0.5
        x *= M
    return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Coil compression for k-space and maps.")
    parser.add_argument('--ksp', type=str, required=True, help='Path to k-space .npy file')
    parser.add_argument('--mps', type=str, required=True, help='Path to sensitivty maps .npy file')
    parser.add_argument('--t0',  type=str, required=True, help='Initial transform parameters .npy file')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory')
    #parser.add_argument('--shots', type=int, required=True, help='Number of motion states/shots to recon for')
    #parser.add_argument('--lowres', type=float, default=1.0, help='Downsample factor')
    parser.add_argument('--max_iter', type=int, default=10, help='Max iterations for joint recon')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ksp = np.load(args.ksp)
    mps = np.load(args.mps)
    t0 = np.load(args.t0)

    full_res = ksp.shape[1:]
    ncoils = ksp.shape[0]
    num_shots = t0.shape[0]
    shot_size = ksp.shape[2] // num_shots

    #Create sequential shot mask based on indexing axis 1
    #This can be modified later on as a todo
    #Also we can change this so as to input your own mask as a npy array
    rss_ksp = np.sum(np.abs(ksp)**2, axis=0, dtype=bool)
    shot_mask = np.zeros([num_shots, *ksp.shape[1:]], dtype=bool)
    for s in range(num_shots):
        shot_mask[s, :, s*shot_size:(s+1)*shot_size] = rss_ksp[:, s*shot_size:(s+1)*shot_size, :]

    #Unshift in preperations for estimation
    ksp = np.fft.fftshift(ksp)
    mps = np.fft.fftshift(mps)
    shot_mask = np.fft.fftshift(shot_mask)


    ksp = cp.asarray(ksp)
    mps = cp.asarray(mps)
    shot_mask = cp.asarray(shot_mask)
    t0 = cp.asarray(t0)

    P = cp.ones(full_res)
    #M = cp.ones(full_res)
    #P = 1 / (cp.sum(cp.abs(mps) ** 2, axis=0) + 1e-6)
    M = (cp.sum(cp.abs(mps) ** 2, axis=0) > 0.1)

    #RECON
    kgrid, rkgrid = compute_transform_grids_voxel(full_res, [0.8, 0.8, 0.8], xp=cp)
    final = estimate_image_cg(ksp, mps, shot_mask, t0, kgrid, rkgrid,P,M, max_iter=args.max_iter, tol=1e-12)

    final = cp.fft.fftshift(final)
    final = cp.asnumpy(final)

    #Plot summaries and save images
    #plot_joint_recon_summary(app.objective_history, app.transform_history, save_path=outdir / 'summary')
    show_mid_slices(pad_to_square(final), outdir / "final.png")
    np.save(outdir / "final.npy", final)
    affine = np.eye(4, dtype=np.float32)
    nifti_final = nib.Nifti1Image(final, affine)
    nib.save(nifti_final, outdir / 'final.nii.gz')


    #np.save(outdir / "recon_lowres.npy", img_lowres)
    #np.save(outdir / f"recon_{int(args.lowres)}_ds_{num_shots}_shots", recon)
    #np.save(outdir / f"transforms_{int(args.lowres)}_ds_{num_shots}_shots", t)  # shape: (shots, iters, 6)
    #np.save(outdir / "objective_loss.npy", loss_history)