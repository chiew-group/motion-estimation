import sigpy as sp
from transform import RigidTransformDerivativeCuda, RigidTransformCudaOptimzied
import cupy as cp
from cupyx.scipy.fft import fftn as gpu_fftn, ifftn as gpu_ifftn
import numpy as np 

def estimate_transform(ksp, mps, shot_mask, image, kgrid, rkgrid, damp, convergence_flags,
                       residual_buf, partial_buf, gradient_buf, hessian_buf,
                       transforms=None, max_iter=10, tol=1e-6, use_ga=False
                       ):
    """
    Estimates motion transforms for each motion state one at a time. The transform is then updated inplace and a zero mean updated transform of the image is returned.

    Args:
        ksp [ncoils, kx, ky, kz] complex64: K-space.
        mps [ncoils, kx, ky, kz] complex64: Sensitivity maps.
        shot_mask [nshots, x, y, z] bool: Sampling mask for each motion state.
        image [x, y, z] complex64: Current image used in motion state estimation, will be update in place with new transforms.
        kgrid: Dictionary holding the precomputed grid sizes for kspace.
        rkgrid: Dictionary holding the precomputed grid sizes for spatial-spectral space.
        damp [nshots] float: Each motion state will have a dampening parameter that changes hessian estimation and will be updated inplace.
        convergence_flags [nshots] bool: Flags for determining convergence of a particular motion state, all states need to achieve convergence.
        residual_buf [ncoils, kx, ky, kz] complex64: Pre allocated buffer that will hold the residual results for intermediate calculations.
        partial_buf [6, ncoils, kx, ky, kz] complex64: Pre allocated buffer that will hold all the partial derivatives for intermediate calculations.
        gradient_buf [6] float: Pre allocated buffer holding intermediate results for the dot product of residual and gradient buffers.
        hessian_buf [6,6] float: Pre allocated buffer to hold the estimated hessian of a single motion state estiamtion.
        transforms [nshots, 6] float: Current set of transforms, these will be update inplace or initialized to zero if None .
        max_iter: Number of iterations for motion estimation, typically only set to one for joint reconstructions.
        tol: Tolerance for all motion states.
        use_ga: Boolean flag to determine if geodesic acceleration should be used in newton's method steps.

    Returns:
        image: Updated image based on new transform estimations.
        [transforms]: Updated transforms only done inplace.
    """
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

        #Calculate the resid for the new transform
        T = RigidTransformCudaOptimzied(next_transform, kgrid, rkgrid)
        xp.multiply(mps, T.apply(image), out=residual_buf)
        residual_buf = gpu_fftn(residual_buf, axes=(-3,-2,-1), norm='ortho', overwrite_x=True)
        xp.subtract(residual_buf, ksp, out=residual_buf)
        xp.multiply(residual_buf, shot_mask[s], out=residual_buf)

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