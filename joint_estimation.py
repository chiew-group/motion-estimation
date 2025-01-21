import sigpy as sp
import numpy as np
from image_estimation import ImageEstimation
from transform_estimation import TransformEstimation
import argparse

class JointEstimation(sp.app.App):
    def __init__(
        self, 
        ksp, 
        mps, 
        shot_mask,
        kgrid,
        rkgrid,
        img=None, 
        transforms=None,
        constraint=None,
        P=None,
        max_cg_iter=3,
        max_nm_iter=1,
        max_joint_iter=100,
        tol=1e-6,
        device=sp.cpu_device,
        verbose=False,
        comm=None,
        show_pbar=True
    ):
        
        #Configureation Parameters
        self.P = P
        self.max_cg_iter = max_cg_iter
        self.max_nm_iter = max_nm_iter
        self.max_joint_iter = max_joint_iter
        self.tol = tol
        self.device = device
        self.verbose = verbose
        self.comm = comm
        self.num_shots = len(shot_mask)
        self.img_shape = ksp.shape[1:]
        self.objective_values = []
        self.kgrid = kgrid
        self.rkgrid = rkgrid
        #Device dependant parameters
        xp = self.device.xp
        with self.device:
            #Const required input data for the reconstruction
            self.ksp =       sp.to_device(ksp, device=device)
            self.mps =       sp.to_device(mps, device=device)
            self.shot_mask = sp.to_device(shot_mask, device=device)

            if img is None:
                self.img = xp.zeros(self.img_shape, dtype=self.ksp.dtype)
            else:
                self.img = sp.to_device(img, device=device)

            if transforms is None:
                self.transforms = xp.zeros((self.num_shots, 6), dtype=float)
            else:
                self.transforms = sp.to_device(transforms, device=device)

            if constraint is None:
                self.constraint = xp.ones_like(self.img)
            else:
                self.constraint = sp.to_device(constraint, device=device)
            
            self.damp = xp.ones((self.num_shots))
            self.convergence_flags = xp.zeros((self.num_shots), dtype=bool)

            #self.t_iterations = [self.transforms.get()]

            if comm is not None:
                show_pbar = show_pbar and comm.rank == 0

        alg = sp.alg.AltMin(self._minX, self._minT, self.max_joint_iter)
        super().__init__(alg, show_pbar=show_pbar)
    
    def _summarize(self):
        if self.show_pbar:
            self.pbar.set_postfix(
                max_voxel_change="{0:.2E}".format(
                    sp.backend.to_device(3, sp.backend.cpu_device)
                )
            )

    def _output(self):
        return self.img, self.transforms#, np.array(self.objective_values), np.stack(self.t_iterations, axis=-1)
    
    def _post_update(self):        
        if self.verbose:
            xp = self.device.xp
            estimate = self.transforms.copy()
            estimate[:, 3:] *= (180 / xp.pi)
            print('\n')
            print(f'Iteration {self.alg.iter} |')
            for shot in range(len(estimate)):
                print(f'MotionState {shot+1}: {xp.array_str(estimate[shot], precision=2)}')
            #print(f'Objective: {objective_all_shot(self.img, self.ksp, self.mps, self.shot_mask, self.transforms, self.kgrid, self.rkgrid)}')
            print('-' * 80)
        
        if self.convergence_flags.all():
            self.alg.done = lambda: True
    
    def objective(self):
        pass
    
    def _minX(self):
        ImageEstimation(
            self.ksp,
            self.mps,
            self.shot_mask,
            self.transforms,
            self.kgrid,
            self.rkgrid,
            self.img,
            self.P,
            self.constraint,
            device=self.device,
            max_iter=self.max_cg_iter,
            comm=self.comm,
            show_pbar=False,
            tol=0
            ).run()
    
    def _minT(self):
        TransformEstimation(
            self.ksp,
            self.mps,
            self.shot_mask,
            self.transforms,
            self.img,
            self.kgrid,
            self.rkgrid,
            self.damp,
            self.convergence_flags,
            self.constraint,
            max_iter=self.max_nm_iter,
            tol=self.tol,
            show_pbar=False
        ).run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint estimation for motion corruption')

    parser.add_argument('--max_iter', type=int, default=1000, help='Max number of joint estimation iterations.')
    parser.add_argument('--tol', type=int, default=1e-6, help='Tolerance for joint convergence.')
    parser.add_argument('--init_img_file', type=str, default=None, help='Inital img input file.')
    parser.add_argument('--init_transforms_file', type=str, default=None, help='Inital guess transform params file.')
    parser.add_argument('--device', type=int, default=-1, help='Device for computation.')
    parser.add_argument('--downsample_factor', type=int, default=0, 
                        help='Downsample the data by this factor along motion axis.')

    parser.add_argument('ksp_file', type=str, help='K-space file.')
    parser.add_argument('mps_file', type=str, help='Coil maps file.')
    parser.add_argument('img_file', type=str, help='Output image file.')
    parser.add_argument('estimates_file', type=str, help='Output estimates file.')
    parser.add_argument('motion_states', type=int, help='Number of line scanes to group for motion estiamte.')
    parser.add_argument('motion_axis', type=int, 
                        help='0: Par, 1: Lin, 2: Col, used for sampling of states along an axis.')

    args = parser.parse_args()

    ksp = np.load(args.ksp_file)
    mps = np.load(args.mps_file)

    max_norm = np.max(np.abs(sp.ifft(ksp, axes=[-3,-2,-1])))
    ksp /= max_norm

    full_res_shape = ksp.shape
    full_res_img_shape = ksp.shape[1:]

    num_channels = ksp.shape[0]
    downsample_factor = args.downsample_factor
    desired_axis = args.motion_axis

    #Downsample
    if downsample_factor > 0:
        
        downsample_shape = []
        for axis in range(3):
            if not axis == desired_axis:
                downsample_shape.append(full_res_img_shape[axis] // downsample_factor) 
            else:
                downsample_shape.append(full_res_img_shape[axis])
        downsample_shape = [num_channels] + downsample_shape 

        ksp = sp.resize(ksp, downsample_shape)
        mps = sp.ifft(sp.resize(sp.fft(mps, axes=[-3,-2,-1]), downsample_shape), axes=[-3,-2,-1])

    num_shots = args.motion_states
    shot_mask = np.zeros([num_shots] + downsample_shape[1:], dtype=bool)
    bin_size = downsample_shape[1+desired_axis] // num_shots
    
    rss_ksp = (sp.rss(ksp) > 0)
    for shot in range(num_shots):
        indicies = [slice(None)] * len(downsample_shape[1:])
        indicies[desired_axis] = slice(shot * bin_size, (shot+1) * bin_size)
        idx = [shot] + indicies
        shot_mask[*idx] = rss_ksp[*indicies]
    
    print(f"ksp shape: {ksp.shape}")
    print(f"mps shape: {mps.shape}")
    print(f"msk shape: {shot_mask.shape}")
    print(f"Resolution downsample factor: {downsample_factor}")
    print(f"Number of motion states is {num_shots} each of size {bin_size}")
    
    #Recon
    device = sp.Device(args.device)

    from utils import compute_transform_grids

    rss = sp.rss(mps, axes=(0,))
    M = rss > np.max(rss) * 0.1
    P = sp.linop.Multiply(ksp.shape[1:], 1 / (np.sum(np.abs(mps) ** 2, axis=0) + 1e-3))

    kgrid, rkgrid = compute_transform_grids(full_res_img_shape, downsample_shape[1:], device=device)

    recon, estimates = JointEstimation(ksp, mps, shot_mask, kgrid, rkgrid, 
                                    device=device, P=P, constraint=M,
                                    max_joint_iter=args.max_iter, tol=args.tol).run()

    recon = sp.to_device(recon)
    estimates = sp.to_device(estimates)
     
    recon = sp.ifft(sp.resize(sp.fft(recon), full_res_img_shape))

    np.save(args.img_file, recon)
    np.save(args.estimates_file, estimates)
