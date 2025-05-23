import sigpy as sp
from cupyx.scipy.fft import fftn as gpu_fftn, ifftn as gpu_ifftn, fft as gpu_fft, ifft as gpu_ifft
import cupy as cp

def _compute_shear_factor(rk, theta, mode, inverse=False):
    xp = sp.get_array_module(rk)
    if mode == 'tan':
        factor = xp.exp(1j * xp.tan(theta/2) * rk)
        return factor if not inverse else factor.conj()
    elif mode == 'sin':
        factor = xp.exp(-1j * xp.sin(theta) * rk)
        return factor if not inverse else factor.conj()
    else:
        raise ValueError("Mode must be either 'tan' or 'sin'!")

def _compute_translation_factor(kgrid, q1,q2,q3, inverse=False):
    k1, k2, k3 = kgrid
    xp = sp.get_array_module(k1)
    factor = xp.exp(-1j * (q1 * k1 + q2 * k2 + q3 * k3))
    return factor if not inverse else factor.conj()

def _compute_translation_factor_derivative(p_idx, kgrid, q1, q2, q3):
    u = _compute_translation_factor(kgrid, q1,q2,q3)
    return -1j * kgrid[p_idx] * u
    
def _compute_shear_factor_derivative(p_idx, rk, theta, vfactor, mode):
    xp = sp.get_array_module(rk)
    if mode == 'tan':
        factor = 1j * ((1 + (xp.tan(theta/2) ** 2)) / 2) * rk * vfactor
    elif mode == 'sin':
        factor = -1j * xp.cos(theta) * rk * vfactor
    else:
        raise ValueError("Mode must be either 'tan' or 'sin'!")
    return factor

def _rotation(input, theta, rk_tan, rk_sin, tan_axis, sin_axis):
    vtan = _compute_shear_factor(rk_tan, theta, 'tan')
    vsin = _compute_shear_factor(rk_sin, theta, 'sin')

    #Apply tan and sin shears
    input = sp.fft(input, axes=(tan_axis,))
    input *= vtan
    input = sp.ifft(input, axes=(tan_axis,))

    input = sp.fft(input, axes=(sin_axis,))
    input *= vsin
    input = sp.ifft(input, axes=(sin_axis,))

    input = sp.fft(input, axes=(tan_axis,))
    input *= vtan
    input = sp.ifft(input, axes=(tan_axis,))
            
    return input

def _rotation_derivative(input, p_idx, theta, rk_tan, rk_sin, tan_axis, sin_axis):
    vtan = _compute_shear_factor(rk_tan, theta, 'tan')
    vsin = _compute_shear_factor(rk_sin, theta, 'sin')
    vtan_derivative = _compute_shear_factor_derivative(p_idx, rk_tan, theta, vtan, 'tan')
    vsin_derivative = _compute_shear_factor_derivative(p_idx, rk_sin, theta, vsin, 'sin')

    input = sp.fft(input, axes=(tan_axis,))

    summand_1 = vtan_derivative * input
    summand_1 = sp.ifft(summand_1, axes=(tan_axis,))
    summand_1 = sp.fft(summand_1, axes=(sin_axis,))
    summand_1 *= vsin
    summand_1 = sp.ifft(summand_1, axes=(sin_axis,))
    summand_1 = sp.fft(summand_1, axes=(tan_axis,))
    summand_1 *= vtan

    summand_2 = vtan * input
    summand_2 = sp.ifft(summand_2, axes=(tan_axis,))
    summand_2 = sp.fft(summand_2, axes=(sin_axis,))
    summand_2 *= vsin_derivative
    summand_2 = sp.ifft(summand_2, axes=(sin_axis,))
    summand_2 = sp.fft(summand_2, axes=(tan_axis,))
    summand_2 *= vtan

    summand_3 = vtan * input
    summand_3 = sp.ifft(summand_3, axes=(tan_axis,))
    summand_3 = sp.fft(summand_3, axes=(sin_axis,))
    summand_3 *= vsin
    summand_3 = sp.ifft(summand_3, axes=(sin_axis,))
    summand_3 = sp.fft(summand_3, axes=(tan_axis,))
    summand_3 *= vtan_derivative

    input = summand_1 + summand_2 + summand_3
    input = sp.ifft(input, axes=(tan_axis,))

    return input

def _translation(input, kgrid, q1, q2, q3):
    u = _compute_translation_factor(kgrid, q1,q2,q3)
    input = sp.fft(input)
    input *= u
    input = sp.ifft(input)
    return input

def _translation_derivative(input, p_idx, kgrid, q1, q2, q3):
    u = _compute_translation_factor_derivative(p_idx, kgrid, q1,q2,q3)
    input = sp.fft(input)
    input *= u
    input = sp.ifft(input)
    return input

class RigidTransform(sp.linop.Linop):
    def __init__(self, oshape, ishape, parameters, kgrid, rkgrid, inverse=False):
        self.parameters = parameters
        self.img_axes = list(range(-len(ishape), 0))
        self.oshape = oshape
        self.ishape = ishape
        self.kgrid = kgrid
        self.rkgrid = rkgrid
        self.inverse = inverse
        super().__init__(oshape, ishape, "RigidTransform")

    def _apply(self, input):
        q1, q2, q3 = self.parameters[:3]
        theta1, theta2, theta3 = self.parameters[3:]
        
        for (theta, rk_tan, rk_sin, tan_axis, sin_axis) in zip([theta1, theta2, theta3], self.rkgrid[0], self.rkgrid[1], [1,2,0], [2,0,1]):
            vtan = _compute_shear_factor(rk_tan, theta, 'tan', self.inverse)
            vsin = _compute_shear_factor(rk_sin, theta, 'sin', self.inverse)

            #Apply tan and sin shears
            input = sp.fft(input, axes=(tan_axis,))
            input *= vtan
            input = sp.ifft(input, axes=(tan_axis,))

            input = sp.fft(input, axes=(sin_axis,))
            input *= vsin
            input = sp.ifft(input, axes=(sin_axis,))

            input = sp.fft(input, axes=(tan_axis,))
            input *= vtan
            input = sp.ifft(input, axes=(tan_axis,))

        #Apply the tranlation
        U = _compute_translation_factor(self.kgrid, q1,q2,q3, self.inverse)
        input = sp.fft(input)
        input *= U
        input = sp.ifft(input)

        return input

    def _adjoint_linop(self):
        return RigidTransformAdjoint(self.oshape, self.ishape, self.parameters, self.kgrid, self.rkgrid)

class RigidTransformAdjoint(sp.linop.Linop):
    def __init__(self, oshape, ishape, parameters, kgrid, rkgrid, inverse=True):
        self.parameters = parameters
        self.img_axes = list(range(-len(ishape), 0))
        self.oshape = oshape
        self.ishape = ishape
        self.kgrid = kgrid
        self.rkgrid = rkgrid
        self.inverse = inverse
        super().__init__(oshape, ishape, "RigidTransformAdjoint")

    def _apply(self, input):
        q1, q2, q3 = self.parameters[:3]
        theta1, theta2, theta3 = self.parameters[3:]
        
        #Apply the tranlation
        U = _compute_translation_factor(self.kgrid, q1,q2,q3, self.inverse)
        input = sp.fft(input)
        input *= U
        input = sp.ifft(input)
        
        for (theta, rk_tan, rk_sin, tan_axis, sin_axis) in zip([theta3, theta2, theta1], self.rkgrid[0][::-1], self.rkgrid[1][::-1], [0,2,1], [1,0,2]):
            vtan = _compute_shear_factor(rk_tan, theta, 'tan', self.inverse)
            vsin = _compute_shear_factor(rk_sin, theta, 'sin', self.inverse)

            #Apply tan and sin shears
            input = sp.fft(input, axes=(tan_axis,))
            input *= vtan
            input = sp.ifft(input, axes=(tan_axis,))

            input = sp.fft(input, axes=(sin_axis,))
            input *= vsin
            input = sp.ifft(input, axes=(sin_axis,))

            input = sp.fft(input, axes=(tan_axis,))
            input *= vtan
            input = sp.ifft(input, axes=(tan_axis,))



        return input

    def _adjoint_linop(self):
        return RigidTransform(self.oshape, self.ishape, self.parameters, self.kgrid, self.rkgrid)

class RigidTransformDerivative(sp.linop.Linop):
    def __init__(self, oshape, ishape, p_idx, parameters, kgrid, rkgrid):
        self.parameters = parameters
        self.img_axes = list(range(-len(ishape), 0))
        #self.oshape = oshape
        #self.ishape = ishape
        self.kgrid = kgrid
        self.rkgrid = rkgrid
        self.p_idx = p_idx
        super().__init__(oshape, ishape, f"RigidTransformDerivative wrt. {self.p_idx + 1}")

    def _apply(self, input):
        q1, q2, q3 = self.parameters[:3]
        theta1, theta2, theta3 = self.parameters[3:]
    
        if self.p_idx < 3:                
            input = _rotation(input, theta3, self.rkgrid[0][2], self.rkgrid[1][2], 0, 1)
            input = _rotation(input, theta2, self.rkgrid[0][1], self.rkgrid[1][1], 2, 0)
            input = _rotation(input, theta1, self.rkgrid[0][0], self.rkgrid[1][0], 1, 2)
            input = _translation_derivative(input, self.p_idx, self.kgrid, q1, q2, q3)
            return input

        elif self.p_idx == 3:
            input = _rotation(input, theta3, self.rkgrid[0][2], self.rkgrid[1][2], 0, 1)
            input = _rotation(input, theta2, self.rkgrid[0][1], self.rkgrid[1][1], 2, 0)
            input = _rotation_derivative(input, self.p_idx, theta1, self.rkgrid[0][0], self.rkgrid[1][0], 1, 2)
            input = _translation(input, self.kgrid, q1, q2, q3)
            return input

        elif self.p_idx == 4:
            input = _rotation(input, theta3, self.rkgrid[0][2], self.rkgrid[1][2], 0, 1)
            input = _rotation_derivative(input, self.p_idx, theta2, self.rkgrid[0][1], self.rkgrid[1][1], 2, 0)
            input = _rotation(input, theta1, self.rkgrid[0][1], self.rkgrid[1][0], 1, 2)
            input = _translation(input, self.kgrid, q1, q2, q3)
            return input

        elif self.p_idx == 5:
            input = _rotation_derivative(input, self.p_idx, theta3, self.rkgrid[0][2], self.rkgrid[1][2], 0, 1)
            input = _rotation(input, theta2, self.rkgrid[0][1], self.rkgrid[1][1], 2, 0)
            input = _rotation(input, theta1, self.rkgrid[0][0], self.rkgrid[1][0], 1, 2)
            input = _translation(input, self.kgrid, q1, q2, q3)
            return input

        else:
            raise ValueError("Partial index must be an int between [0,5]!")

    def _adjoint_linop(self):
        pass

class RigidTransformCudaOptimzied:
    """
    Rigid-body 3D transform via three shear steps + translation.
    Supports CPU (NumPy+SciPy) and GPU (CuPy+FFT-plan) backends.
    """
    def __init__(self, parameters, kgrid, rkgrid, inverse=False):
        self.params = parameters
        self.kgrid = kgrid
        self.rkgrid = rkgrid
        self.inverse = inverse

    def _xp(self, x):
        return cp.get_array_module(x)

    def _compute_translation_factor(self, kgrid, q1,q2,q3, inverse=False):
        k1, k2, k3 = kgrid.values()
        xp = sp.get_array_module(k1)
        factor = xp.exp(-1j * (q1 * k1 + q2 * k2 + q3 * k3))
        return factor if not inverse else factor.conj()

    def apply(self, x, out=None):
        xp = self._xp(x)
        arr = out if out is not None else x.copy()
        q1, q2, q3, t1, t2, t3 = self.params
        steps = [
            (t1, self.rkgrid['tan']['x'], self.rkgrid['sin']['x'], 1, 2),
            (t2, self.rkgrid['tan']['y'], self.rkgrid['sin']['y'], 2, 0),
            (t3, self.rkgrid['tan']['z'], self.rkgrid['sin']['z'], 0, 1)
        ]

        # Shear sequence with orthonormal norm
        for theta, rk_tan, rk_sin, ax_t, ax_s in steps:
            vtan = xp.exp(1j * xp.tan(theta/2) * rk_tan, dtype=xp.complex64)
            vsin = xp.exp(-1j * xp.sin(theta) * rk_sin, dtype=xp.complex64)

            if xp is cp:
                arr = gpu_fft(arr, axis=ax_t, overwrite_x=True, norm='ortho')
                xp.multiply(arr, vtan, out=arr)
                arr = gpu_ifft(arr, axis=ax_t, overwrite_x=True, norm='ortho')
            else:
                arr = xp.fft.fftn(arr, axes=(ax_t,), norm='ortho')
                arr *= vtan
                arr = xp.fft.ifftn(arr, axes=(ax_t,), norm='ortho')
            
            # sin-axis shear
            if xp is cp:
                arr = gpu_fft(arr, axis=ax_s, overwrite_x=True, norm='ortho')
                xp.multiply(arr, vsin, out=arr)
                arr = gpu_ifft(arr, axis=ax_s, overwrite_x=True, norm='ortho')
            else:
                arr = xp.fft.fftn(arr, axes=(ax_s,), norm='ortho')
                arr *= vsin
                arr = xp.fft.ifftn(arr, axes=(ax_s,), norm='ortho')
            
            # repeat tan-axis
            if xp is cp:
                arr =gpu_fft(arr, axis=ax_t, overwrite_x=True, norm='ortho')
                xp.multiply(arr, vtan, out=arr)
                arr = gpu_ifft(arr, axis=ax_t, overwrite_x=True, norm='ortho')
            else:
                arr = xp.fft.fftn(arr, axes=(ax_t,), norm='ortho')
                arr *= vtan
                arr = xp.fft.ifftn(arr, axes=(ax_t,), norm='ortho')

        # Translation with orthonormal norm
        U = self._compute_translation_factor(self.kgrid, q1, q2, q3, self.inverse)
        if xp is cp:
            arr = gpu_fftn(arr, axes=tuple(range(-3,0)), overwrite_x=True, norm='ortho')
            xp.multiply(arr, U, out=arr)
            arr = gpu_ifftn(arr, axes=tuple(range(-3,0)), overwrite_x=True, norm='ortho')
        else:
            arr = xp.fft.fftn(arr, axes=tuple(range(arr.ndim)), norm='ortho')
            arr *= U
            arr = xp.fft.ifftn(arr, axes=tuple(range(arr.ndim)), norm='ortho')

        return arr

    def adjoint(self, x, out=None):
        xp = self._xp(x)
        arr = out if out is not None else x.copy()

        q1, q2, q3, t1, t2, t3 = self.params
        steps = [
            (t1, self.rkgrid['tan']['x'], self.rkgrid['sin']['x'], 1, 2),
            (t2, self.rkgrid['tan']['y'], self.rkgrid['sin']['y'], 2, 0),
            (t3, self.rkgrid['tan']['z'], self.rkgrid['sin']['z'], 0, 1)
        ]

        # Translation with orthonormal norm
        U = self._compute_translation_factor(self.kgrid, q1, q2, q3, True)
        if xp is cp:
            arr = gpu_fftn(arr, axes=tuple(range(-3,0)), overwrite_x=True, norm='ortho')
            xp.multiply(arr, U, out=arr)
            arr = gpu_ifftn(arr, axes=tuple(range(-3,0)), overwrite_x=True, norm='ortho')
        else:
            arr = xp.fft.fftn(arr, axes=tuple(range(arr.ndim)), norm='ortho')
            arr *= U
            arr = xp.fft.ifftn(arr, axes=tuple(range(arr.ndim)), norm='ortho')

        # Shear sequence with orthonormal norm
        for theta, rk_tan, rk_sin, ax_t, ax_s in steps[::-1]:
            vtan = _compute_shear_factor(rk_tan, theta, 'tan', True)
            vsin = _compute_shear_factor(rk_sin, theta, 'sin', True)

            # tan-axis shear
            if xp is cp:
                arr = gpu_fft(arr, axis=ax_t, overwrite_x=False, norm='ortho')
                xp.multiply(arr, vtan, out=arr)
                arr = gpu_ifft(arr, axis=ax_t, overwrite_x=False, norm='ortho')
            else:
                arr = xp.fft.fftn(arr, axes=(ax_t,), norm='ortho')
                arr *= vtan
                arr = xp.fft.ifftn(arr, axes=(ax_t,), norm='ortho')
            
            # sin-axis shear
            if xp is cp:
                arr = gpu_fft(arr, axis=ax_s, overwrite_x=True, norm='ortho')
                xp.multiply(arr, vsin, out=arr)
                arr = gpu_ifft(arr, axis=ax_s, overwrite_x=True, norm='ortho')
            else:
                arr = xp.fft.fftn(arr, axes=(ax_s,), norm='ortho')
                arr *= vsin
                arr = xp.fft.ifftn(arr, axes=(ax_s,), norm='ortho')
            
            # repeat tan-axis
            if xp is cp:
                arr =gpu_fft(arr, axis=ax_t, overwrite_x=True, norm='ortho')
                xp.multiply(arr, vtan, out=arr)
                arr = gpu_ifft(arr, axis=ax_t, overwrite_x=True, norm='ortho')
            else:
                arr = xp.fft.fftn(arr, axes=(ax_t,), norm='ortho')
                arr *= vtan
                arr = xp.fft.ifftn(arr, axes=(ax_t,), norm='ortho')

        return arr
    
class RigidTransformDerivativeCuda:
    def __init__(self, shape, parameters, kgrid, rkgrid):
        self.parameters = parameters
        self.kgrid = kgrid
        self.rkgrid = rkgrid

        #Preallocated result buffer and two intermediates for derivative calculations
        #This way we can retain the parameters and apply the transforms continually for
        #which ever partial derivative index
        self.out = cp.empty(shape, dtype=cp.complex64)
        self.s2 = cp.empty_like(self.out)
        self.s3 = cp.empty_like(self.out)

    def _rotation(self, theta, rk_tan, rk_sin, ax_t, ax_s):
        xp = cp
        vtan = xp.exp(1j * xp.tan(theta/2) * rk_tan, dtype=xp.complex64)
        vsin = xp.exp(-1j * xp.sin(theta) * rk_sin, dtype=xp.complex64)

        self.out = gpu_fft(self.out, axis=ax_t, overwrite_x=True, norm='ortho')
        xp.multiply(self.out, vtan, out=self.out)
        self.out = gpu_ifft(self.out, axis=ax_t, overwrite_x=True, norm='ortho')
        
        # sin-axis shear
        self.out = gpu_fft(self.out, axis=ax_s, overwrite_x=True, norm='ortho')
        xp.multiply(self.out, vsin, out=self.out)
        self.out = gpu_ifft(self.out, axis=ax_s, overwrite_x=True, norm='ortho')

        # repeat tan-axis
        self.out =gpu_fft(self.out, axis=ax_t, overwrite_x=True, norm='ortho')
        xp.multiply(self.out, vtan, out=self.out)
        self.out = gpu_ifft(self.out, axis=ax_t, overwrite_x=True, norm='ortho')

    def _translation(self):
        xp = cp
        # Translation with orthonormal norm
        #U = self._compute_translation_factor(self.kgrid, q1, q2, q3, self.inverse)
        k1, k2, k3 = self.kgrid.values()
        q1, q2, q3 = self.parameters[:3]
        factor = xp.exp(-1j * (q1 * k1 + q2 * k2 + q3 * k3))
        self.out = gpu_fftn(self.out, axes=tuple(range(-3,0)), overwrite_x=True, norm='ortho')
        xp.multiply(self.out, factor, out=self.out)
        self.out = gpu_ifftn(self.out, axes=tuple(range(-3,0)), overwrite_x=True, norm='ortho')

    def _translation_derivative(self, p_idx):
        xp = cp
        k1, k2, k3 = self.kgrid.values()
        q1, q2, q3 = self.parameters[:3]
        key = {0:'x', 1:'y', 2:'z'}
        pk = self.kgrid[key[p_idx]]
        factor = -1j * pk * xp.exp(-1j * (q1 * k1 + q2 * k2 + q3 * k3))
        self.out = gpu_fftn(self.out, axes=tuple(range(-3,0)), overwrite_x=True, norm='ortho')
        xp.multiply(self.out, factor, out=self.out)
        self.out = gpu_ifftn(self.out, axes=tuple(range(-3,0)), overwrite_x=True, norm='ortho')

    def _rotation_derivative(self, theta, rk_tan, rk_sin, tan_axis, sin_axis):
        xp = cp
        vtan = xp.exp(1j * xp.tan(theta/2) * rk_tan, dtype=xp.complex64)
        vsin = xp.exp(-1j * xp.sin(theta) * rk_sin, dtype=xp.complex64)
        vtan_derivative = (1j * ((1 + (xp.tan(theta/2) ** 2)) / 2) * rk_tan * vtan).astype(xp.complex64)
        vsin_derivative = (-1j * xp.cos(theta) * rk_sin * vsin).astype(xp.complex64)

        self.out = gpu_fft(self.out, axis=tan_axis, overwrite_x=True, norm='ortho')
        self.s2[:] = self.out
        self.s3[:] = self.out
        
        cp.multiply(self.out, vtan_derivative, out=self.out)
        self.out[:] = gpu_ifft(self.out, axis=tan_axis, overwrite_x=True, norm='ortho')
        self.out[:] = gpu_fft(self.out, axis=sin_axis, overwrite_x=True, norm='ortho')
        cp.multiply(self.out, vsin, out=self.out)
        self.out[:] = gpu_ifft(self.out, axis=sin_axis, overwrite_x=True, norm='ortho')
        self.out[:] = gpu_fft(self.out, axis=tan_axis, overwrite_x=True, norm='ortho')
        cp.multiply(self.out, vtan, out=self.out)

        cp.multiply(self.s2, vtan, out=self.s2)
        self.s2[:] = gpu_ifft(self.s2, axis=tan_axis, overwrite_x=True, norm='ortho')
        self.s2[:] = gpu_fft(self.s2, axis=sin_axis, overwrite_x=True, norm='ortho')
        cp.multiply(self.s2, vsin_derivative, out=self.s2)
        self.s2[:] = gpu_ifft(self.s2, axis=sin_axis, overwrite_x=True, norm='ortho')
        self.s2[:] = gpu_fft(self.s2, axis=tan_axis, overwrite_x=True, norm='ortho')
        cp.multiply(self.s2, vtan, out=self.s2)


        cp.multiply(self.s3, vtan, out=self.s3)
        self.s3[:] = gpu_ifft(self.s3, axis=tan_axis, overwrite_x=True, norm='ortho')
        self.s3[:] = gpu_fft(self.s3, axis=sin_axis, overwrite_x=True, norm='ortho')
        cp.multiply(self.s3, vsin, out=self.s3)
        self.s3[:] = gpu_ifft(self.s3, axis=sin_axis, overwrite_x=True, norm='ortho')
        self.s3[:] = gpu_fft(self.s3, axis=tan_axis, overwrite_x=True, norm='ortho')
        cp.multiply(self.s3, vtan_derivative, out=self.s3)

        # out = out + s2 + s3 in-place
        cp.add(self.out, self.s2, out=self.out)
        cp.add(self.out, self.s3, out=self.out)

        # last IFFT
        self.out = gpu_ifft(self.out, axis=tan_axis, overwrite_x=True, norm='ortho')
    
    def apply(self, input, p_idx):
        theta1, theta2, theta3 = self.parameters[3:]
        self.out[:] = input

        if p_idx < 3:                
            self._rotation(theta3, self.rkgrid['tan']['z'], self.rkgrid['sin']['z'], 0, 1)
            self._rotation(theta2, self.rkgrid['tan']['y'], self.rkgrid['sin']['y'], 2, 0)
            self._rotation(theta1, self.rkgrid['tan']['x'], self.rkgrid['sin']['x'], 1, 2)
            self._translation_derivative(p_idx)

        elif p_idx == 3:
            self._rotation(theta3, self.rkgrid['tan']['z'], self.rkgrid['sin']['z'], 0, 1)
            self._rotation(theta2, self.rkgrid['tan']['y'], self.rkgrid['sin']['y'], 2, 0)
            self._rotation_derivative(theta1, self.rkgrid['tan']['x'], self.rkgrid['sin']['x'], 1, 2)
            self._translation()

        elif p_idx == 4:
            self._rotation(theta3, self.rkgrid['tan']['z'], self.rkgrid['sin']['z'], 0, 1)
            self._rotation_derivative(theta2, self.rkgrid['tan']['y'], self.rkgrid['sin']['y'], 2, 0)
            self._rotation(theta1, self.rkgrid['tan']['x'], self.rkgrid['sin']['x'], 1, 2)
            self._translation()

        elif p_idx == 5:
            self._rotation_derivative(theta3, self.rkgrid['tan']['z'], self.rkgrid['sin']['z'], 0, 1)
            self._rotation(theta2, self.rkgrid['tan']['y'], self.rkgrid['sin']['y'], 2, 0)
            self._rotation(theta1, self.rkgrid['tan']['x'], self.rkgrid['sin']['x'], 1, 2)
            self._translation()

        return self.out.copy()