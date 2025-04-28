import sigpy as sp

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