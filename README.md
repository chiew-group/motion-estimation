# Motion Correction in MRI

Motion correction for MRI reconstruction using joint optimization techniques from the papers titled "Sensitivity Encoding for Aligned Multishot Magnetic Resonance Reconstruction" and DISORDER sampling techniques from a subsequent paper written by the same authors titled "Motion-corrected MRI with DISORDER: Distributed and incoherent sample orders for reconstruction deblurring using encoding redundancy".

## Environment Installation

Use a basic python venv with the requirements.txt file

## Directory Structure

The directory is just a flat collection of files that have various functions that are used in the two main scripts namely simulation.py which handles any experimental simulations, and recon.py that will take a dat file as input and run the reconstruction end to end.

### Preprocessing
- `extract_from_dat.py`: extracts kspace, ref scan and noise raw data from a .dat file
- `coil_compression.py`: compresses sensitivity maps and kspace data by reducing number of coils
- `espirit.py`: implementation of Espirit algorithm for large in memory datasets, used to estimate sensitivity maps
- `noise_whiten.py`: applies whitening matrix to kspace using noise data aquired from raw dat file data

### Core Algorithm
- `image_estimation.py`: this script contains the GPU accelerated image reconstruction portion of the algorithm, the function named `estimate_image_cg` everything else is legacy implementations. This function assumes a gpu and has not been tested yet for CPU usage.
- `transform_estimation.py`: this script contains the motion estimate step of the algorithm contained in the function call `estimate_transform`, the rest is all legacy code that should not be used.
- `transform.py`: this script contains the linop classes for applying the rigid transforms. The two in question are called `RigidTransformCudaOptimzied` and `RigidTransformDerivativeCuda` the rest is legacy implementations that should not be used.
- `joint_recon.py`: this script contains the `JointRecon` class that is the main algorithm and alternated between image estimation and motion estimate returning the fully reconstructed image with estiamted motion parameters.
- `pyramid.py`: this is an orchestration script which handles downsampling the input to a correct size and running a joint recon and continuing the process till we are back at full resolution.

### Miscellaneous
- `utils.py`: self explanatory file that conatins small utility functions

## Usage

Mentioned in the previous section the two main scripts that you will be using are `recon.py` and `simulation.py` and they can be simply run via commaline with some agruments supplied. Using the `--help` functionality will list what possibly inputs are allowed.

The standard way 3D and 4D image data are represented is the following:
- kspace: [ncoils, kx, ky, kz] complex64
- mps: [ncoils, kx, ky, kz] complex64
- ground/image: [x, y, z] complex64
- sampling mask: [nshots, x, y, z] bool
- transforms: [nshots, 6] float32

Another common naming convention used throughout is when a reconstruction is labeled as **sense** this refers to a simple sense reconstruction wihout motino correction. **Joint Recon** indicates a reconstruction that estimates motion and automatically corrects the image for it.

```bash
python recon.py \
    --dat <path to .dat file> \
    --out <path to output dir\
    --nshots 64 \
    --sampling disorder\
    --iters 100
```

```bash
python simulation.py \
    --ground <path to image file in .npy format>\
    --mps <path to sensitivity maps file in .npy format>\
    --out_dir <path to output dir>\
    --ord disorder\
    --tile_size 16 16\
    --accel 2 \
    --iters 1000\
    --continuous
```

