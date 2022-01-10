# pmapper
pmapper is a super-resolution and deconvolution toolkit for python 3.6+.  PMAP stands for Poisson Maximum A-Posteriori, a highly flexible and adaptable algorithm for these problems.  An implementation of the contemporary Richardson-Lucy algorithm is included for comparison.

The name of this repository is an homage to MTF-Mapper, a slanted edge MTF measurement program written by Frans van den Bergh.

The implementations of all algorithms in this repository are CPU/GPU agnostic and performant, able to perform 4K restoration at hundreds of iterations per second.

# Usage

## Basic PMAP, Multi-frame PMAP
```python
import pmapper

img = ... # load an image somehow
psf = ... # acquire the PSF associated with the img
pmp = pmapper.PMAP(img, psf)  # "PMAP problem"
while pmp.iter < 100:  # number of iterations
    fHat = pmp.step()  # fHat is the object estimate
```

In simulation studies, the true object can be compared to fHat (for example, mean square error) to track convergence.  If the psf is "larger" than the image, for example a 1024x1024 image and a 2048x2048 psf, the output will be super-resolved at the 2048x2048 resolution.

PMAP is able to combine multiple images of the same objec with different PSFs into one with the multi-frame variant.  This can be used to combat dynamical atmospheric seeing conditions, line of sight jitter, or even perform incoherent aperture synthesis; rendering images from sparse aperture systems that mimic or exceed a system with a fully filled aperture.

```python
import pmapper

# load a sequence of images; could be any iterable,
# or e.g. a kxmxn ndarray, with k = num frames
# psfs must have the same "size" (k) and correspond
# to the images in the same indices
imgs = ...
psfs = ...
pmp = pmapper.MFPMAP(imgs, psfs)  # "PMAP problem"
while pmp.iter < len(imgs)*100:  # number of iterations
    fHat = pmp.step()  # fHat is the object estimate
```

Multi-frame PMAP cycles through the images and PSFs, so the total number of iterations "should" be an integer multiple of the number of source images.  In this way, each image is "visited" an equal number of times.


# GPU computing

As mentioned previously, pmapper can be used trivially on a GPU.  To do so, simply execute the following modification:

```python
import pmapper
from pmapper import backend

import cupy as cp
from cupyx.scipy import (
    ndimage as cpndimage,
    fft as cpfft
)

backend.np._srcmodule = cp
backend.fft._srcmodule = cpfft
backend.ndimage._srcmodule = cpndimage

# if your data is not on the GPU already
img = cp.array(img)
psf = cp.array(psf)

# ... do PMAP, it will run on a GPU without changing anything about your code

fHatCPU = fHat.get()
```

cupy is not the only way to do so; anything that quacks like numpy, scipy fft, and scipy ndimage can be used to substitute the backend.  This can be done dynamically and at runtime.  You likely will want to cast your imagery from fp64 to fp32 for performance scaling on the GPU.
