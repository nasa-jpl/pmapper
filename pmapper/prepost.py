"""Routines for pre-processing data."""
import math

from .backend import np


def cropcenter(img, out_shape):
    """Crop the central (out_shape) of an image, with FFT alignment.

    As an example, if img=512x512 and out_shape=200
    out_shape => 200x200 and the returned array is 200x200, 156 elements from the [0,0]th pixel

    This function is the adjoint of padcenter.

    Parameters
    ----------
    img : `numpy.ndarray`
        ndarray of shape (m, n)
    out_shape : `int` or `iterable` of int
        shape to crop out, either a scalar or pair of values

    """
    if isinstance(out_shape, int):
        out_shape = (out_shape, out_shape)

    padding = [i-o for i, o in zip(img.shape, out_shape)]
    left = [p//2 for p in padding]
    slcs = tuple((slice(l, l+o) for l, o in zip(left, out_shape)))  # NOQA -- l ambiguous
    return img[slcs]


def padcenter(img, out_shape):
    """Pad an image symmetrically with zeros.

    This function is similar to prysm.fttools.pad2d with a different interface.

    The output array may share memory with img.

    Parameters
    ----------
    img : `numpy.ndarray`
        ndarray of shape (m, n) of any dtype
    out_shape : `int` or `iterable`
        shape of the output array, either a or (a, b)
        if a, the output shape is (a, a); b==a

    Returns
    -------
    `numpy.ndarray`
        ndarray of the shape (a, b), made by padding img with zeros

    """
    if isinstance(out_shape, int):
        out_shape = (out_shape, out_shape)

    out = np.zeros(out_shape, dtype=img.dtype)
    pad = [(a-b)/2 for a, b in zip(out_shape, img.shape)]
    pad = [math.ceil(p) for p in pad]
    slcs = tuple((slice(p, p+w) for p, w in zip(pad, img.shape)))
    out[slcs] = img

    return out


def pre_jjg(img, Q, guardband=None):
    """Precondition an image in the style of Joe Green.

    A deviation from Joe's recipe is that this uses a Bartlett window;
    the main lobe of the window is narrower in k-space, but the sidelobes
    are higher; -30dB first lobe (Bartlett) vs -60dB, and slower taper
    for the asymptotic side lobe height

    Parameters
    ----------
    img : `numpy.ndarray`
        ndarray of shape (m,n), any dtype.
        Floating point datatypes will have less quantization in the guardband.
    Q : `float`
        sampling factor; λF#/pp of the data in the image
    guardband : `int`, optional
        guardband (in px) to use; used to override automatic estimate

    Returns
    -------
    `numpy.ndarray`
        ndarray with a Fourier guardband and zero padding to square dimension

    """
    # first phase, apply a Fourier guardband
    # Rationale: we want the guardband to capture, say, PSF >= 0.02 max
    # then we want to capture the first few sidelobes, which is ~= 16px * Q
    if guardband is None:
        guardband = int(16 * Q)

    pad_samples = ((guardband, guardband), (guardband, guardband))
    img2 = np.pad(img, pad_samples, mode='linear_ramp')

    # second phase, use zero padding to make the array square
    out = padcenter(img2, max(img2.shape))
    return out


def post_jjg(out, in_, oversampling):
    """Reciprocal operation to pre_jjg.

    Parameters
    ----------
    out : `numpy.ndarray`
        output image
    in_ : `numpy.ndarray`
        input image (given to pre_jjg)
    oversampling : `int`
        integer oversampling factor; ratio of output to input shape in a larger
        PMAP routine

    Returns
    -------
    `numpy.ndarray`
        cropped output; shares memory with out

    """
    output_shape = [s*oversampling for s in in_.shape]
    return cropcenter(out, output_shape)


def match_psf_and_data_sampling(img, psf, oversampling):
    """Modify psf such that it has matched sampling to img.

    psf will either be cropped or Bartlett padded, depending if
    it has too many or too few samples.

    Parameters
    ----------
    img : `numpy.ndarray`
        ndarray of shape (m, n) containing image data
    psf : `numpy.ndarray`
        ndarray of shape (a, b) containing PSF data
    oversampling : `float`
        oversampling of the PSF relative to the image

    Returns
    -------
    `numpy.ndarray`
        modified PSF such that the output (a, b) are equal to (oversampling * m, oversampling * n)
        The "modified" PSF may share memory with the input psf.

    """
    output_shape = [s * oversampling for s in img.shape]
    need_to_crop = False
    if psf.shape[0] > output_shape[0]:
        need_to_crop = True
    if psf.shape[1] > output_shape[1]:
        need_to_crop = True

    if need_to_crop:
        psf = cropcenter(psf, output_shape)
    else:
        psf = padcenter(psf, output_shape)

    return psf
