"""CPU/GPU/Framework agnostic backend."""
import numpy as np

from scipy import ndimage, fft

# the contents of this file is taken from the 'prysm' repository on GitHub which
# is MIT licensed.  backend.py must not be separated from prysm-LICENSE.md.  No
# other behavior is required to comply with prysm's license.
# https://github.com/brandondube/prysm/blob/master/LICENSE.md


class NumpyEngine:
    """An engine allowing an interchangeable backend for mathematical functions."""

    def __init__(self, np=np):
        """Create a new math engine.

        Parameters
        ----------
        source : `module`
            a python module.
        """
        self.numpy = np

    def __getattr__(self, key):
        """Get attribute.

        Parameters
        ----------
        key : `str`
            attribute name

        """
        return getattr(self.numpy, key)


class NDImageEngine:
    """An engine which allows scipy.ndimage to be redirected to another lib at runtime."""

    def __init__(self, ndimage=ndimage):
        """Create a new scipy engine.

        Parameters
        ----------
        ndimage : `module`
            a python module, with the same API as scipy.ndimage
        interpolate : `module`
            a python module, with the same API as scipy.interpolate
        special : `module`
            a python module, with the same API as scipy.special

        """
        self.ndimage = ndimage

    def __getattr__(self, key):
        """Get attribute.

        Parameters
        ----------
        key : `str`
            attribute name

        """
        return getattr(self.ndimage, key)


class FFTEngine:
    """An engine which allows scipy.fft to be redirected to another lib at runtime."""

    def __init__(self, fft=fft):
        """Create a new scipy engine.

        Parameters
        ----------
        fft : `module`
            a python module, with the same API as scipy.fft
        interpolate : `module`
            a python module, with the same API as scipy.interpolate
        special : `module`
            a python module, with the same API as scipy.special

        """
        self.fft = fft

    def __getattr__(self, key):
        """Get attribute.

        Parameters
        ----------
        key : `str`
            attribute name

        """
        return getattr(self.fft, key)


np = NumpyEngine()
ndimage = NDImageEngine(ndimage)
fft = FFTEngine(fft)


def ft_fwd(a):
    """Forward Fourier transform.

    Parameters
    ----------
    a : `numpy.ndarray`
        ndarray of shape (m, n)

    Returns
    -------
    `numpy.ndarray`
        complex-valued FT of shape (m, n)

    Notes
    -----
    this function makes those that use it clearer, replacing
    ft_fwd(a))) with ft_fwd(a)

    """
    return fft.fftshift(fft.fft2(fft.ifftshift(a)))


def ft_rev(a):
    """Reverse (inverse) Fourier transform.

    Parameters
    ----------
    a : `numpy.ndarray`
        ndarray of shape (m, n)

    Returns
    -------
    `numpy.ndarray`
        complex-valued FT of shape (m, n)

    Notes
    -----
    this function makes those that use it clearer, replacing
    fft.fftshift(fft.ifft2(fft.ifftshift(a))) with ft_fwd(a)

    """
    return fft.fftshift(fft.ifft2(fft.ifftshift(a)))
