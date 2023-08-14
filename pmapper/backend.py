"""CPU/GPU/Framework agnostic backend."""
import numpy as np

from scipy import ndimage, fft

# the contents of this file is taken from the 'prysm' repository on GitHub which
# is MIT licensed.  backend.py must not be separated from prysm-LICENSE.md.  No
# other behavior is required to comply with prysm's license.
# https://github.com/brandondube/prysm/blob/master/LICENSE.md


class BackendShim:
    """A shim that allows a backend to be swapped at runtime."""
    def __init__(self, src):
        self._srcmodule = src

    def __getattr__(self, key):
        if key == '_srcmodule':
            return self._srcmodule

        return getattr(self._srcmodule, key)

np = BackendShim(np)
ndimage = BackendShim(ndimage)
fft = BackendShim(fft)


# TODO: check if it is waste work to do the fft shifts, since ft_fwd and ft_rev
# are always used together
def ft_fwd(a):
    """Forward Fourier transform.

    Parameters
    ----------
    a : numpy.ndarray
        ndarray of shape (m, n)

    Returns
    -------
    numpy.ndarray
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
    a : numpy.ndarray
        ndarray of shape (m, n)

    Returns
    -------
    numpy.ndarray
        complex-valued FT of shape (m, n)

    Notes
    -----
    this function makes those that use it clearer, replacing
    fft.fftshift(fft.ifft2(fft.ifftshift(a))) with ft_fwd(a)

    """
    return fft.fftshift(fft.ifft2(fft.ifftshift(a)))
