"""CPU/GPU/Framework agnostic backend."""
import numpy as np

from scipy import ndimage

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


np = NumpyEngine()
ndimage = NDImageEngine(ndimage)
