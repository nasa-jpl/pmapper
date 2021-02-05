"""PMAP family of algorithms."""

import numpy as np

# Refs:
# [1] "Multiframe restoration methods for image synthesis and recovery",
# Joseph J. Green, PhD thesis, University of Arizona, 2000
#
# [2] Super-Resolution In a Synthetic Aperture Imaging System
# Joseph J. Green and B. R. Hunt,
# Proceedings of International Conference on Image Processing,
# 26-29 Oct. 1997, DOI 10.1109/ICIP.1997.648103


# TODO:
# - Bayer
# - superresolution via interpolation and decimation
# - copy/paste engine from prysm for CPU/GPU flexibility
# - FFT vs MDFT (MDFT seems not meaningfully faster to justify complexity)


class PMAP:
    """Classical PMAP algorithm.  Suitable for panchromatic restoration.

    Implements Ref [1], Eq. 2.16.
    """

    def __init__(self, img, psf, fHat=None):
        """Initialize a new PMAP problem.

        Parameters
        ----------
        img : `numpy.ndarray`
            image from the camera, ndarray of shape (n, m)
        psf : `numpy.ndarray`
            psf corresponding to img, ndarray of shape (n, m)
        fhat : `numpy.ndarray`
            initial object estimate, ndarray of shape (n, m)

        """
        self.img = img
        self.psf = psf
        self.fHat = fHat

        otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf)))
        center = tuple(s // 2 for s in otf.shape)
        otf /= otf[center]  # definition of OTF, normalize by DC
        self.otf = otf
        self.otfconj = np.conj(otf)

        if fHat is None:
            fHat = img
        self.fHat = fHat

    def step(self):
        """Iterate the algorithm one step.

        Returns
        -------
        fhat : `numpy.ndarray`
            updated object estimate, of shape (n, m)

        """
        Fhat = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.fHat)))
        denom = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Fhat * self.otf))).real

        # # kernel is the expression { g/(f_n conv h) - 1 } from 2.16, J. J. Green's thesis
        kernel = (self.img / denom) - 1
        R = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(kernel)))
        grad = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(R * self.otfconj))).real
        self.fHat = self.fHat * np.exp(grad)
        return self.fHat
