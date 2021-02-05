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
            if None, taken to be the img

        """
        self.img = img
        self.psf = psf

        otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf)))
        center = tuple(s // 2 for s in otf.shape)
        otf /= otf[center]  # definition of OTF, normalize by DC
        self.otf = otf
        self.otfconj = np.conj(otf)

        if fHat is None:
            fHat = img

        self.fHat = fHat
        self.iter = 0

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
        self.iter += 1
        return self.fHat


class MFPMAP:
    """Multi-Frame PMAP algorithm.  Suitable for panchromatic restoration.

    Implements Ref [1], Eq. 2.26.
    """

    def __init__(self, imgs, psfs, fHat=None):
        """Initialize a new PMAP problem.

        Parameters
        ----------
        imgs : `numpy.ndarray`
            images from the camera, sequence of ndarray of shape (n, m).
            The images must be fully co-registered before input to the algorithm.
            A (k, n, m) shaped array iterates correctly, as does a list or other
            iterable of (n, m) arrays
        psfs : `numpy.ndarray`
            psfs corresponding to imgs, sequence of ndarray of shape (n, m)
        fhat : `numpy.ndarray`
            initial object estimate, ndarray of shape (n, m)
            if None, taken to be the first img

        Notes
        -----
        This implementation is optimized for performance on hardware with a large
        amount of memory.  The OTFs can be computed during each step to use less
        memory overall, in exchange for slower iterations.

        """
        self.imgs = imgs
        self.psfs = psfs

        otfs = [np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf))) for psf in psfs]
        center = tuple(s // 2 for s in otfs[0].shape)
        for otf in otfs:
            otf /= otf[center]  # definition of OTF, normalize by DC

        self.otfs = otfs
        self.otfsconj = [np.conj(otf) for otf in otfs]

        if fHat is None:
            fHat = imgs[0]

        self.fHat = fHat
        self.iter = 0

    def step(self):
        """Iterate the algorithm one step.

        Because this implementation cycles through the images, the steps can be
        thought of as mini-batches.  Intuitively, you may wish to make len(imgs)
        steps at a time.

        Returns
        -------
        fhat : `numpy.ndarray`
            updated object estimate, of shape (n, m)

        """
        i = self.iter % len(self.otfs)

        otf = self.otfs[i]
        img = self.imgs[i]
        otfconj = self.otfsconj[i]

        Fhat = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.fHat)))
        denom = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Fhat * otf))).real

        # # kernel is the expression { g/(f_n conv h) - 1 } from 2.16, J. J. Green's thesis
        kernel = (img / denom) - 1
        R = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(kernel)))
        grad = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(R * otfconj))).real
        self.fHat = self.fHat * np.exp(grad)
        self.iter += 1
        return self.fHat
