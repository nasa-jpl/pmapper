"""PMAP family of algorithms."""

from .backend import np, ndimage

# Refs:
# [1] "Multiframe restoration methods for image synthesis and recovery",
# Joseph J. Green, PhD thesis, University of Arizona, 2000
#
# [2] Super-Resolution In a Synthetic Aperture Imaging System
# Joseph J. Green and B. R. Hunt,
# Proceedings of International Conference on Image Processing,
# 26-29 Oct. 1997, DOI 10.1109/ICIP.1997.648103

# implementation notes:
# A version which used matrix DFTs was tried, but proved (slightly) slower for
# 512x512 -- 5.2 ms/iter vs 4.6 ms/iter.  FFT has better asymptotic time
# complexity than MDFT, and image resolution is only increasing.  Therefore, the
# FFT version is preferred.
#
# A slightly different implementation would replace the prefilter init argument
# with an actual function to be used to perform up/downsampling.  This would be
# a bit more flexible, but require a bit more work on the user and likely need
# functools.partial to get everything down to one interface.  What is here
# forces a particular resizing algorithm, but it is a good enough choice to work
# well in this application.

# TODO:
# - Bayer


class PMAP:
    """Classical PMAP algorithm.  Suitable for panchromatic restoration.

    Implements Ref [1], Eq. 2.16.
    """

    def __init__(self, img, psf, fHat=None, prefilter=False):
        """Initialize a new PMAP problem.

        Parameters
        ----------
        img : `numpy.ndarray`
            image from the camera, ndarray of shape (n, m)
        psf : `numpy.ndarray`
            psf corresponding to img, ndarray of shape (a, b)
        fhat : `numpy.ndarray`, optional
            initial object estimate, ndarray of shape (a, b)
            if None, taken to be the img, rescaled if necessary to match PSF
            sampling
        prefilter : `bool`, optional
            if True, uses input stage filters when performing spline-based
            resampling, else no input filter is used.  No pre-filtering is
            generally a best fit for image chain modeling and allows aliasing
            into the problem that would be present in a hardware system.

        """
        self.img = img
        self.psf = psf

        otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf)))
        center = tuple(s // 2 for s in otf.shape)
        otf /= otf[center]  # definition of OTF, normalize by DC
        self.otf = otf
        self.otfconj = np.conj(otf)

        self.zoomfactor = self.psf.shape[0] / self.img.shape[0]
        self.invzoomfactor = 1 / self.zoomfactor
        self.prefilter = prefilter

        if fHat is None:
            fHat = ndimage.zoom(img, self.zoomfactor, prefilter=prefilter)
        self.fHat = fHat
        self.bufup = np.empty(self.psf.shape, dtype=self.psf.dtype)
        self.bufdown = np.empty(self.img.shape, dtype=self.img.dtype)
        self.iter = 0

    def step(self):
        """Iterate the algorithm one step.

        Returns
        -------
        fhat : `numpy.ndarray`
            updated object estimate, of shape (a, b)

        """
        Fhat = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.fHat)))
        denom = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Fhat * self.otf))).real
        # denom may be supre-resolved, i.e. have more pixels than g
        # zoomfactor is the ratio of their sampling, the below does
        # inline up and down scaling as denoted in Ref [1] Eq. 2.26
        # re-assign denom and kernel, non-allocating invocation of zoom
        if self.zoomfactor != 1:
            ndimage.zoom(denom, self.invzoomfactor, prefilter=self.prefilter, output=self.bufdown)
            denom = self.bufdown
            kernel = (self.img / denom) - 1
            ndimage.zoom(kernel, self.zoomfactor, prefilter=self.prefilter, output=self.bufup)
            kernel = self.bufup
        else:
            # kernel is the expression { g/(f_n conv h) - 1 } from 2.16, J. J. Green's thesis
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

    def __init__(self, imgs, psfs, fHat=None, prefilter=False):
        """Initialize a new PMAP problem.

        Parameters
        ----------
        imgs : `numpy.ndarray`
            images from the camera, sequence of ndarray of shape (n, m).
            The images must be fully co-registered before input to the algorithm.
            A (k, n, m) shaped array iterates correctly, as does a list or other
            iterable of (n, m) arrays
        psfs : `numpy.ndarray`
            psfs corresponding to imgs, sequence of ndarray of shape (a, b)
        fhat : `numpy.ndarray`
            initial object estimate, ndarray of shape (a, b)
            if None, taken to be the first img rescaled if necessary to match
            PSF sampling
        prefilter : `bool`, optional
            if True, uses input stage filters when performing spline-based
            resampling, else no input filter is used.  No pre-filtering is
            generally a best fit for image chain modeling and allows aliasing
            into the problem that would be present in a hardware system.

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

        self.zoomfactor = self.psfs[0].shape[0] / self.imgs[0].shape[0]
        self.invzoomfactor = 1 / self.zoomfactor
        self.prefilter = prefilter

        if fHat is None:
            fHat = ndimage.zoom(imgs[0], self.zoomfactor, prefilter=prefilter)

        self.fHat = fHat
        self.bufup = np.empty(self.psfs[0].shape, dtype=self.psfs[0].dtype)
        self.bufdown = np.empty(self.imgs[0].shape, dtype=self.imgs[0].dtype)
        self.iter = 0

    def step(self):
        """Iterate the algorithm one step.

        Because this implementation cycles through the images, the steps can be
        thought of as mini-batches.  Intuitively, you may wish to make len(imgs)
        steps at a time.

        Returns
        -------
        fhat : `numpy.ndarray`
            updated object estimate, of shape (a, b)

        """
        i = self.iter % len(self.otfs)

        otf = self.otfs[i]
        img = self.imgs[i]
        otfconj = self.otfsconj[i]

        Fhat = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.fHat)))
        denom = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Fhat * otf))).real
        if self.zoomfactor != 1:
            ndimage.zoom(denom, self.invzoomfactor, prefilter=self.prefilter, output=self.bufdown)
            denom = self.bufdown
            kernel = (img / denom) - 1
            ndimage.zoom(kernel, self.zoomfactor, prefilter=self.prefilter, output=self.bufup)
            kernel = self.bufup
        else:
            # kernel is the expression { g/(f_n conv h) - 1 } from 2.16, J. J. Green's thesis
            kernel = (img / denom) - 1

        R = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(kernel)))
        grad = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(R * otfconj))).real
        self.fHat = self.fHat * np.exp(grad)
        self.iter += 1
        return self.fHat
