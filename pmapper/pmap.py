"""Several specialized implementations of the PMAP super-resolution algorithm."""

from .backend import np, ft_fwd, ft_rev, ndimage
from .bayer import decomposite_bayer

# If you wish to understand how PMAP works from this code, it is recommended
# that you read :class:PMAP first.  MFPMAP is just PMAP that cycles through the
# frames, and Bayer implementations are the same, but with inner iterations per
# step for the color planes.

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
# forces a particular resizing algorithm, but it is as good as can be done to
# the author's knowledge anyway.


def pmap_core(fHat, g, H, Hstar, bufup=None, bufdown=None, prefilter=False, zoomfactor=1, invzoomfactor=1):
    """Core routine of PMAP, produce a new object estimate.

    Parameters
    ----------
    fHat : `numpy.ndarray`
        Nth object estimate, ndarray of shape (m, n)
    g : `numpy.ndarray`
        source image, ndarray of shape (a, b)
    H : `numpy.ndarray`
        OTF, complex ndarray of shape (m, n)
    Hstar : `numpy.ndarray`
        complex conjugate of H, ndarray of shape (m, n)
    bufup : `numpy.ndarray`, optional
        real-valued buffer for upsampled data, of shape (m, n)
    bufdown : `numpy.ndarray`, optional
        real-valued buffer for downsampled data, of shape (a, b)
    prefilter : `bool`, optional
        if True, use spline prefiltering
        False is generally better at getting realistic image chain aliasing correct
    zoomfactor : `float`, optional
        ratio m/a
    invzoomfactor : `float`, optional
        ratio a/m

    Returns
    -------
    fHat : `numpy.ndarray`
        N+1th object estimate, ndarray of shape (m, n)

    """
    if zoomfactor == 1 and fHat.shape[0] != H.shape[0]:
        raise ValueError(f'PMAP: zoom factor was given as 1, but fHat and OTF have unequal shapes {fHat.shape} and {H.shape}')  # NOQA

    Fhat = ft_fwd(fHat)
    denom = ft_rev(Fhat * H).real
    # denom may be supre-resolved, i.e. have more pixels than g
    # zoomfactor is the ratio of their sampling, the below does
    # inline up and down scaling as denoted in Ref [1] Eq. 2.26
    # re-assign denom and kernel, non-allocating invocation of zoom
    if zoomfactor != 1:
        denom = ndimage.zoom(denom, invzoomfactor, prefilter=prefilter, output=bufdown)
        kernel = (g / denom) - 1
        kernel = ndimage.zoom(kernel, zoomfactor, prefilter=prefilter, output=bufup)
    else:
        # kernel is the expression { g/(f_n conv h) - 1 } from 2.16, J. J. Green's thesis
        kernel = (g / denom) - 1

    R = ft_fwd(kernel)
    grad = ft_rev(R * Hstar).real
    fHat = fHat * np.exp(grad)
    return fHat


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

        otf = ft_fwd(psf)
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
        self.fHat = pmap_core(self.fHat, self.img, self.otf, self.otfconj,
                              self.bufup, self.bufdown, self.prefilter,
                              self.zoomfactor, self.invzoomfactor)
        self.iter += 1
        return self.fHat


class BayerPMAP:
    """Classical Bayer-aware PMAP algorithm.  Suitable for trichromatic restoration.

    Implements Ref [1], Eq. 2.16 with modifications for bayer.
    """

    def __init__(self, img, psfs, fHats=None, prefilter=False, cfa='rggb'):
        """Initialize a new PMAP problem.

        Parameters
        ----------
        img : `numpy.ndarray`
            image from the camera, ndarray of shape (n, m)
        psfs : iterable of `numpy.ndarray`
            sequence of PSFs corresponding to img, ndarrays of shape (a, b) in
            the same order as the cfa string
        fhats : iterable of `numpy.ndarray`, optional
            sequence of  initial object estimates for each color plane,
            ndarrays of shape (a, b).  if None, taken to be the img, rescaled
            if necessary to match PSF sampling
        prefilter : `bool`, optional
            if True, uses input stage filters when performing spline-based
            resampling, else no input filter is used.  No pre-filtering is
            generally a best fit for image chain modeling and allows aliasing
            into the problem that would be present in a hardware system.

        """
        self.imgs = decomposite_bayer(img, cfa)
        self.psfs = psfs
        self.cfa = cfa

        otfs = [ft_fwd(psf) for psf in psfs]
        center = tuple(s // 2 for s in otfs[0].shape)
        for otf in otfs:
            otf /= otf[center]  # definition of OTF, normalize by DC

        self.otfs = otfs
        self.otfsconj = [np.conj(otf) for otf in otfs]

        self.zoomfactor = self.psfs[0].shape[0] / self.imgs[0].shape[0]
        self.invzoomfactor = 1 / self.zoomfactor
        self.prefilter = prefilter

        if fHats is None:
            fHats = decomposite_bayer(img, cfa)
            fHats = [ndimage.zoom(f, self.zoomfactor, prefilter=prefilter) for f in fHats]

        self.fHats = fHats
        self.bufup = np.empty(self.psfs[0].shape, dtype=self.psfs[0].dtype)
        self.bufdown = np.empty(self.imgs[0].shape, dtype=self.imgs[0].dtype)
        self.iter = 0

    def step(self):
        """Iterate the algorithm one step.

        Returns
        -------
        fhat : `numpy.ndarray`
            updated object estimate, of shape (a, b)

        """
        lcl_fHats = []
        for img, otf, otfconj, fHat in zip(self.imgs, self.otfs, self.otfsconj, self.fHats):
            fHat = pmap_core(fHat, img, otf, otfconj,
                             self.bufup, self.bufdown, self.prefilter,
                             self.zoomfactor, self.invzoomfactor)
            lcl_fHats.append(fHat)

        self.fHats = lcl_fHats
        self.iter += 1
        return self.fHats


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
        psfs : `iterable` of `numpy.ndarray`
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

        otfs = [ft_fwd(psf) for psf in psfs]
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
        self.fHat = pmap_core(self.fhat, img, otf, otfconj,
                              self.bufup, self.bufdown, self.prefilter,
                              self.zoomfactor, self.invzoomfactor)
        self.iter += 1
        return self.fHat


class BayerMFPMAP:
    """Multi-Frame PMAP algorithm.  Suitable for Bayer image restoration.

    Implements Ref [1], Eq. 2.26, modified for use with Bayer imagery.

    Does not demosaic the data, performs estimation on each interleaved plane
    independently.
    """

    def __init__(self, imgs, psfs, fHats=None, prefilter=False, cfa='rggb'):
        """Initialize a new PMAP problem.

        Parameters
        ----------
        imgs : `numpy.ndarray`
            images from the camera, sequence of ndarray of shape (n, m).
            The images must be fully co-registered before input to the algorithm.
            A (k, n, m) shaped array iterates correctly, as does a list or other
            iterable of (n, m) arrays
        psfs : iterable of iterable of `numpy.ndarray`
            outer iteration is the color channel, inner iteration is the image.
            An ndarray of shape (s, k, a, b) iterates correctly.  The dimension
            s notionally must always be of size 4 (the number of channels in a
            bayer mosaic).
        fhats : `iterable` of `numpy.ndarray`
            r, g1, g2, b initial object estimates, ndarrays of shape (a, b)
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

        The reconstruction is done on the color planes independently, without
        debayering.  The caller must debayer the output estimate using e.g.
        Malvar debayering.  This package includes an implementation of Malvar.

        """
        self.imgs = np.array([decomposite_bayer(i, cfa) for i in imgs])
        self.psfs = psfs
        self.otfs = []
        self.otfsconj = []
        psfs = np.array(psfs)
        self.psfs = psfs
        for s in range(psfs.shape[0]):
            # lcl = local
            lcl_otfs = [ft_fwd(psf) for psf in psfs[s]]
            # definition of OTF, normalize by DC
            center = tuple(e // 2 for e in lcl_otfs[0].shape)
            for otf in lcl_otfs:
                otf /= otf[center]

            lcl_otfs_conj = [np.conj(otf) for otf in lcl_otfs]
            self.otfs.append(lcl_otfs)
            self.otfsconj.append(lcl_otfs_conj)

        self.zoomfactor = self.psfs.shape[-1] / self.imgs.shape[-1]
        self.invzoomfactor = 1 / self.zoomfactor
        self.prefilter = prefilter

        if fHats is None:
            imgs = self.imgs[0]
            fHats = [ndimage.zoom(i, self.zoomfactor, prefilter=prefilter) for i in imgs]

        # swapaxis moves the bayer channels to the first dimension.  As contiguous makes sure
        # that the memory is copied instead of a view created -- we will revisit these arrays
        # many times and paying the copy here will pay dividends later
        self.imgs = np.ascontiguousarray(self.imgs.swapaxes(0, 1))
        self.fHats = fHats

        self.bufup = np.empty(self.psfs.shape[-2:], dtype=self.psfs.dtype)
        self.bufdown = np.empty(self.imgs.shape[-2:], dtype=self.imgs.dtype)
        self.iter = 0

    def step(self):
        """Iterate the algorithm one step.

        Because this implementation cycles through the images, the steps can be
        thought of as mini-batches.  Intuitively, you may wish to make len(imgs)
        steps at a time.

        Parameters
        ----------
        ret : `str`, optional, {'planes', 'mosaic'}
            controls what is returned.  If planes, returns r, g1, g2, b each
            of shape (a, b), else returns mosaiced planes of shape (2*a, 2*b)

        Returns
        -------
        fhats : iterable of `numpy.ndarray`
            updated object estimates, of shape 4x (a, b)

        """
        i = self.iter % self.imgs.shape[1]

        lcl_fHats = []
        for imgs, otfs, otfsconj, fHat in zip(self.imgs, self.otfs, self.otfsconj, self.fHats):
            otf = otfs[i]
            img = imgs[i]
            otfconj = otfsconj[i]
            fHat = pmap_core(fHat, img, otf, otfconj,
                             self.bufup, self.bufdown, self.prefiler,
                             self.zoomfactor, self.invzoomfactor)
            lcl_fHats.append(fHat)

        self.fHats = lcl_fHats
        self.iter += 1
        return self.fHats
