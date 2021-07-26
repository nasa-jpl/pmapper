"""Richardson-Lucy algorithm, implemented for comparison to PMAP."""

from .backend import fft, np


class RichardsonLucy:
    """Non-blind or sighted Richardson-Lucy deconvolution algorithm.

    Implements the contemporary RL algorithm, uhat(n+1) = uhat * (d/[uhat conv P] conv P*)
    """

    def __init__(self, img, psf, fHat=None):
        """Initialize a new Richardson-Lucy problem.

        Parameters
        ----------
        img : `numpy.ndarray`
            image from the camera, ndarray of shape (n, m)
        psf : `numpy.ndarray`
            psf corresponding to img, ndarray of shape (n, m)
        fhat : `numpy.ndarray`, optional
            initial object estimate, ndarray of shape (n, m)
            if None, taken to be the img

        """
        self.img = img
        self.psf = psf
        self.otf = fft.fftshift(fft.fft2(fft.ifftshift(psf)))
        self.otfconj = np.conj(self.otf)

        if fHat is None:
            fHat = img

        self.iter = 0

    def step(self):
        """Step the algorithm forward one step."""
        num = self.img
        FHat = fft.fftshift(fft.fft2(fft.ifftshift(self.fHat)))
        den = fft.fftshift(fft.ifft2(fft.ifftshift(FHat*self.otf)))
        term1 = num / den
        D = fft.fftshift(fft.fft2(fft.ifftshift(term1)))
        Dprime = D * self.otfT
        update = fft.fftshift(fft.ifft2(fft.ifftshift(Dprime)))
        self.fHat = self.fHat * update
        self.iter += 1
        return self.fHat
