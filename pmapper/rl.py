"""Richardson-Lucy algorithm, implemented for comparison to PMAP."""

from .backend import np, ft_fwd, ft_rev


class RichardsonLucy:
    """Non-blind or sighted Richardson-Lucy deconvolution algorithm.

    Implements the contemporary RL algorithm, uhat(n+1) = uhat * (d/[uhat conv P] conv P*)
    """

    def __init__(self, img, psf, fHat=None):
        """Initialize a new Richardson-Lucy problem.

        Parameters
        ----------
        img : numpy.ndarray
            image from the camera, ndarray of shape (n, m)
        psf : numpy.ndarray
            psf corresponding to img, ndarray of shape (n, m)
        fhat : numpy.ndarray, optional
            initial object estimate, ndarray of shape (n, m)
            if None, taken to be the img

        """
        self.img = img
        self.psf = psf
        otf = ft_fwd(psf)
        center = tuple(s // 2 for s in otf.shape)
        otf /= otf[center]  # definition of OTF, normalize by DC
        self.otf = otf
        self.otfconj = np.conj(otf)

        if fHat is None:
            fHat = img

        self.iter = 0

    def step(self):
        """Step the algorithm forward one step."""
        num = self.img
        FHat = ft_fwd(self.fHat)
        den = ft_rev(FHat*self.otf)
        term1 = num / den
        D = ft_fwd(term1)
        Dprime = D * self.otfconj
        update = ft_rev(Dprime)
        self.fHat = self.fHat * update
        self.iter += 1
        return self.fHat
