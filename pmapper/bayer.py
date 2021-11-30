"""Basic operations for bayer data."""
from .backend import np, ndimage, fft

top_left = (slice(0, None, 2), slice(0, None, 2))
top_right = (slice(0, None, 2), slice(1, None, 2))
bottom_left = (slice(1, None, 2), slice(0, None, 2))
bottom_right = (slice(1, None, 2), slice(1, None, 2))

ErrBadCFA = NotImplementedError('only rggb, bggr bayer patterns currently implemented')


def wb_prescale(mosaic, wr, wg1, wg2, wb, cfa='rggb'):
    """Apply white-balance prescaling in-place to mosaic.

    Parameters
    ----------
    mosaic : `numpy.ndarray`
        ndarray of shape (m, n), a float dtype
    wr : `float`
        red white balance prescalar
    wg1 : `float`
        G1 white balance prescalar
    wg2 : `float`
        G2 white balance prescalar
    wb : `float`
        blue white balance prescalar
    cfa : `str`, optional, {'rggb', 'bggr'}
        color filter arrangement

    """
    cfa = cfa.lower()
    if cfa == 'rggb':
        mosaic[top_left] *= wr
        mosaic[top_right] *= wg1
        mosaic[bottom_left] *= wg2
        mosaic[bottom_right] *= wb
    elif cfa == 'bggr':
        mosaic[top_left] *= wb
        mosaic[top_right] *= wg1
        mosaic[bottom_left] *= wg2
        mosaic[bottom_right] *= wr
    else:
        raise ErrBadCFA


def composite_bayer(r, g1, g2, b, cfa='rggb', output=None):
    """Composite an interleaved image from densely sampled bayer color planes.

    Parameters
    ----------
    r : `numpy.ndarray`
        ndarray of shape (m, n)
    g1 : `numpy.ndarray`
        ndarray of shape (m, n)
    g2 : `numpy.ndarray`
        ndarray of shape (m, n)
    b : `numpy.ndarray`
        ndarray of shape (m, n)
    cfa : `str`, optional, {'rggb', 'bggr'}
        color filter arangement
    output : `numpy.ndarray`, optional
        output array, of shape (m, n) and same dtype as r, g1, g2, b

    Returns
    -------
    `numpy.ndarray`
        array of interleaved data

    """
    if output is None:
        output = np.empty_like(r)

    cfa = cfa.lower()
    if cfa == 'rggb':
        output[top_left] = r[top_left]
        output[top_right] = g1[top_right]
        output[bottom_left] = g2[bottom_left]
        output[bottom_right] = b[bottom_right]
    elif cfa == 'bggr':
        output[top_left] = b[top_left]
        output[top_right] = g1[top_right]
        output[bottom_left] = g2[bottom_left]
        output[bottom_right] = r[bottom_right]
    else:
        raise ErrBadCFA

    return output


def decomposite_bayer(img, cfa='rggb'):
    """Decomposite an interleaved image into densely sampled color planes.

    Parameters
    ----------
    img : `numpy.ndarray`
        composited ndarray of shape (m, n)
    cfa : `str`, optional, {'rggb', 'bggr'}
        color filter arangement

    Returns
    -------
    r : `numpy.ndarray`
        ndarray of shape (m//2, n//2)
    g1 : `numpy.ndarray`
        ndarray of shape (m//2, n//2)
    g2 : `numpy.ndarray`
        ndarray of shape (m//2, n//2)
    b : `numpy.ndarray`
        ndarray of shape (m//2, n//2)

    """
    if cfa == 'rggb':
        r = img[top_left]
        g1 = img[top_right]
        g2 = img[bottom_left]
        b = img[bottom_right]
    elif cfa == 'bggr':
        b = img[top_left]
        g1 = img[top_right]
        g2 = img[bottom_left]
        r = img[bottom_right]

    return r, g1, g2, b


def recomposite_bayer(r, g1, g2, b, cfa='rggb', output=None):
    """Recomposite raw color planes back into a mosaic.

    This function is the reciprocal of decomposite_bayer

    Parameters
    ----------
    r : `numpy.ndarray`
        ndarray of shape (m, n)
    g1 : `numpy.ndarray`
        ndarray of shape (m, n)
    g2 : `numpy.ndarray`
        ndarray of shape (m, n)
    b : `numpy.ndarray`
        ndarray of shape (m, n)
    cfa : `str`, optional, {'rggb', 'bggr'}
        color filter arangement
    output : `numpy.ndarray`, optional
        output array, of shape (2m, 2n) and same dtype as r, g1, g2, b

    Returns
    -------
    `numpy.ndarray`
        array containing the re-composited color planes

    """
    m, n = r.shape
    if output is None:
        output = np.empty((2*m, 2*n), dtype=r.dtype)

    cfa = cfa.lower()
    if cfa == 'rggb':
        output[top_left] = r
        output[top_right] = g1
        output[bottom_left] = g2
        output[bottom_right] = b
    elif cfa == 'bggr':
        output[top_left] = b
        output[top_right] = g1
        output[bottom_left] = g2
        output[bottom_right] = r
    else:
        raise ErrBadCFA

    return output


def assemble_superresolved(r, g1, g2, b, zoomfactor, cfa='rggb', out=None):
    """Assemble a trichromatic image from super-resolved color planes.

    Parameters
    ----------
    r : `numpy.ndarray`
        ndarray of shape (m, n) representing the R bayer color channel
    g1 : `numpy.ndarray`
        ndarray of shape (m, n) representing the G1 bayer color channel
    g2 : `numpy.ndarray`
        ndarray of shape (m, n) representing the G2 bayer color channel
    b : `numpy.ndarray`
        ndarray of shape (m, n) representing the B bayer color channel
    zoomfactor : `float`
        amount of upsampling applied, e.g. 500 => 1500; zoomfactor = 3
    cfa : `str`, {'rggb', 'bggr'}
        color filter arrangement
    out : `numpy.ndarray`
        array to place the output in, shape of (m,n,3)
        if None, freshly allocated

    Returns
    -------
    `numpy.ndarray`
        array of shape (m, n, 3) containing the trichromatic image
    """
    if cfa == 'rggb':
        g2_to_g1 = [-zoomfactor, zoomfactor]  # "up and over"
        r_to_g1 = [-zoomfactor, 0]  # "over"
        b_to_g1 = [0, zoomfactor]  # "up"
    else:
        raise NotImplementedError('assemble_superresolved: only rggb patterns supported at this time')

    if out is None:
        out = np.empty((*r.shape, 3), dtype=r.dtype)

    R = fft.fft2(r)
    B = fft.fft2(b)
    G2 = fft.fft2(g2)
    Rp = ndimage.fourier_shift(R, r_to_g1)
    rp = fft.ifft2(Rp).real
    Bp = ndimage.fourier_shift(B, b_to_g1)
    bp = fft.ifft2(Bp).real
    G2p = ndimage.fourier_shift(G2, g2_to_g1)
    g2p = fft.ifft2(G2p).real
    gp = (g2p+g1)/2

    out[..., 0] = rp
    out[..., 1] = gp
    out[..., 2] = bp
    return out


# Kernels from Malvar et al, fig 2.
# names derived from the paper,
# in demosaic_malvar the naming
# may be more clear
# "G at R locations" or G at B locations
kernel_G_at_R_or_B = [
    [ 0, 0, -1, 0,  0], # NOQA
    [ 0, 0,  2, 0,  0], # NOQA
    [-1, 2,  4, 2, -1], # NOQA
    [ 0, 0,  2, 0,  0], # NOQA
    [ 0, 0, -1, 0,  0], # NOQA
]

# R at green in R row, B column
kernel_R_at_G_in_RB = [
    [ 0,  0, .5, 0,  0], # NOQA
    [ 0, -1, 0, -1,  0], # NOQA
    [-1,  4, 5,  4, -1], # NOQA
    [ 0, -1, 0, -1,  0], # NOQA
    [ 0,  0, .5, 0,  0], # NOQA
]

kernel_R_at_G_in_BR = [
    [0,  0, -1,  0, 0 ], # NOQA
    [0, -1,  4, -1, 0 ], # NOQA
    [.5, 0,  5,  0, .5], # NOQA
    [0, -1,  4, -1, 0 ], # NOQA
    [0,  0, -1,  0, 0 ], # NOQA
]

kernel_R_at_B_in_BB = [
    [0,    0, -3/2, 0,  0],   # NOQA
    [0,    2,  0,   2,  0],   # NOQA
    [-3/2, 0,  6,   0, -3/2], # NOQA
    [0,    2,  0,   2,  0],   # NOQA
    [0,    0, -3/2, 0,  0],   # NOQA
]


kernel_B_at_G_BR = kernel_R_at_G_in_RB
kernel_B_at_G_RB = kernel_R_at_G_in_BR
kernel_B_at_R_in_RR = kernel_R_at_B_in_BB


def demosaic_malvar(img, cfa='rggb'):
    """Demosaic an image using the Malvar algorithm.

    Parameters
    ----------
    img : `numpy.ndarray`
        ndarray of shape (m, n) containing mosaiced (interleaved) pixel data,
        as from a raw file
    cfa : `str`, optional, {'rggb', 'bggr'}
        color filter arrangement

    Returns
    -------
    `numpy.ndarray`
        ndarray of shape (m, n, 3) that has been demosaiced.  Final dimension
        is ordered R, G, B.  Is of the same dtype as img and has the same energy
        content and sense of z scaling

    """
    cfa = cfa.lower()
    # create all of our convolution kernels (FIR filters)
    # division by 8 is to make the kernel sum to 1
    # (preserve energy)
    kgreen = np.array(kernel_G_at_R_or_B) / 8
    kgreensameColumn = np.array(kernel_R_at_G_in_RB) / 8
    kgreensameRow = np.array(kernel_R_at_G_in_BR) / 8
    kdiagonalRB = np.array(kernel_R_at_B_in_BB) / 8

    # there is only one filter for G
    Gest = ndimage.convolve(img, kgreen)

    # there are only three unique convolutions remaining
    c1 = ndimage.convolve(img, kgreensameColumn)
    c2 = ndimage.convolve(img, kgreensameRow)
    c3 = ndimage.convolve(img, kdiagonalRB)

    red = np.empty_like(img)
    green = Gest
    blue = np.empty_like(img)

    green[top_right] = img[top_right]
    green[bottom_left] = img[bottom_left]

    if cfa == 'rggb':
        red[top_left] = img[top_left]
        red[top_right] = c2[top_right]
        red[bottom_left] = c1[bottom_left]
        red[bottom_right] = c3[bottom_right]

        blue[top_left] = c3[top_left]
        blue[top_right] = c1[top_right]
        blue[bottom_left] = c2[bottom_left]
        blue[bottom_right] = img[bottom_right]
    elif cfa == 'bggr':
        blue[top_left] = img[top_left]
        blue[top_right] = c2[top_right]
        blue[bottom_left] = c1[bottom_left]
        blue[bottom_right] = c3[bottom_right]

        red[top_left] = c3[top_left]
        red[top_right] = c1[top_right]
        red[bottom_left] = c2[bottom_left]
        red[bottom_right] = img[bottom_right]
    else:
        raise ErrBadCFA

    return np.stack((red, green, blue), axis=2)
