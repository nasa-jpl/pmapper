# pmapper
Poisson Maximum A-Posteriori algorithm in python 3.6+

The name of this repository is an homage to MTF-Mapper, a slanted edge MTF measurement program written by Frans van den Bergh.

The contents are implementations of PMAP, an image restoration and superresolution algorithm developed by Joe Green, who is now a Principal Optical Engineer at JPL.  Several flavors of PMAP are implemented under a common interface.  They are:

- `PMAP`, the contemporary PMAP algorithm
- `BayerPMAP`, PMAP, adapted for raw bayer data (no demosaicing requried)
- `MFPMAP`, Multi-Frame PMAP
- `BayerMFPMAP`, Multi-Frame PMAP, adapted to bayer same as PMAP.

All support basic as well as super-resolution variants, based on whether the input PSF is of the same shape (rows x cols) as the input image(s) or not.

This implementation of PMAP is extremely fast.  PMAP itself is usually superior to Richardson-Lucy deconvolution or a Weiner filter.
