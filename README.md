# pmapper
Poisson Maximum A-Posteriori algorithm in python 3.6+

The name of this repository is an homage to MTF-Mapper, a slanted edge MTF measurement program written by Frans van den Bergh.

The contents are implementations of PMAP, an image restoration and superresolution algorithm developed by Joe Green, who is now a Principal Optical Engineer at JPL.  Several flavors of PMAP are implemented under a common interface.  They are:

- `PMAP`, the contemporary PMAP algorithm
- `MFPMAP`, Multi-Frame PMAP

All support basic as well as super-resolution variants, based on whether the input PSF is of the same shape (rows x cols) as the input image(s) or not.  At this time, they only support panchromatic imagery, but a future extension to Bayer detectors is planned.  The algorithms are implemented in a backend-agonstic way which allows the user to execute the algorithms on either a CPU or a GPU at whim.
