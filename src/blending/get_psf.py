import numpy as np
from lsst.afw.image import ExposureF
from lsst.geom import Point2D
from xlens.utils.image import resize_array
import matplotlib.pyplot as plt

# 1) Read the exposure FITS off disk.
expfname = '/taiga/illinois/las/astro/xinliuxl/DC1_sim_blended/sim_mode0/exp-i-09993.fits'
exposure = ExposureF.readFits(expfname)

# 2) Pull the PSF object off the exposure.
lsst_psf = exposure.getPsf()

# 3) Evaluate the PSF at a sky position (here: the centre of the exposure).
# since the exposure has the same PSF
# `computeImage` returns an lsst.afw.image.ImageD whose .getArray() gives a
# numpy view of the pixel data.
bbox = exposure.getBBox()
xc = bbox.getMinX() + bbox.getWidth()  // 2
yc = bbox.getMinY() + bbox.getHeight() // 2
psf_array = lsst_psf.computeImage(Point2D(xc, yc)).getArray()
plt.imshow(psf_array, origin='lower')
plt.colorbar()
plt.title("PSF at the center of the exposure")
plt.savefig("psf_center.png")
plt.show()


# 4) Force a fixed stamp size (matches xlens.utils.image.resize_array). The PSF
# image returned by computeImage is naturally sized — pad/crop to `npix`, and
# the center is at (npix//2, npix//2)
#psf_array = resize_array(psf_array, (npix, npix))