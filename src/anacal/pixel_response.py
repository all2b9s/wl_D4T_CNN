import anacal
import numpy as np

def anacal_pix_r(img, psf, center=None, 
                 sigma_arcsec=0.85/2.355, 
                 scale_arcsec_per_pix=0.2,
                 freq_lim=10.0,
                 noise_map = None):
    """
    Computes the pixel response for a given image and point spread function (PSF).

    Parameters:
    img (numpy.ndarray): The input image array.
    psf (numpy.ndarray): The point spread function array.
    center (tuple, optional): The (x, y) coordinates of the center. If None, the center of the image is used.
    sigma_arcsec (float, optional): The sigma value in arcseconds for Gaussian smoothing. Default is 0.7/2.355.
    scale_arcsec_per_pix (float, optional): The scale in arcseconds per pixel. Default is 0.2.
    freq_lim (float, optional): The frequency limit for the computation. Default is 10.0.

    Returns:
    numpy.ndarray: get [v,g1,g2,j1,j2] as numpy (float64)

    """
    
    klim = freq_lim / scale_arcsec_per_pix
    nx = img.shape[0]
    ny = img.shape[1]
    if center is None:
        xcen = nx // 2
        ycen = ny // 2
    else:
        xcen, ycen = center

    iq = anacal.image.ImageQ(nx, ny, scale_arcsec_per_pix, sigma_arcsec, klim, True)

    # get [v,g1,g2,j1,j2] as numpy (float64)
    if noise_map is None:
        q_img_np = iq.prepare_qnumber_image(img, psf, xcen, ycen)
    else:
        q_img_np = iq.prepare_qnumber_image(img, psf, xcen, ycen, noise_map)
    return q_img_np

