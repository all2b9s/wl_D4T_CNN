import numpy as np
import galsim

def get_ellipticity(a,b,theta):
    """Convert from (a,b,theta) to (e1,e2)"""

    e = (a**2 - b**2)
    e[e>0]/= (a[e>0]**2 + b[e>0]**2)
    e1 = e * np.cos(2*theta/180*np.pi)
    e2 = e * np.sin(2*theta/180*np.pi)
    return np.array([e1, e2])

def get_weighted_e(a_b, b_b, theta_b, flux_b,
                   a_d, b_d, theta_d, flux_d,
                   angle):
    """Compute the weighted ellipticity of a bulge+disk system"""
    e_b = get_ellipticity(a_b, b_b, theta_b+angle/np.pi*180)
    e_d = get_ellipticity(a_d, b_d, theta_d+angle/np.pi*180)
    e = (flux_b*e_b + flux_d*e_d)/(flux_b+flux_d)
    return e

def galsim_e(bulge_params, disk_params, angle,shift=[0,0],
                n_bulge=4.0, n_disk=1.0,
                pixel_scale=0.2, stamp_size=64):
    """
    Parameters
    ----------
    bulge_params : (a, b, pa, flux)
    disk_params : (a, b, pa, flux)
    n_bulge : float
    n_disk : float
    pixel_scale : float
    stamp_size : int

    Returns
    -------
    img_total : galsim.Image
    ellip : dict {e1, e2}
    """

    def make_comp(a, b, pa, flux, n):
        q = b / a
        hlr = np.sqrt(a * b)   # area-preserving
        base = galsim.Sersic(n=n, half_light_radius=hlr, flux=flux)
        return base.shear(q=q, beta=pa * galsim.degrees)
    disk  = make_comp(*disk_params,  n=n_disk)
    galaxy = disk
    if bulge_params[0]*bulge_params[1]>0:
        bulge = make_comp(*bulge_params, n=n_bulge)
        galaxy = bulge + disk
    galaxy = galaxy.rotate(angle * galsim.radians).shift(shift[0], shift[1])
    psf = galsim.Gaussian(fwhm=0.8)
    final = galsim.Convolve([galaxy, psf])
    img_total = final.drawImage(nx=stamp_size, ny=stamp_size,
                             scale=pixel_scale,
                             method="no_pixel",
                             )

    try:
        res = galsim.hsm.FindAdaptiveMom(img_total)
        e1, e2 = res.observed_shape.e1, res.observed_shape.e2
        ellip = {"e1": e1, "e2": e2, "e": np.hypot(e1, e2)}
    except galsim.errors.GalSimHSMError:
        ellip = {"e1": np.nan, "e2": np.nan, "e": np.nan}

    return img_total, [e1,e2]

def get_snr(img, noise_sigma, crop_size=3):
    """
    img : [B,C,H,W]
    """
    center = (img.shape[-2]//2, img.shape[-1]//2)
    img_crop = img[..., center[0]-crop_size:(center[0]+crop_size+1), center[1]-crop_size:(center[1]+crop_size+1)]
    flux = img_crop.sum(axis=(-2,-1))
    noise_amp = noise_sigma * ((2*crop_size+1))
    return flux/(2*crop_size+1)**2, flux/noise_amp
