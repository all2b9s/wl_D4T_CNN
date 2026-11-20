from dataclasses import dataclass
from typing import Optional

import numpy as np
import anacal

RENOISE_MODES = ("rotate", "spectral")


@dataclass(frozen=True)
class RenoiseConfig:
    """Noise-bias correction shared by Q preprocessing, detection and forced FPFS.

    ``spectral`` (default since 2026-11-19) is AnaCal 0.8.4's minimal-variance
    renoising (total noise (1 + c) S_sym instead of 2 S): it needs a second
    noise realization (``noise2``) independent of both the science noise and
    the first renoise realization, with the same correlation.
    ``rotate`` is the legacy 90-degree rotated noise copy; pass
    ``RenoiseConfig(mode="rotate")`` to reproduce rotate-trained models and
    catalogs bit-identically (no extra arguments are passed to AnaCal).

    noise_corr is the (unrotated) science-noise correlation function with lag 0
    at pixel (ny // 2, nx // 2); None means white noise.  noise_rotated says
    whether the first renoise realization was drawn from the 90-degree-rotated
    correlation (xlens convention); this pipeline draws it unrotated.
    The spectral kernels depend only on the spectrum shape, not its amplitude.
    """

    mode: str = "spectral"
    c: float = 0.1
    delta: float = 0.01
    noise_corr: Optional[np.ndarray] = None
    noise_rotated: bool = False

    def __post_init__(self):
        if self.mode not in RENOISE_MODES:
            raise ValueError(f"renoise mode must be one of {RENOISE_MODES}, got {self.mode!r}")

    @property
    def spectral(self) -> bool:
        return self.mode == "spectral"

    def _spectrum(self, n, scale, variance):
        corr = None if self.noise_corr is None else np.ascontiguousarray(self.noise_corr, dtype=np.float64)
        return anacal.renoise.make_noise_spectrum(int(n), float(scale), float(variance), corr)

    def task_setup(self, cell_size, scale, variance):
        """``renoise.Setup`` for ``anacal.task.Task.process_image`` (spectral only).

        The spectrum must live on the detection cell grid (e.g. 250), not the stamp grid.
        """
        if not self.spectral:
            return None
        return anacal.renoise.Setup(
            c=self.c, spectra=[self._spectrum(cell_size, scale, variance)],
            noise_rotated=self.noise_rotated, mode="spectral", delta=self.delta,
        )

    def fpfs_kwargs(self) -> dict:
        """Extra kwargs for ``anacal.fpfs.process_image`` (empty for rotate)."""
        if not self.spectral:
            return {}
        return dict(renoise_mode="spectral", renoise_c=self.c, renoise_delta=self.delta,
                    noise_corr=self.noise_corr, noise_rotated=self.noise_rotated)

    def q_kernels(self, psf, n, scale, klim):
        """``RenoiseKernels`` for ``ImageQ`` on the n x n stamp grid (spectral only)."""
        if not self.spectral:
            return None
        return anacal.renoise.make_renoise_kernels(
            self._spectrum(n, scale, 1.0), np.ascontiguousarray(psf, dtype=np.float64),
            klim, self.c, self.delta, True, self.noise_rotated, "spectral",
        )


def anacal_pix_r(img, psf, center=None,
                 sigma_arcsec=0.85/2.355,
                 scale_arcsec_per_pix=0.2,
                 freq_lim=10.0,
                 noise_map=None,
                 noise2_map=None,
                 renoise: Optional[RenoiseConfig] = None):
    """
    Computes the pixel response for a given image and point spread function (PSF).

    Parameters:
    img (numpy.ndarray): The input image array.
    psf (numpy.ndarray): The point spread function array.
    center (tuple, optional): The (x, y) coordinates of the center. If None, the center of the image is used.
    sigma_arcsec (float, optional): The sigma value in arcseconds for Gaussian smoothing. Default is 0.85/2.355.
    scale_arcsec_per_pix (float, optional): The scale in arcseconds per pixel. Default is 0.2.
    freq_lim (float, optional): The frequency limit for the computation. Default is 10.0.
    noise_map: first renoise realization (rotate: rotated and added; spectral: eta0 / tau).
    noise2_map: second independent realization, the spectral spin-2 filler (spectral only).
    renoise: RenoiseConfig; None means RenoiseConfig() (spectral) whenever
        noise_map is given; mode 'rotate' keeps the legacy call.

    Returns:
    numpy.ndarray: get [v,g1,g2,j1,j2] as numpy (float64)

    """

    klim = freq_lim / scale_arcsec_per_pix
    nx = img.shape[1]
    ny = img.shape[0]

    if center is None:
        xcen = nx // 2
        ycen = ny // 2
    else:
        xcen, ycen = center

    xcen = int(xcen+0.5)
    ycen = int(ycen+0.5)

    iq = anacal.image.ImageQ(nx, ny, scale_arcsec_per_pix, sigma_arcsec, klim, True)

    # get [v,g1,g2,j1,j2] as numpy (float64)
    if renoise is None and noise_map is not None:
        renoise = RenoiseConfig()
    if renoise is not None and renoise.spectral:
        if noise_map is None or noise2_map is None:
            raise ValueError("spectral renoising (the default) needs noise_map and an independent "
                             "noise2_map; pass renoise=RenoiseConfig(mode='rotate') for the legacy path")
        if nx != ny:
            raise ValueError(f"spectral renoising needs a square stamp, got {ny}x{nx}")
        kernels = renoise.q_kernels(psf, nx, scale_arcsec_per_pix, klim)
        return iq.prepare_qnumber_image(img, psf, xcen, ycen, noise_map, noise2_map, kernels)
    if noise_map is None:
        return iq.prepare_qnumber_image(img, psf, xcen, ycen)
    return iq.prepare_qnumber_image(img, psf, xcen, ycen, noise_map)
