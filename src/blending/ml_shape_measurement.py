"""
ML Shape Measurement for Postage Stamps
=======================================

Reuses:
    - src/anacal/batch_calibration.py : Calibrator, prepare_q_images
    - src/anacal/pixel_response.py    : anacal_pix_r (via prepare_q_images)
"""

import numpy as np
from typing import Tuple, Optional

# ===========================================================================
# ML Shape Measurement
# ===========================================================================

def ml_shape_measure(
    stamps: np.ndarray,
    psf_image: np.ndarray,
    calibrator: "Calibrator",
    pixel_scale: float = 0.2,
    psf_fwhm: float = 0.85,
    noise_std: float = 0.0,
    noise_levels: Optional[np.ndarray] = None,  # per-stamp noise sigma (N,)
    n_workers: int = 32,
    flim: float = 10.0,
    dtype: str = "float32",
    ml_batch_size: int = 800,
    seed: int = 20020620,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run ML shape measurement on a set of postage stamps.

    Steps:
        1. Prepare Q-images via ``prepare_q_images()`` (from batch_calibration.py).
        2. Run ML measurement via ``calibrator.shape_measure()``.

    Parameters
    ----------
    stamps : ndarray (N, H, W)
        Postage stamps cut from the large image (already noisy).
    psf_image : ndarray (H_psf, W_psf)
        Single PSF stamp image. Will be tiled to match the number of stamps.
    calibrator : Calibrator
        Instance of ``Calibrator`` from ``src/anacal/batch_calibration.py``,
        wrapping the trained ML model.
    pixel_scale : float
        Pixel scale in arcsec/pixel.
    psf_fwhm : float
        PSF FWHM in arcsec. Used to compute sigma_arcsec = fwhm / 2.355.
    noise_std : float
        Noise standard deviation (sigma) of the image noise.
        An independent Gaussian noise realization with this amplitude is
        generated and passed to ``prepare_q_images`` for noise calibration.
        Set to 0 for noise-free data.
    noise_levels : Optional[np.ndarray] = None  # per-stamp noise sigma (N,)
        If provided, use these noise levels instead of the global noise_std.
    n_workers : int
        Number of parallel workers for Q-image preparation.
    flim : float
        Frequency limit for anacal Q-image computation.
    dtype : str
        'float32' or 'float64' — precision for Q-image and measurement.
    ml_batch_size : int
        Batch size for ML measurement.
    seed : int
        Random seed for generating the independent noise realization.

    Returns
    -------
    ml_shapes : ndarray (N, 2)
        ML-measured (e1, e2) for each stamp.
    ml_R : ndarray (N, 1, 2, 2)
        ML response matrix for each stamp.
    """
    from src.anacal.batch_calibration import prepare_q_images

    n_stamps = stamps.shape[0]
    sigma_arcsec = psf_fwhm / 2.355

    # Tile PSF to (N, H_psf, W_psf)
    psfs = np.tile(psf_image, (n_stamps, 1, 1))

    # Independent Gaussian noise realization for Q-image calibration.
    # Must NOT be the same noise that is already in the stamps —
    # it is used internally by anacal to compute noise bias corrections.
    # Supports both scalar noise_std and per-stamp noise_levels array.
    if noise_levels is not None and np.any(noise_levels > 0):
        # Per-stamp noise levels: shape (N,)  →  broadcast to (N, H, W)
        rng = np.random.default_rng(seed)
        noises = rng.normal(0, 1, stamps.shape) * noise_levels.reshape(-1, 1, 1)
    elif isinstance(noise_std, (int, float)) and noise_std > 0:
        rng = np.random.default_rng(seed)
        noises = rng.normal(0, noise_std, stamps.shape)
    else:
        noises = None

    # Prepare Q-images
    q_imgs = prepare_q_images(
        images=stamps,
        psfs=psfs,
        noises=noises,
        cat=None,                     # no subpixel offsets needed
        pixel_scale=pixel_scale,
        sigma_arcsec=sigma_arcsec,
        flim=flim,
        workers=n_workers,
    )

    # Q-images have shape (N, 5, H, W) with an extra border;
    # strip the padding added by anacal: keep the central valid region.
    # prepare_q_images returns padded images; we crop to the core:
    # The border added by anacal.image.ImageQ is typically 1 pixel on each side
    # for the 5-tap kernels used in the pixel response computation.
    # For 64x64 input with default settings, output is (N, 5, 62, 62).
    # We can use them directly — Calibrator.shape_measure handles the shape.
    if q_imgs.shape[-1] < stamps.shape[-1]:
        # Q-images are already cropped by anacal; proceed as-is
        pass

    # Cast to requested dtype
    if dtype == "float32":
        q_imgs = q_imgs.astype(np.float32)
    elif dtype == "float64":
        q_imgs = q_imgs.astype(np.float64)

    # Run ML measurement
    ml_shapes, ml_R = calibrator.shape_measure(q_imgs, batch_size=ml_batch_size)

    return ml_shapes, ml_R


# ===========================================================================
# Convenience: full Phase 3 pipeline for one batch
# ===========================================================================

def run_ml_on_detections(
    large_image: np.ndarray,
    det_cat: np.ndarray,
    psf_image: np.ndarray,
    calibrator: "Calibrator",
    stamp_size: int = 64,
    pixel_scale: float = 0.2,
    psf_fwhm: float = 0.85,
    noise_std: float = 0.0,
    noise_levels: Optional[np.ndarray] = None,
    n_workers: int = 32,
    dtype: str = "float32",
    ml_batch_size: int = 800,
    seed: int = 20020620,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function: cut stamps from large image at detection positions,
    then run ML shape measurement.

    Parameters
    ----------
    large_image, det_cat, psf_image, calibrator :
        See individual functions.
    stamp_size, pixel_scale, psf_fwhm, n_workers, dtype :
        See ``cut_stamps_from_large`` and ``ml_shape_measure``.

    Returns
    -------
    stamps : ndarray (N, stamp_size, stamp_size)
    offsets : ndarray (N, 2)
    ml_shapes : ndarray (N, 2)
    ml_R : ndarray (N, 1, 2, 2)
    """
    from src.anacal.detection_ml_pipeline import cut_stamps_from_large

    stamps, offsets = cut_stamps_from_large(
        large_image=large_image,
        det_cat=det_cat,
        stamp_size=stamp_size,
    )

    ml_shapes, ml_R = ml_shape_measure(
        stamps=stamps,
        psf_image=psf_image,
        calibrator=calibrator,
        pixel_scale=pixel_scale,
        psf_fwhm=psf_fwhm,
        noise_std=noise_std,
        noise_levels=noise_levels,
        n_workers=n_workers,
        dtype=dtype,
        ml_batch_size=ml_batch_size,
        seed=seed,
    )

    return stamps, offsets, ml_shapes, ml_R
