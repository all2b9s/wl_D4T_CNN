"""
FPFS Detection + ML Shape Measurement Pipeline
==============================================

Phase 1: Stitch isolated galaxy cutouts into large images
Phase 2: Run AnaCal detection + measurement (anacal.task.Task,
         do_measure=True → wsel selection weight) +
         FPFS measurement (anacal.fpfs.process_image, detection=...)
Phase 3: Cut postage stamps at detected positions & run ML shape measurement
Phase 4: Merge anacal selection info (wsel/dwsel, m00/dm00, FPFS shape)
         with ML results (shape/R)

Reuses heavily from:
    - src/anacal/cal_toolkit.py   (load_img, build_dataset, format_number)
    - src/anacal/batch_calibration.py (Calibrator, prepare_q_images)
    - src/anacal/fpfs.py          (FpfsConfig, process_image reference)
"""

import numpy as np
import os
from typing import Tuple, Optional, Any

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import anacal
except ImportError:
    anacal = None

# ===========================================================================
# Phase 1: Stitching
# ===========================================================================

def generate_grid_positions(
    nx: int = 10,
    ny: int = 10,
    sep: int = 80,
    margin: int = 32,
    stamp_size: int = 64,
) -> Tuple[np.ndarray, int]:
    """
    Generate regular grid positions for placing cutouts on a large canvas.

    Parameters
    ----------
    nx, ny : int
        Number of columns (nx) and rows (ny) in the grid.
    sep : int
        Separation between adjacent galaxy centers, in pixels.
    margin : int
        Margin from canvas edge to the outermost stamp edges, in pixels.
    stamp_size : int
        Size of each cutout stamp, in pixels (assumed square).

    Returns
    -------
    positions : ndarray of shape (nx*ny, 2)
        (y_center, x_center) for each grid cell, row-major order.
    canvas_size : int
        Required canvas dimension (square) in pixels.
    """
    # Canvas: margin on each side + (max_dim-1)*sep + one stamp_size
    max_dim = max(nx, ny)
    canvas_size = 2 * margin + (max_dim - 1) * sep + stamp_size

    positions = []
    for iy in range(ny):          # row
        for ix in range(nx):      # column
            yc = margin + stamp_size // 2 + iy * sep
            xc = margin + stamp_size // 2 + ix * sep
            positions.append((yc, xc))

    return np.array(positions, dtype=np.int32), canvas_size


def stitch_large_image(
    cutouts: np.ndarray,
    positions: np.ndarray,
    canvas_size: int,
) -> np.ndarray:
    """
    Stitch cutout stamps onto a large blank canvas at given positions.

    Each cutout is placed onto the canvas by adding its pixel values.
    No additional noise is added — noise is assumed to already be in the cutouts.

    Parameters
    ----------
    cutouts : ndarray of shape (N, H, W)
        The cutout stamps to place. Already PSF-convolved and (optionally) noisy.
    positions : ndarray of shape (N, 2)
        (y_center, x_center) for each cutout, in pixels.
    canvas_size : int
        Side length of the square canvas.

    Returns
    -------
    large_image : ndarray of shape (canvas_size, canvas_size)
        The stitched large image.
    """
    n_cutouts = cutouts.shape[0]
    stamp_h, stamp_w = cutouts.shape[1], cutouts.shape[2]
    half_h = stamp_h // 2
    half_w = stamp_w // 2

    large_image = np.zeros((canvas_size, canvas_size), dtype=cutouts.dtype)

    for idx in range(n_cutouts):
        yc, xc = positions[idx]
        y1 = yc - half_h
        y2 = yc + half_h
        x1 = xc - half_w
        x2 = xc + half_w

        # Basic bounds check
        if y1 < 0 or x1 < 0 or y2 > canvas_size or x2 > canvas_size:
            print(f"[stitch] WARNING: cutout {idx} at ({yc},{xc}) "
                  f"exceeds canvas bounds ({canvas_size}). Skipping.")
            continue

        large_image[y1:y2, x1:x2] += cutouts[idx]

    return large_image


# ===========================================================================
# Convenience: load one batch + stitch
# ===========================================================================

def load_and_stitch_one_batch(
    shear_comp: str,
    shear_mode: int,
    index: int,
    directory: str,
    abs_shear_value: float = 0.02,
    noise_level: float = 0.594,
    noise_mode: str = "fixed",
    nx: int = 10,
    ny: int = 10,
    sep: int = 80,
    margin: int = 32,
    stamp_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any, int, np.ndarray, np.ndarray]:
    """
    Load one batch of cutouts (using load_img from cal_toolkit) and stitch
    them into a large image.

    Parameters
    ----------
    shear_comp : str
        Shear component, e.g. 'g1' or 'g2'.
    shear_mode : int
        Shear mode (0, 1, 2).
    index : int
        Batch index (corresponds to img_{index}/ directory).
    directory : str
        Root directory of the dataset.
    abs_shear_value : float
        Absolute shear value used in simulation.
    noise_level : float
        Noise standard deviation (ADU). Pass 0 for noise-free.
    noise_mode : str
        'fixed', 'adaptive', or 'uniform'.
    nx, ny : int
        Grid dimensions.
    sep : int
        Separation between galaxy centers in pixels.
    margin : int
        Margin from canvas edge in pixels.
    stamp_size : int
        Size of each cutout stamp.

    Returns
    -------
    large_image : ndarray (canvas_size, canvas_size)
    positions : ndarray (nx*ny, 2)
    cutouts : ndarray (nx*ny, stamp_size, stamp_size)
    cat : pd.DataFrame
        Truth catalog concatenated across the batch.
    canvas_size : int
    psf_img : ndarray
        PSF stamp image.
    noise_levels : ndarray
        Noise levels for each cutout.
    """
    from src.anacal.cal_toolkit import load_img

    cutouts, psf_img, cat, noise_levels = load_img(
        shear_comp=shear_comp,
        shear_mode=shear_mode,
        index=index,
        abs_shear_value=abs_shear_value,
        noise_level=noise_level,
        noise_mode=noise_mode,
        directory=directory,
    )

    # load_img tiles the PSF to (N, 64, 64) for per-galaxy use;
    # extract the single (64, 64) PSF stamp for detection on the large image.
    if psf_img.ndim == 3:
        psf_img = psf_img[0]

    n_gal = cutouts.shape[0]

    # Truncate or pad: expect exactly nx*ny galaxies
    expected = nx * ny
    if n_gal < expected:
        print(f"[load_and_stitch] WARNING: only {n_gal} cutouts in batch {index}, "
              f"expected {expected}. Using all {n_gal}.")
        # Recompute grid for actual count
        actual_ny = int(np.ceil(n_gal / nx))
        positions, canvas_size = generate_grid_positions(
            nx=nx, ny=actual_ny, sep=sep, margin=margin, stamp_size=stamp_size
        )
        # Trim positions to match
        positions = positions[:n_gal]
    elif n_gal > expected:
        print(f"[load_and_stitch] WARNING: {n_gal} cutouts in batch {index}, "
              f"truncating to {expected}.")
        cutouts = cutouts[:expected]
        cat = cat.iloc[:expected]
        noise_levels = noise_levels[:expected]
        positions, canvas_size = generate_grid_positions(
            nx=nx, ny=ny, sep=sep, margin=margin, stamp_size=stamp_size
        )
    else:
        positions, canvas_size = generate_grid_positions(
            nx=nx, ny=ny, sep=sep, margin=margin, stamp_size=stamp_size
        )

    large_image = stitch_large_image(cutouts, positions, canvas_size)

    return large_image, positions, cutouts, cat, canvas_size, psf_img, noise_levels


# ===========================================================================
# Internal helpers: AnaCal detection (new two-step API)
# ===========================================================================

def _build_task_from_fpfs_config(
    mag_zero: float = 30.0,
    pixel_scale: float = 0.2,
    sigma_arcsec: float = 0.40,
    snr_peak_min: float = 5.0,
    omega_f: Optional[float] = None,
    omega_v: Optional[float] = None,
    fpfs_c0: float = 30.0,
    num_epochs: int = 0,
    force_center: bool = True,
    prior: Optional[Any] = None,
    prior_sigma_a: float = 0.05,
    prior_sigma_x: float = 0.05,
):
    """Build an ``anacal.task.Task`` for source detection + measurement.

    Task used with ``process_image(do_measure=True)``, so the output catalog
    carries the full selection weight ``wsel`` / ``dwsel_dg{1,2}`` (the
    detection weight refined by the FPFS flux/size cut) that the shear
    estimator uses -- the same configuration as ``example_fpfs_blended.ipynb``.

    The flux-related thresholds (``omega_f``, ``omega_v``, ``fpfs_c0``) are
    BASE values defined at the reference zeropoint ``mag_zero = 31.4``
    (``anacal.task.THRESHOLD_REF_MAG_ZERO``); the C++ ``Task`` constructor
    rescales them internally to this image's ``mag_zero``.  Callers must pass
    the base-31.4 values unchanged -- do NOT rescale them here (the legacy
    ``10**((mag_zero - 30)/2.5)`` base-30 scaling would double-scale).

    Parameters
    ----------
    mag_zero : float
        Photometric zero point.
    pixel_scale : float
        Pixel scale in arcsec / pixel.
    sigma_arcsec : float
        Detection smoothing kernel size in arcsec (default 0.40).
        Controls peak-finding SNR; larger values smooth more noise
        but may blend close sources.
    snr_peak_min : float
        Minimum peak SNR for detection (default 10.0).
    omega_f : float or None
        Base flux threshold for peak selection (defined at mag_zero=31.4).
        Default ``0.218`` (example_fpfs_blended; effective ~0.060 at
        mag_zero=30).
    omega_v : float or None
        Base variance threshold (defined at mag_zero=31.4).
        Default ``0.011`` (effective ~0.0030 at mag_zero=30).
    fpfs_c0 : float
        Base FPFS flux/size selection cut for the measurement stage
        (defined at mag_zero=31.4).  Default 30.0 (example).
    num_epochs : int
        Measurement fit epochs.  Default 0 (example).
    force_center : bool
        Force the measured center to the detection position.
        Default True (example).
    prior : anacal.ngmix.modelPrior or None
        Prior for the C++ ngmix fit.  If None, a default ``modelPrior`` with
        sigma_a = sigma_x = 0.05 is built (same as example).
    prior_sigma_a, prior_sigma_x : float
        Prior widths used when ``prior`` is None.

    Returns
    -------
    task : anacal.task.Task
    """
    import anacal

    # omega_f / omega_v / fpfs_c0 are BASE thresholds defined at the
    # reference zeropoint THRESHOLD_REF_MAG_ZERO = 31.4; the C++ Task
    # rescales them internally to this image's mag_zero (task.h).  Pass the
    # base-31.4 values unchanged -- the legacy base-30 external rescaling
    # (10**((mag_zero - 30)/2.5)) would double-scale.
    if omega_f is None:
        omega_f = 0.218   # example_fpfs_blended (effective ~0.060 at mag_zero=30)
    if omega_v is None:
        omega_v = 0.011   # example_fpfs_blended (effective ~0.0030 at mag_zero=30)
    if prior is None:
        # Same prior as example_fpfs_blended.ipynb.
        prior = anacal.ngmix.modelPrior()
        prior.set_sigma_a(anacal.math.qnumber(prior_sigma_a))
        prior.set_sigma_x(anacal.math.qnumber(prior_sigma_x))

    task = anacal.task.Task(
        scale=pixel_scale,
        sigma_arcsec=sigma_arcsec,
        snr_peak_min=snr_peak_min,
        omega_f=omega_f,
        omega_v=omega_v,
        prior=prior,
        num_epochs=num_epochs,
        force_center=force_center,
        fpfs_c0=fpfs_c0,
        mag_zero=mag_zero,
    )
    return task


def _build_cell_list(
    img_height: int,
    img_width: int,
    pixel_scale: float,
    cell_nx: int = 250,
    cell_ny: int = 250,
    cell_overlap: int = 80,
) -> list:
    """Build processing cells for ``anacal.task.Task.process_image``.

    Tiles the image with ``anacal.geometry.get_cell_list`` for detection.
    The (constant) PSF is passed directly to ``process_image`` — no PSF
    attachment to cells is needed.

    Parameters
    ----------
    img_height, img_width : int
        Image dimensions in pixels.
    pixel_scale : float
        Pixel scale in arcsec / pixel.
    cell_nx, cell_ny : int
        Cell size in pixels (default 250).
    cell_overlap : int
        Overlap between cells in pixels (default 80).

    Returns
    -------
    cells : list of ``anacal.geometry.Cell``
    """
    import anacal

    return anacal.geometry.get_cell_list(
        img_nx=img_width,
        img_ny=img_height,
        cell_nx=cell_nx,
        cell_ny=cell_ny,
        cell_overlap=cell_overlap,
        scale=pixel_scale,
    )


# ===========================================================================
# Phase 2: AnaCal Detection + FPFS Measurement (new two-step API)
# ===========================================================================

def run_detection_and_fpfs(
    large_image: np.ndarray,
    psf_image: np.ndarray,
    fpfs_config: Any = None,  # anacal.fpfs.FpfsConfig
    mag_zero: float = 30.0,
    pixel_scale: float = 0.2,
    noise_variance: float = 1e-4,
    noise_array: Optional[np.ndarray] = None,
    task: Any = None,  # cached anacal.task.Task (detection)
    cells: Any = None,  # cached cell list
    sigma_arcsec: float = 0.40,
    snr_peak_min: float = 5.0,
    omega_f: Optional[float] = None,
    omega_v: Optional[float] = None,
    fpfs_c0: float = 30.0,
    num_epochs: int = 0,
    force_center: bool = True,
    prior: Optional[Any] = None,
) -> np.ndarray:
    """
    Run AnaCal detection + FPFS measurement on a large image (new two-step API).

    FPFS no longer detects internally.  This function runs:

    1. **Detection + measurement** — ``anacal.task.Task.process_image(
       do_measure=True)`` with a cell-list tiling.  Returns detector
       positions (``x1_det``, ``x2_det`` in arcsec) and the full
       differentiable selection weight (``wsel``, ``dwsel_dg1``,
       ``dwsel_dg2`` — detection weight refined by the FPFS flux/size cut).
    2. **FPFS measurement** — the Task's task-kernel FPFS measurement
       (``fpfs_e{1,2}`` / ``fpfs_de{1,2}_dg{1,2}`` / ``fpfs_m0``) is the
       production estimator, self-consistent with the selection weight.
       A forced multi-kernel measurement (``anacal.fpfs.process_image``,
       ``fpfs1_*``) is also run and kept as a fallback.

    The selection weight is carried into the output catalog as
    ``fpfs_w`` / ``fpfs_dw_dg{1,2}``, and the task-kernel FPFS quantities as
    ``fpfs_e{1,2}`` / ``fpfs_R{11,22}`` / ``fpfs_m00``.

    Parameters
    ----------
    large_image : ndarray (H, W)
        The large image with galaxies.
    psf_image : ndarray (H_psf, W_psf)
        PSF stamp image.
    fpfs_config : anacal.fpfs.FpfsConfig
        FPFS measurement configuration (two shapelet scales).
    mag_zero : float
        Magnitude zero point.
    pixel_scale : float
        Pixel scale in arcsec/pixel.
    noise_variance : float
        Noise variance (sigma^2) for the image.
    noise_array : ndarray or None
        Pure-noise realization with the same statistics as the image
        noise (only needed for noisy images).  Used for the renoising
        noise-bias correction: it is forwarded to BOTH the Task
        (``task.process_image`` -- detection + task-kernel FPFS
        measurement) and the forced ``anacal.fpfs.process_image`` call.
        When present, the Task internally adds the noise to the
        deconvolved data and doubles the input ``noise_variance`` so the
        renoised covariance is consistent.  Pass ``None`` only for
        noise-free images.
    task : anacal.task.Task or None
        Cached detection Task. If None, a new one is built.
    cells : list of anacal.geometry.Cell or None
        Cached cell list. If None, a new one is built.
    sigma_arcsec : float
        Detection smoothing kernel in arcsec (default 0.40).
    snr_peak_min : float
        Minimum peak SNR for detection (default 5.0).
    omega_f, omega_v : float or None
        Base flux / variance thresholds, defined at mag_zero = 31.4 and
        rescaled internally by the Task (defaults 0.218 / 0.011).
    fpfs_c0 : float
        Base FPFS flux/size selection cut for the measurement stage
        (defined at mag_zero = 31.4; default 30.0).
    num_epochs : int
        Measurement fit epochs (default 0).
    force_center : bool
        Force the measured center to the detection position (default True).
    prior : anacal.ngmix.modelPrior or None
        Prior for the C++ ngmix fit (default: sigma_a = sigma_x = 0.05).

    Returns
    -------
    det_cat : ndarray
        Structured array with one row per detected object.
        Columns include detection positions, FPFS shapes, responses,
        selection weights (``wsel``), and flux measurements.
    """
    import anacal

    gal_array = large_image.astype(np.float64, copy=False)
    psf_array = psf_image.astype(np.float64, copy=False)

    if noise_array is not None:
        noise_array = noise_array.astype(np.float64, copy=False)

    # ---- Step 1: AnaCal detection + measurement (do_measure=True) ----
    if task is None:
        task = _build_task_from_fpfs_config(
            mag_zero=mag_zero,
            pixel_scale=pixel_scale,
            sigma_arcsec=sigma_arcsec,
            snr_peak_min=snr_peak_min,
            omega_f=omega_f,
            omega_v=omega_v,
            fpfs_c0=fpfs_c0,
            num_epochs=num_epochs,
            force_center=force_center,
            prior=prior,
        )
    if cells is None:
        cells = _build_cell_list(
            img_height=gal_array.shape[0],
            img_width=gal_array.shape[1],
            pixel_scale=pixel_scale,
        )

    det_cat_raw = task.process_image(
        np.asarray(gal_array, dtype=np.float32),
        psf_array,
        variance=noise_variance,
        # Renoising (noise-bias correction): passing the pure-noise
        # realization activates the analytic noise correction inside the
        # Task -- it adds the deconvolved noise to the measurement data
        # and doubles ``variance`` internally (task.h).  This is required
        # for the production task-kernel FPFS estimator (fpfs_e1 / fpfs_R11)
        # to be unbiased under noise; without it the moment ratio
        # e = M22/(M00+C0) picks up a positive multiplicative noise bias.
        noise_array=noise_array,
        cell_list=cells,
        do_measure=True,
    )

    n_det = len(det_cat_raw) if det_cat_raw is not None else 0
    if n_det == 0:
        return _empty_det_cat()

    # ---- Convert detector positions (arcsec) → pixel catalogue ----
    det_names = list(det_cat_raw.dtype.names)
    pos_y_arc = _extract_column(det_cat_raw, det_names, ["x2_det", "x2", "y"])
    pos_x_arc = _extract_column(det_cat_raw, det_names, ["x1_det", "x1", "x"])
    if pos_y_arc is None or pos_x_arc is None:
        return _empty_det_cat()

    detection = np.zeros(n_det, dtype=[("y", "f8"), ("x", "f8")])
    detection["y"] = pos_y_arc / pixel_scale
    detection["x"] = pos_x_arc / pixel_scale

    # ---- Step 2: FPFS measurement at detected positions ----
    # FPFS requires the PSF stamp to be (npix, npix) with npix =
    # fpfs_config.npix (default 64).  The raw PSF may be a different size
    # (e.g. 48×48 after border trimming), so resize it to match -- the
    # same convention the legacy block-based pipeline used (it resized
    # the PSF to stamp_size before measurement).
    npix = fpfs_config.npix
    if psf_array.shape != (npix, npix):
        psf_fpfs = anacal.utils.resize_array(psf_array, (npix, npix))
    else:
        psf_fpfs = psf_array

    fpfs_out = anacal.fpfs.process_image(
        fpfs_config=fpfs_config,
        mag_zero=mag_zero,
        gal_array=gal_array,
        psf_array=psf_fpfs,
        pixel_scale=pixel_scale,
        noise_variance=noise_variance,
        noise_array=noise_array,
        detection=detection,
    )

    # ---- Merge detector weight + FPFS shapes ----
    return _merge_detection_and_fpfs(det_cat_raw, fpfs_out, pixel_scale)


def _merge_detection_and_fpfs(
    det_cat: np.ndarray,
    fpfs_out: np.ndarray,
    pixel_scale: float,
) -> np.ndarray:
    """
    Merge AnaCal detector output with FPFS measurement output.

    Detector output provides positions (``x1_det``, ``x2_det`` in arcsec),
    the full differentiable selection weight (``wsel``, ``dwsel_dg1``,
    ``dwsel_dg2`` — detection weight refined by the FPFS flux/size cut)
    and the task-kernel FPFS shapes/flux (``fpfs_e{1,2}``,
    ``fpfs_de{1,2}_dg{1,2}``, ``fpfs_m0``).  The task-kernel quantities are
    the production estimator and are used directly; the forced ``fpfs1_*``
    columns in ``fpfs_out`` are only a fallback.

    Parameters
    ----------
    det_cat : ndarray
        Structured array from ``anacal.task.Task.process_image(do_measure=True)``.
    fpfs_out : ndarray
        Structured array from ``anacal.fpfs.process_image(detection=...)``.
    pixel_scale : float
        Pixel scale for arcsec → pixel conversion.

    Returns
    -------
    merged : ndarray
        Standardized detection catalog (same dtype as before).
    """
    if det_cat is None or len(det_cat) == 0:
        return _empty_det_cat()

    n_det = len(det_cat)
    det_names = list(det_cat.dtype.names)
    fpfs_names = list(fpfs_out.dtype.names) if fpfs_out is not None else []

    # Consistency guard: never silently mix task-kernel columns with the
    # forced fpfs1_* columns (measured at a different shapelet scale).
    _check_task_fpfs_columns(det_names)

    # Build a clean structured array (unchanged schema)
    dtype = np.dtype([
        ("det_id", np.int32),
        ("y", np.float64), ("x", np.float64),
        ("fpfs_e1", np.float64), ("fpfs_e2", np.float64),
        ("fpfs_R11", np.float64), ("fpfs_R22", np.float64),
        ("fpfs_w", np.float64),
        ("fpfs_dw_dg1", np.float64), ("fpfs_dw_dg2", np.float64),
        ("fpfs_m00", np.float64),
        ("fpfs_dm00_dg1", np.float64), ("fpfs_dm00_dg2", np.float64),
    ])

    merged = np.empty(n_det, dtype=dtype)
    merged["det_id"] = np.arange(n_det, dtype=np.int32)

    # --- positions: from detector (x2_det → y, x1_det → x; arcsec → pixels) ---
    pos_y = _extract_column(det_cat, det_names, ["x2_det", "x2", "y"])
    pos_x = _extract_column(det_cat, det_names, ["x1_det", "x1", "x"])
    merged["y"] = pos_y / pixel_scale if pos_y is not None else np.nan
    merged["x"] = pos_x / pixel_scale if pos_x is not None else np.nan

    # --- selection weight: from detector (wsel = wdet * FPFS cut) ---
    merged["fpfs_w"] = _fill_nan(
        _extract_column(det_cat, det_names, ["wsel", "wdet", "w"]), n_det
    )
    merged["fpfs_dw_dg1"] = _fill_nan(
        _extract_column(det_cat, det_names, ["dwsel_dg1", "dwdet_dg1", "dw_dg1"]), n_det
    )
    merged["fpfs_dw_dg2"] = _fill_nan(
        _extract_column(det_cat, det_names, ["dwsel_dg2", "dwdet_dg2", "dw_dg2"]), n_det
    )

    # --- FPFS shapes (task kernel, self-consistent with wsel/dwsel) ---
    # The production FPFS estimator is the Task's task-kernel measurement
    # (fpfs_e{1,2} / fpfs_de{1,2}_dg{1,2} / fpfs_m0), the same kernel that
    # wsel/dwsel come from (see example_fpfs_blended.ipynb).  The forced
    # fpfs1_* columns are only a defensive fallback; the consistency guard
    # _check_task_fpfs_columns guarantees they are never actually used.
    provenance = {}

    col, src = _extract_preferred(
        det_cat, det_names, fpfs_out, fpfs_names,
        ["fpfs_e1", "e1"],
        ["fpfs1_e1", "fpfs_e1", "e1"])
    merged["fpfs_e1"] = _fill_nan(col, n_det)
    provenance["fpfs_e1"] = src

    col, src = _extract_preferred(
        det_cat, det_names, fpfs_out, fpfs_names,
        ["fpfs_e2", "e2"],
        ["fpfs1_e2", "fpfs_e2", "e2"])
    merged["fpfs_e2"] = _fill_nan(col, n_det)
    provenance["fpfs_e2"] = src

    col, src = _extract_preferred(
        det_cat, det_names, fpfs_out, fpfs_names,
        ["fpfs_de1_dg1", "de1_dg1", "R11"],
        ["fpfs1_de1_dg1", "fpfs_de1_dg1", "de1_dg1", "R11"])
    merged["fpfs_R11"] = _fill_nan(col, n_det)
    provenance["fpfs_R11"] = src

    col, src = _extract_preferred(
        det_cat, det_names, fpfs_out, fpfs_names,
        ["fpfs_de2_dg2", "de2_dg2", "R22"],
        ["fpfs1_de2_dg2", "fpfs_de2_dg2", "de2_dg2", "R22"])
    merged["fpfs_R22"] = _fill_nan(col, n_det)
    provenance["fpfs_R22"] = src

    # --- FPFS flux (task kernel) ---
    col, src = _extract_preferred(
        det_cat, det_names, fpfs_out, fpfs_names,
        ["fpfs_m0", "m00", "flux_gauss0"],
        ["fpfs1_m00", "fpfs_m00", "m00", "flux_gauss0"])
    merged["fpfs_m00"] = _fill_nan(col, n_det)
    provenance["fpfs_m00"] = src

    col, src = _extract_preferred(
        det_cat, det_names, fpfs_out, fpfs_names,
        ["fpfs_dm0_dg1", "dm00_dg1", "dflux_gauss0_dg1"],
        ["fpfs1_dm00_dg1", "fpfs_dm00_dg1", "dm00_dg1", "dflux_gauss0_dg1"])
    merged["fpfs_dm00_dg1"] = _fill_nan(col, n_det)
    provenance["fpfs_dm00_dg1"] = src

    col, src = _extract_preferred(
        det_cat, det_names, fpfs_out, fpfs_names,
        ["fpfs_dm0_dg2", "dm00_dg2", "dflux_gauss0_dg2"],
        ["fpfs1_dm00_dg2", "fpfs_dm00_dg2", "dm00_dg2", "dflux_gauss0_dg2"])
    merged["fpfs_dm00_dg2"] = _fill_nan(col, n_det)
    provenance["fpfs_dm00_dg2"] = src

    # ---- Concise measurement log: which source each FPFS column came from ----
    _log_fpfs_provenance(provenance, n_det)

    return merged


def _fill_nan(col: Optional[np.ndarray], n: int) -> np.ndarray:
    """Return ``col`` as float64, or an all-NaN array of length ``n`` if None."""
    if col is None:
        return np.full(n, np.nan, dtype=np.float64)
    return np.asarray(col, dtype=np.float64)


def _column_1d(col: np.ndarray) -> np.ndarray:
    """Return a structured-array column as a 1-D float64 array."""
    if col.ndim > 1:
        col = col.reshape(col.shape[0], -1)
        if col.shape[1] == 1:
            col = col[:, 0]
    return np.asarray(col, dtype=np.float64)


def _extract_column(
    out: np.ndarray, available_names: list, candidates: list
) -> Optional[np.ndarray]:
    """Try to extract a column by checking multiple possible names."""
    for name in candidates:
        if name in available_names:
            return _column_1d(out[name])
    return None


def _check_task_fpfs_columns(det_names: list) -> None:
    """Fail fast if the Task's task-kernel FPFS columns are missing.

    The production FPFS estimator (``fpfs_e{1,2}`` / ``fpfs_de{1,2}_dg{1,2}`` /
    ``fpfs_m0`` / ``wsel``) is measured by the Task's kernel at
    ``sigma_arcsec``; the forced ``fpfs1_*`` columns are measured at a
    DIFFERENT shapelet scale (``fpfs_config.sigma_shapelets1``).  Silently
    substituting them would produce an inconsistent catalog (e.g. shapes at
    sigma=0.45 with a selection weight at sigma=0.40), so raise here instead
    of falling back when the Task columns are absent (anacal schema change,
    ``do_measure``/``do_fpfs`` turned off, ...).
    """
    required = [
        "fpfs_e1", "fpfs_e2",
        "fpfs_de1_dg1", "fpfs_de2_dg2",
        "fpfs_m0", "fpfs_dm0_dg1", "fpfs_dm0_dg2",
        "wsel", "dwsel_dg1", "dwsel_dg2",
    ]
    missing = [n for n in required if n not in det_names]
    if missing:
        raise RuntimeError(
            "[fpfs] Task output is missing task-kernel FPFS columns: "
            f"{missing}. Available columns: {sorted(det_names)}. Refusing to "
            "fall back to fpfs1_* (different shapelet scale, inconsistent "
            "with wsel)."
        )


def _extract_preferred(
    det_cat: np.ndarray,
    det_names: list,
    fpfs_out: np.ndarray,
    fpfs_names: list,
    det_candidates: list,
    fpfs_candidates: list,
) -> Tuple[Optional[np.ndarray], Tuple[Optional[str], Optional[str]]]:
    """Extract a column, preferring the Task's task-kernel FPFS measurement.

    The production FPFS estimator -- the one that is self-consistent with
    the selection weight ``wsel``/``dwsel`` (see example_fpfs_blended.ipynb)
    -- is the task-kernel measurement carried in ``det_cat``
    (``fpfs_e1``/``fpfs_de1_dg1``/``fpfs_m0`` ...).  The forced ``fpfs1_*``
    columns in ``fpfs_out`` are only a defensive fallback; the consistency
    guard ``_check_task_fpfs_columns`` raises before it can be reached.

    Returns
    -------
    col : ndarray or None
        The extracted column, or ``None`` if not found in either catalog.
    source : (str, str) or (None, None)
        ``("task", actual_name)`` when taken from ``det_cat``,
        ``("fpfs1", actual_name)`` when fallen back to ``fpfs_out``, or
        ``(None, None)`` when the column was not found anywhere.
    """
    for name in det_candidates:
        if name in det_names:
            return _column_1d(det_cat[name]), ("task", name)
    for name in fpfs_candidates:
        if name in fpfs_names:
            return _column_1d(fpfs_out[name]), ("fpfs1", name)
    return None, (None, None)


def _log_fpfs_provenance(provenance: dict, n_det: int) -> None:
    """Print, once per measurement, the source of each FPFS column.

    One concise line of ``field<-source.column`` entries, e.g.
    ``fpfs_e1<-task.fpfs_e1``.  Warns if any column had to fall back to the
    forced ``fpfs1_*`` measurement (a different shapelet scale than the task
    kernel).
    """
    tags = []
    for field, (source, name) in provenance.items():
        if source is None:
            tags.append(f"{field}<-MISSING")
        else:
            tags.append(f"{field}<-{source}.{name}")
    print(f"[fpfs] n={n_det} sources: " + " ".join(tags))
    if any(source == "fpfs1" for source, _ in provenance.values()):
        print(
            "[fpfs] WARNING: some columns fell back to fpfs1_* (measured at "
            "sigma_shapelets1, a different scale than the task kernel at "
            "sigma_arcsec) -- inconsistent with wsel/dwsel."
        )


def _empty_det_cat() -> np.ndarray:
    """Return an empty detection catalog with the correct dtype."""
    dtype = np.dtype([
        ("det_id", np.int32),
        ("y", np.float64), ("x", np.float64),
        ("fpfs_e1", np.float64), ("fpfs_e2", np.float64),
        ("fpfs_R11", np.float64), ("fpfs_R22", np.float64),
        ("fpfs_w", np.float64),
        ("fpfs_dw_dg1", np.float64), ("fpfs_dw_dg2", np.float64),
        ("fpfs_m00", np.float64),
        ("fpfs_dm00_dg1", np.float64), ("fpfs_dm00_dg2", np.float64),
    ])
    return np.empty(0, dtype=dtype)


# ===========================================================================
# Phase 3: Stamp Cutting
# ===========================================================================

# Pre-computed arange for stamp cutting (stamp_size = 64 constant)
_CUT_DY = np.arange(64, dtype=np.int32)
_CUT_DX = np.arange(64, dtype=np.int32)


def cut_stamps_from_large(
    large_image: np.ndarray,
    det_cat: np.ndarray,
    stamp_size: int = 64,
    stamps_out: Optional[np.ndarray] = None,
    yy_buf: Optional[np.ndarray] = None,
    xx_buf: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cut postage stamps from the large image at each detection position.

    Vectorized version using numpy broadcasting advanced indexing.
    Unlike the original loop-based version, this extracts all stamps
    in a single C-level operation (~50-100x faster).

    Parameters
    ----------
    large_image : ndarray (H, W)
        The stitched large image.
    det_cat : ndarray
        Structured detection catalog with at least 'y' and 'x' columns.
    stamp_size : int
        Size of each square stamp in pixels.
    stamps_out : ndarray or None
        Optional pre-allocated output of shape (N_det, stamp_size, stamp_size).
        Avoids internal allocation.
    yy_buf, xx_buf : ndarray or None
        Optional pre-allocated index buffers of shape (N_det, stamp_size, stamp_size).
        Pass the **same** buffers for signal + noise calls to reuse them.

    Returns
    -------
    stamps : ndarray (N_det, stamp_size, stamp_size)
        Cutout stamps for each detection.
    offsets : ndarray (N_det, 2)
        Subpixel offsets (dy, dx) = (y - floor(y+0.5), x - floor(x+0.5)),
        i.e. the galaxy's subpixel phase relative to the rounded detection
        pixel.
    """
    half = stamp_size // 2
    n_det = len(det_cat)
    H, W = large_image.shape

    # Integer pixel center (round to nearest integer)
    yc = np.floor(det_cat["y"] + 0.5).astype(np.int32)
    xc = np.floor(det_cat["x"] + 0.5).astype(np.int32)

    # Subpixel offsets (galaxy phase relative to the rounded detection pixel)
    offsets = np.column_stack([det_cat["y"] - yc, det_cat["x"] - xc])

    # Top-left corner of each stamp
    y1 = yc - half  # (N_det,)
    x1 = xc - half  # (N_det,)

    # Build index grids using pre-allocated buffers or pre-computed aranges
    # (avoids repeated np.arange + np.clip allocations per call)
    if yy_buf is not None and xx_buf is not None:
        # Use pre-allocated buffers — fill in-place
        np.add(y1[:, None, None], _CUT_DY[None, :, None], out=yy_buf[:n_det])
        np.add(x1[:, None, None], _CUT_DX[None, None, :], out=xx_buf[:n_det])
        np.clip(yy_buf[:n_det], 0, H - 1, out=yy_buf[:n_det])
        np.clip(xx_buf[:n_det], 0, W - 1, out=xx_buf[:n_det])
        yy = yy_buf[:n_det]
        xx = xx_buf[:n_det]
    else:
        dy = np.arange(stamp_size, dtype=np.int32)
        dx = np.arange(stamp_size, dtype=np.int32)
        yy = y1[:, None, None] + dy[None, :, None]
        xx = x1[:, None, None] + dx[None, None, :]
        yy = np.clip(yy, 0, H - 1)
        xx = np.clip(xx, 0, W - 1)

    # Single vectorized extraction — no Python loop!
    if stamps_out is not None:
        np.copyto(stamps_out[:n_det], large_image[yy, xx])
        stamps = stamps_out[:n_det]
    else:
        stamps = large_image[yy, xx]  # (N_det, 64, 64)

    return stamps, offsets


# ===========================================================================
# Phase 4: Catalog Merging & Saving
# ===========================================================================

def match_detections_to_truth(
    det_cat: np.ndarray,
    positions: np.ndarray,
    truth_cat: Any = None,  # pd.DataFrame
    match_threshold: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match each detection to the nearest grid position, establishing a
    link between the detection catalog and the truth catalog.

    The truth catalog rows are in the same order as the grid positions
    (row-major). For each detection, we find the nearest grid position.
    If the distance is within ``match_threshold``, we consider it a match.

    Parameters
    ----------
    det_cat : ndarray
        Detection catalog with 'y' and 'x' columns.
    positions : ndarray (N_truth, 2)
        Grid positions (y, x) for each truth galaxy.
    truth_cat : pd.DataFrame
        Truth catalog with columns 'e1', 'e2', 'gamma1', 'gamma2', etc.
    match_threshold : float
        Maximum distance (pixels) for a valid match.

    Returns
    -------
    truth_indices : ndarray (N_det,)
        Index into truth_cat/positions for each detection.
        -1 indicates no match found.
    match_distances : ndarray (N_det,)
        Distance to matched truth object, or inf if no match.
    """
    from scipy.spatial import cKDTree

    n_det = len(det_cat)
    det_pts = np.column_stack([det_cat["y"], det_cat["x"]])

    tree = cKDTree(positions)
    distances, indices = tree.query(det_pts, k=1)

    # Mask out poor matches
    indices[distances > match_threshold] = -1
    distances[distances > match_threshold] = np.inf

    return indices.astype(np.int32), distances


def merge_and_save_catalog(
    det_cat: np.ndarray,
    ml_shapes: np.ndarray,
    ml_R: np.ndarray,
    positions: np.ndarray,
    output_dir: str,
    truth_cat: Any = None,  # pd.DataFrame
    fname: str = "catalog",
    match_threshold: float = 3.0,
    save_npz: bool = True,
    save_csv: bool = True,
    verbose: bool = True,
) -> str:
    """
    Merge FPFS detection catalog, ML shape measurements, and truth catalog
    into a unified catalog, then save as .npz and .csv.

    Parameters
    ----------
    det_cat : ndarray
        FPFS detection catalog (from ``run_detection_and_fpfs``).
    ml_shapes : ndarray (N_det, 2)
        ML-measured (e1, e2).
    ml_R : ndarray (N_det, 1, 2, 2)
        ML response matrices.
    positions : ndarray (N_truth, 2)
        Grid positions for truth matching.
    output_dir : str
        Directory to save catalog files.
    truth_cat : pd.DataFrame
        Truth catalog (from ``load_img``), same order as grid positions.
    fname : str
        Base filename (without extension).
    match_threshold : float
        Maximum pixel distance for truth matching.
    save_npz : bool
        If True, save the catalog as a compressed .npz file.
    save_csv : bool
        If True, save the catalog as a .csv file.

    Returns
    -------
    output_path : str
        Path to the saved .npz file.
    """
    import os
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    n_det = len(det_cat)
    n_ml = len(ml_shapes)

    # Match detections to truth
    truth_idx, match_dist = match_detections_to_truth(
        det_cat, positions, truth_cat, match_threshold=match_threshold
    )

    # ---- Build merged catalog ----
    # Use the smaller of n_det and n_ml (they should be equal)
    n_merged = min(n_det, n_ml)

    # Extract truth values for matched detections
    truth_e1 = np.full(n_merged, np.nan, dtype=np.float64)
    truth_e2 = np.full(n_merged, np.nan, dtype=np.float64)
    truth_g1 = np.full(n_merged, np.nan, dtype=np.float64)
    truth_g2 = np.full(n_merged, np.nan, dtype=np.float64)
    truth_e1_weighted = np.full(n_merged, np.nan, dtype=np.float64)
    truth_e2_weighted = np.full(n_merged, np.nan, dtype=np.float64)

    matched_mask = truth_idx[:n_merged] >= 0
    matched_tidx = truth_idx[:n_merged][matched_mask]

    if np.any(matched_mask):
        truth_e1[matched_mask] = truth_cat["e1"].iloc[matched_tidx].values
        truth_e2[matched_mask] = truth_cat["e2"].iloc[matched_tidx].values
        truth_g1[matched_mask] = truth_cat["gamma1"].iloc[matched_tidx].values
        truth_g2[matched_mask] = truth_cat["gamma2"].iloc[matched_tidx].values
        # Component-weighted truth shape (bulge+disk), if available
        if "truth_e1_weighted" in truth_cat.columns:
            truth_e1_weighted[matched_mask] = truth_cat["truth_e1_weighted"].iloc[matched_tidx].values
        if "truth_e2_weighted" in truth_cat.columns:
            truth_e2_weighted[matched_mask] = truth_cat["truth_e2_weighted"].iloc[matched_tidx].values

    # ML response: squeeze the (N,1,2,2) to extract diagonal elements
    ml_R_squeezed = ml_R[:n_merged].reshape(n_merged, 2, 2)
    ml_R11 = ml_R_squeezed[:, 0, 0]
    ml_R22 = ml_R_squeezed[:, 1, 1]

    # Build structured array
    dtype = np.dtype([
        # Detection info
        ("det_id", np.int32),
        ("y", np.float64), ("x", np.float64),
        # FPFS shape
        ("fpfs_e1", np.float64), ("fpfs_e2", np.float64),
        ("fpfs_R11", np.float64), ("fpfs_R22", np.float64),
        # FPFS detection weights
        ("fpfs_w", np.float64),
        ("fpfs_dw_dg1", np.float64), ("fpfs_dw_dg2", np.float64),
        # FPFS flux
        ("fpfs_m00", np.float64),
        ("fpfs_dm00_dg1", np.float64), ("fpfs_dm00_dg2", np.float64),
        # ML shape
        ("ml_e1", np.float64), ("ml_e2", np.float64),
        ("ml_R11", np.float64), ("ml_R22", np.float64),
        # Truth (matched)
        ("truth_e1", np.float64), ("truth_e2", np.float64),
        ("truth_g1", np.float64), ("truth_g2", np.float64),
        # Truth (component-weighted, bulge+disk)
        ("truth_e1_weighted", np.float64), ("truth_e2_weighted", np.float64),
        # Matching info
        ("match_dist", np.float64), ("match_idx", np.int32),
    ])

    merged = np.empty(n_merged, dtype=dtype)
    merged["det_id"] = np.arange(n_merged, dtype=np.int32)
    merged["y"] = det_cat["y"][:n_merged]
    merged["x"] = det_cat["x"][:n_merged]
    merged["fpfs_e1"] = det_cat["fpfs_e1"][:n_merged]
    merged["fpfs_e2"] = det_cat["fpfs_e2"][:n_merged]
    merged["fpfs_R11"] = det_cat["fpfs_R11"][:n_merged]
    merged["fpfs_R22"] = det_cat["fpfs_R22"][:n_merged]
    merged["fpfs_w"] = det_cat["fpfs_w"][:n_merged]
    merged["fpfs_dw_dg1"] = det_cat["fpfs_dw_dg1"][:n_merged]
    merged["fpfs_dw_dg2"] = det_cat["fpfs_dw_dg2"][:n_merged]
    merged["fpfs_m00"] = det_cat["fpfs_m00"][:n_merged]
    merged["fpfs_dm00_dg1"] = det_cat["fpfs_dm00_dg1"][:n_merged]
    merged["fpfs_dm00_dg2"] = det_cat["fpfs_dm00_dg2"][:n_merged]
    merged["ml_e1"] = ml_shapes[:n_merged, 0]
    merged["ml_e2"] = ml_shapes[:n_merged, 1]
    merged["ml_R11"] = ml_R11
    merged["ml_R22"] = ml_R22
    merged["truth_e1"] = truth_e1
    merged["truth_e2"] = truth_e2
    merged["truth_g1"] = truth_g1
    merged["truth_g2"] = truth_g2
    merged["truth_e1_weighted"] = truth_e1_weighted
    merged["truth_e2_weighted"] = truth_e2_weighted
    merged["match_dist"] = match_dist[:n_merged]
    merged["match_idx"] = truth_idx[:n_merged]

    # ---- Save .npz ----
    npz_path = os.path.join(output_dir, f"{fname}.npz")
    if save_npz:
        np.savez_compressed(
            npz_path,
            det_id=merged["det_id"],
            y=merged["y"], x=merged["x"],
            fpfs_e1=merged["fpfs_e1"], fpfs_e2=merged["fpfs_e2"],
            fpfs_R11=merged["fpfs_R11"], fpfs_R22=merged["fpfs_R22"],
            fpfs_w=merged["fpfs_w"],
            fpfs_dw_dg1=merged["fpfs_dw_dg1"], fpfs_dw_dg2=merged["fpfs_dw_dg2"],
            fpfs_m00=merged["fpfs_m00"],
            fpfs_dm00_dg1=merged["fpfs_dm00_dg1"], fpfs_dm00_dg2=merged["fpfs_dm00_dg2"],
            ml_e1=merged["ml_e1"], ml_e2=merged["ml_e2"],
            ml_R11=merged["ml_R11"], ml_R22=merged["ml_R22"],
            truth_e1=merged["truth_e1"], truth_e2=merged["truth_e2"],
            truth_g1=merged["truth_g1"], truth_g2=merged["truth_g2"],
            truth_e1_weighted=merged["truth_e1_weighted"], truth_e2_weighted=merged["truth_e2_weighted"],
            match_dist=merged["match_dist"], match_idx=merged["match_idx"],
        )

    # ---- Save .csv (scalar columns only) ----
    csv_path = os.path.join(output_dir, f"{fname}.csv")
    if save_csv:
        df = pd.DataFrame(merged)
        df.to_csv(csv_path, index=False)

    # ---- Summary ----
    n_matched = np.sum(matched_mask)
    if verbose:
        saved_parts = []
        if save_npz:
            saved_parts.append(npz_path)
        if save_csv:
            saved_parts.append(csv_path)
        print(f"[merge] Catalog saved to: " + ", ".join(saved_parts))
        print(f"  Total detections: {n_merged}")
        print(f"  Matched to truth: {n_matched} ({100*n_matched/max(n_merged,1):.1f}%)")
        print(f"  Unmatched: {n_merged - n_matched}")
        print(f"  Save npz: {save_npz}, csv: {save_csv}")

    return npz_path
