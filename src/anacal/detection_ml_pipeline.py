"""
FPFS Detection + ML Shape Measurement Pipeline
==============================================

Phase 1: Stitch isolated galaxy cutouts into large images
Phase 2: Run anacal FPFS detection on the large image
Phase 3: Cut postage stamps at detected positions & run ML shape measurement
Phase 4: Merge anacal detection info (w/dw, m00/dm00, FPFS shape) with ML results (shape/R)

Reuses heavily from:
    - src/anacal/cal_toolkit.py   (load_img, build_dataset, format_number)
    - src/anacal/batch_calibration.py (Calibrator, prepare_q_images)
    - src/anacal/fpfs.py          (FpfsConfig, process_image reference)
"""

import numpy as np
import os
from typing import List, Tuple, Optional, Any

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
# Internal helpers: Task API adapter layer
# ===========================================================================

def _build_task_from_fpfs_config(
    fpfs_config: Any = None,
    mag_zero: float = 30.0,
    pixel_scale: float = 0.2,
    stamp_size: int = 64,
    sigma_arcsec: float = 0.40,
):
    """Build an ``anacal.task.Task`` from legacy ``FpfsConfig`` parameters.

    Uses the same defaults as ``xlens.processor.anacal.AnacalTask``
    (see ``example_fpfs_blended_new.ipynb``).  The flux-related
    thresholds are scaled by ``ratio = 10**((mag_zero - 30) / 2.5)``.

    Parameters
    ----------
    fpfs_config : anacal.fpfs.FpfsConfig or None
        Legacy configuration.  Currently unused for Task construction;
        ``sigma_arcsec`` must be set explicitly (default 0.40 arcsec
        matches the xlens / example notebook value).
    mag_zero : float
        Photometric zero point.
    pixel_scale : float
        Pixel scale in arcsec / pixel.
    stamp_size : int
        PSF / measurement stamp size in pixels.
    sigma_arcsec : float
        Detection smoothing kernel size in arcsec (default 0.40).
        Controls peak-finding SNR; larger values smooth more noise
        but may blend close sources.

    Returns
    -------
    task : anacal.task.Task
    """
    import anacal

    ratio = 10.0 ** ((mag_zero - 30.0) / 2.5)

    prior = anacal.ngmix.modelPrior()
    prior.set_sigma_a(anacal.math.qnumber(0.05))
    prior.set_sigma_x(anacal.math.qnumber(0.05))

    task = anacal.task.Task(
        scale=pixel_scale,
        sigma_arcsec=sigma_arcsec,
        snr_peak_min=5.0,
        omega_f=0.06 * ratio,
        v_min=0.013 * ratio,
        omega_v=0.025 * ratio,
        p_min=0.12,
        omega_p=0.05,
        prior=prior,
        stamp_size=stamp_size,
        image_bound=40,
        num_epochs=0,
        force_size=False,
        force_center=True,
        fpfs_c0=8.4 * ratio,
    )
    return task


def _build_block_list(
    img_height: int,
    img_width: int,
    pixel_scale: float,
    psf_array: np.ndarray,
    stamp_size: int = 64,
) -> list:
    """Build processing blocks for ``task.process_image``.

    Tiles the image with ``anacal.geometry.get_block_list`` and attaches
    the (constant) PSF stamp to every block.

    Parameters
    ----------
    img_height, img_width : int
        Image dimensions in pixels.
    pixel_scale : float
        Pixel scale in arcsec / pixel.
    psf_array : ndarray
        PSF stamp image.
    stamp_size : int
        Target PSF stamp size.

    Returns
    -------
    blocks : list of ``anacal.geometry.Block``
    """
    import anacal

    blocks = anacal.geometry.get_block_list(
        img_nx=img_width,
        img_ny=img_height,
        block_nx=img_width//2,
        block_ny=img_height//2,
        block_overlap=80,
        scale=pixel_scale,
    )
    # Attach the same constant PSF to every block
    for bb in blocks:
        bb.psf_array = anacal.utils.resize_array(psf_array, (stamp_size, stamp_size))

    return blocks


def _adapt_task_output_to_legacy(
    out: np.ndarray,
    pixel_scale: float,
) -> np.ndarray:
    """Convert ``task.process_image`` output columns to legacy-compatible names.

    Mapping::

        Task API column               Legacy column
        ─────────────────────────────────────────────────
        x1, x2     (arcsec)        →  fpfs_x, fpfs_y  (pixels)
        wsel                         →  fpfs_w
        dwsel_dg1, dwsel_dg2        →  fpfs_dw_dg1, fpfs_dw_dg2
        flux_gauss0                  →  fpfs_m00
        dflux_gauss0_dg{1,2}        →  fpfs_dm00_dg{1,2}
        fpfs_e1, fpfs_e2             →  fpfs_e1, fpfs_e2         (unchanged)
        fpfs_de1_dg1, fpfs_de2_dg2   →  fpfs_de1_dg1, fpfs_de2_dg2 (unchanged)

    Parameters
    ----------
    out : ndarray
        Structured array returned by ``task.process_image``.
    pixel_scale : float
        Pixel scale for arcsec → pixel conversion.

    Returns
    -------
    adapted : ndarray
        Structured array with legacy-compatible column names.
    """
    n_det = len(out)
    names = list(out.dtype.names)

    new_dtype = np.dtype([
        ("fpfs_y", np.float64),
        ("fpfs_x", np.float64),
        ("fpfs_e1", np.float64),
        ("fpfs_e2", np.float64),
        ("fpfs_de1_dg1", np.float64),
        ("fpfs_de2_dg2", np.float64),
        ("fpfs_w", np.float64),
        ("fpfs_dw_dg1", np.float64),
        ("fpfs_dw_dg2", np.float64),
        ("fpfs_m00", np.float64),
        ("fpfs_dm00_dg1", np.float64),
        ("fpfs_dm00_dg2", np.float64),
    ])

    adapted = np.empty(n_det, dtype=new_dtype)

    # --- positions: x2 → fpfs_y, x1 → fpfs_x  (arcsec → pixels) ---
    adapted["fpfs_y"] = out["x2"] / pixel_scale if "x2" in names else np.nan
    adapted["fpfs_x"] = out["x1"] / pixel_scale if "x1" in names else np.nan

    # --- shapes (same column names in both APIs) ---
    for col in ["fpfs_e1", "fpfs_e2", "fpfs_de1_dg1", "fpfs_de2_dg2"]:
        adapted[col] = out[col] if col in names else np.nan

    # --- selection weights (wsel → fpfs_w) ---
    adapted["fpfs_w"] = out["wsel"] if "wsel" in names else np.nan
    adapted["fpfs_dw_dg1"] = out["dwsel_dg1"] if "dwsel_dg1" in names else np.nan
    adapted["fpfs_dw_dg2"] = out["dwsel_dg2"] if "dwsel_dg2" in names else np.nan

    # --- flux (flux_gauss0 → fpfs_m00) ---
    adapted["fpfs_m00"] = out["flux_gauss0"] if "flux_gauss0" in names else np.nan
    adapted["fpfs_dm00_dg1"] = (
        out["dflux_gauss0_dg1"] if "dflux_gauss0_dg1" in names else np.nan
    )
    adapted["fpfs_dm00_dg2"] = (
        out["dflux_gauss0_dg2"] if "dflux_gauss0_dg2" in names else np.nan
    )

    return adapted


# ===========================================================================
# Phase 2: Anacal FPFS Detection
# ===========================================================================

def run_fpfs_detection(
    large_image: np.ndarray,
    psf_image: np.ndarray,
    fpfs_config: Any = None,  # anacal.fpfs.FpfsConfig
    mag_zero: float = 30.0,
    pixel_scale: float = 0.2,
    noise_variance: float = 1e-4,
    noise_array: Optional[np.ndarray] = None,
    use_task_api: bool = True,
    stamp_size: int = 64,
    sigma_arcsec: float = 0.40,
    task: Any = None,  # cached anacal.task.Task
    blocks: Any = None,  # cached block list
) -> np.ndarray:
    """
    Run anacal FPFS detection on a large image (multiple galaxies).

    By default uses the new ``anacal.task.Task`` API (``use_task_api=True``),
    which internally handles detection + FPFS measurement in a single call.
    Set ``use_task_api=False`` to fall back to the legacy
    ``anacal.fpfs.process_image`` path.

    Parameters
    ----------
    large_image : ndarray (H, W)
        The stitched large image with galaxies placed on a grid.
    psf_image : ndarray (H_psf, W_psf)
        PSF stamp image.
    fpfs_config : anacal.fpfs.FpfsConfig
        FPFS configuration.  When ``use_task_api=True``, only used for
        legacy fallback; Task parameters are set independently.
    mag_zero : float
        Magnitude zero point.
    pixel_scale : float
        Pixel scale in arcsec/pixel.
    noise_variance : float
        Noise variance (sigma^2) for the image.
    noise_array : ndarray or None
        Optional noise realization.
    use_task_api : bool
        If True (default), use ``anacal.task.Task``.  If False, use
        legacy ``anacal.fpfs.process_image``.
    stamp_size : int
        PSF / measurement stamp size in pixels (Task API only).
    sigma_arcsec : float
        Detection smoothing kernel in arcsec (Task API only, default 0.40).
        Larger values improve peak SNR on noisy images but may blend
        close sources.  This is NOT derived from ``fpfs_config``.
    task : anacal.task.Task or None
        Cached Task object. If None, a new one is built.
    blocks : list of anacal.geometry.Block or None
        Cached block list. If None, a new one is built from ``psf_image``.

    Returns
    -------
    det_cat : ndarray
        Structured array with one row per detected object.
        Columns include detection positions, FPFS shapes, responses,
        detection weights, and flux measurements.
    """
    import anacal

    gal_array = large_image.astype(np.float64, copy=False)
    psf_array = psf_image.astype(np.float64, copy=False)

    if noise_array is not None:
        noise_array = noise_array.astype(np.float64, copy=False)

    if use_task_api:
        # ---- New anacal.task.Task API ----
        if task is None:
            task = _build_task_from_fpfs_config(
                fpfs_config=fpfs_config,
                mag_zero=mag_zero,
                pixel_scale=pixel_scale,
                stamp_size=stamp_size,
                sigma_arcsec=sigma_arcsec,
            )
        if blocks is None:
            blocks = _build_block_list(
                img_height=gal_array.shape[0],
                img_width=gal_array.shape[1],
                pixel_scale=pixel_scale,
                psf_array=psf_array,
                stamp_size=stamp_size,
            )
        out = task.process_image(
            gal_array,
            psf_array,
            variance=noise_variance,
            block_list=blocks,
            detection=None,
            noise_array=noise_array,
            mask_array=None,
            do_fpfs=True,
        )
        # Convert arcsec positions → pixels, wsel → fpfs_w, etc.
        out = _adapt_task_output_to_legacy(out, pixel_scale)

    else:
        # ---- Legacy anacal.fpfs.process_image API ----
        out = anacal.fpfs.process_image(
            fpfs_config=fpfs_config,
            mag_zero=mag_zero,
            gal_array=gal_array,
            psf_array=psf_array,
            pixel_scale=pixel_scale,
            noise_variance=noise_variance,
            noise_array=noise_array,
            detection=None,
            do_compute_detect_weight=True,
        )

    return _parse_detection_output(out)


def _parse_detection_output(out: np.ndarray) -> np.ndarray:
    """
    Parse the structured array returned by ``anacal.fpfs.process_image``
    into a standardized detection catalog.

    Handles the case of empty results (no detections) gracefully.
    """
    if out is None:
        return _empty_det_cat()

    # Handle both structured arrays and recarrays
    if hasattr(out, "dtype") and hasattr(out.dtype, "names") and out.dtype.names is not None:
        names = list(out.dtype.names)
    else:
        if hasattr(out, "shape") and out.shape[0] == 0:
            return _empty_det_cat()
        names = []

    n_det = len(out) if hasattr(out, "__len__") else 0
    if n_det == 0:
        return _empty_det_cat()

    # --- Extract positions ---
    pos_y = _extract_column(out, names, ["fpfs_y", "det_y", "y", "fpfs_row", "x2"])
    pos_x = _extract_column(out, names, ["fpfs_x", "det_x", "x", "fpfs_col", "x1"])

    # --- Extract FPFS shape measurements ---
    fpfs_e1 = _extract_column(out, names, ["fpfs1_e1", "fpfs_e1", "e1"])
    fpfs_e2 = _extract_column(out, names, ["fpfs1_e2", "fpfs_e2", "e2"])
    fpfs_R11 = _extract_column(out, names, ["fpfs1_de1_dg1", "fpfs_de1_dg1", "de1_dg1", "R11"])
    fpfs_R22 = _extract_column(out, names, ["fpfs1_de2_dg2", "fpfs_de2_dg2", "de2_dg2", "R22"])

    # --- Extract detection / selection weights ---
    fpfs_w = _extract_column(out, names, ["fpfs_w", "det_weight", "w", "wsel", "wdet"])
    fpfs_dw_dg1 = _extract_column(out, names, ["fpfs_dw_dg1", "dw_dg1", "dwsel_dg1", "dwdet_dg1"])
    fpfs_dw_dg2 = _extract_column(out, names, ["fpfs_dw_dg2", "dw_dg2", "dwsel_dg2", "dwdet_dg2"])

    # --- Extract flux ---
    fpfs_m00 = _extract_column(out, names, ["fpfs1_m00", "fpfs_m00", "m00", "flux", "flux_gauss0"])
    fpfs_dm00_dg1 = _extract_column(out, names, ["fpfs1_dm00_dg1", "fpfs_dm00_dg1", "dm00_dg1", "dflux_gauss0_dg1"])
    fpfs_dm00_dg2 = _extract_column(out, names, ["fpfs1_dm00_dg2", "fpfs_dm00_dg2", "dm00_dg2", "dflux_gauss0_dg2"])

    # Build a clean structured array
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

    det_cat = np.empty(n_det, dtype=dtype)
    det_cat["det_id"] = np.arange(n_det, dtype=np.int32)
    det_cat["y"] = pos_y if pos_y is not None else np.nan
    det_cat["x"] = pos_x if pos_x is not None else np.nan
    det_cat["fpfs_e1"] = fpfs_e1 if fpfs_e1 is not None else np.nan
    det_cat["fpfs_e2"] = fpfs_e2 if fpfs_e2 is not None else np.nan
    det_cat["fpfs_R11"] = fpfs_R11 if fpfs_R11 is not None else np.nan
    det_cat["fpfs_R22"] = fpfs_R22 if fpfs_R22 is not None else np.nan
    det_cat["fpfs_w"] = fpfs_w if fpfs_w is not None else np.nan
    det_cat["fpfs_dw_dg1"] = fpfs_dw_dg1 if fpfs_dw_dg1 is not None else np.nan
    det_cat["fpfs_dw_dg2"] = fpfs_dw_dg2 if fpfs_dw_dg2 is not None else np.nan
    det_cat["fpfs_m00"] = fpfs_m00 if fpfs_m00 is not None else np.nan
    det_cat["fpfs_dm00_dg1"] = fpfs_dm00_dg1 if fpfs_dm00_dg1 is not None else np.nan
    det_cat["fpfs_dm00_dg2"] = fpfs_dm00_dg2 if fpfs_dm00_dg2 is not None else np.nan

    return det_cat


def _extract_column(
    out: np.ndarray, available_names: list, candidates: list
) -> Optional[np.ndarray]:
    """Try to extract a column by checking multiple possible names."""
    for name in candidates:
        if name in available_names:
            col = out[name]
            if col.ndim > 1:
                col = col.reshape(col.shape[0], -1)
                if col.shape[1] == 1:
                    col = col[:, 0]
            return np.asarray(col, dtype=np.float64)
    return None


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
        Subpixel offsets (dy, dx) = (y - floor(y+0.5), x - floor(x+0.5)).
        These represent the difference between the detection centroid and
        the stamp center pixel, useful for subpixel corrections.
    """
    half = stamp_size // 2
    n_det = len(det_cat)
    H, W = large_image.shape

    # Integer pixel center (round to nearest integer)
    yc = np.floor(det_cat["y"] + 0.5).astype(np.int32)
    xc = np.floor(det_cat["x"] + 0.5).astype(np.int32)

    # Subpixel offsets
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
        FPFS detection catalog (from ``run_fpfs_detection``).
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

    matched_mask = truth_idx[:n_merged] >= 0
    matched_tidx = truth_idx[:n_merged][matched_mask]

    if np.any(matched_mask):
        truth_e1[matched_mask] = truth_cat["e1"].iloc[matched_tidx].values
        truth_e2[matched_mask] = truth_cat["e2"].iloc[matched_tidx].values
        truth_g1[matched_mask] = truth_cat["gamma1"].iloc[matched_tidx].values
        truth_g2[matched_mask] = truth_cat["gamma2"].iloc[matched_tidx].values

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
