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
) -> np.ndarray:
    """
    Run anacal FPFS detection on a large image (multiple galaxies).

    Calls ``anacal.fpfs.process_image`` with ``detection=None`` to trigger
    internal peak-finding, and with ``do_compute_detect_weight=True`` to
    obtain detection weights and their shear derivatives.

    Parameters
    ----------
    large_image : ndarray (H, W)
        The stitched large image with galaxies placed on a grid.
    psf_image : ndarray (H_psf, W_psf)
        PSF stamp image.
    fpfs_config : anacal.fpfs.FpfsConfig
        FPFS configuration (sigma_shapelets, etc.).
    mag_zero : float
        Magnitude zero point.
    pixel_scale : float
        Pixel scale in arcsec/pixel.
    noise_variance : float
        Noise variance (sigma^2) for the image.
    noise_array : ndarray or None
        Optional noise realization. If None, anacal estimates internally.

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

    # Run detection: detection=None triggers internal peak-finding
    out = anacal.fpfs.process_image(
        fpfs_config=fpfs_config,
        mag_zero=mag_zero,
        gal_array=gal_array,
        psf_array=psf_array,
        pixel_scale=pixel_scale,
        noise_variance=noise_variance,
        noise_array=noise_array,
        detection=None,                # <-- auto-detect
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
    pos_y = _extract_column(out, names, ["fpfs_y", "det_y", "y", "fpfs_row"])
    pos_x = _extract_column(out, names, ["fpfs_x", "det_x", "x", "fpfs_col"])

    # --- Extract FPFS shape measurements ---
    fpfs_e1 = _extract_column(out, names, ["fpfs1_e1", "fpfs_e1", "e1"])
    fpfs_e2 = _extract_column(out, names, ["fpfs1_e2", "fpfs_e2", "e2"])
    fpfs_R11 = _extract_column(out, names, ["fpfs1_de1_dg1", "fpfs_de1_dg1", "de1_dg1", "R11"])
    fpfs_R22 = _extract_column(out, names, ["fpfs1_de2_dg2", "fpfs_de2_dg2", "de2_dg2", "R22"])

    # --- Extract detection weights ---
    fpfs_w = _extract_column(out, names, ["fpfs_w", "det_weight", "w"])
    fpfs_dw_dg1 = _extract_column(out, names, ["fpfs_dw_dg1", "dw_dg1"])
    fpfs_dw_dg2 = _extract_column(out, names, ["fpfs_dw_dg2", "dw_dg2"])

    # --- Extract flux ---
    fpfs_m00 = _extract_column(out, names, ["fpfs1_m00", "fpfs_m00", "m00", "flux"])
    fpfs_dm00_dg1 = _extract_column(out, names, ["fpfs1_dm00_dg1", "fpfs_dm00_dg1", "dm00_dg1"])
    fpfs_dm00_dg2 = _extract_column(out, names, ["fpfs1_dm00_dg2", "fpfs_dm00_dg2", "dm00_dg2"])

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

def cut_stamps_from_large(
    large_image: np.ndarray,
    det_cat: np.ndarray,
    stamp_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cut postage stamps from the large image at each detection position.

    Parameters
    ----------
    large_image : ndarray (H, W)
        The stitched large image.
    det_cat : ndarray
        Structured detection catalog with at least 'y' and 'x' columns.
    stamp_size : int
        Size of each square stamp in pixels.

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
    canvas_h, canvas_w = large_image.shape

    stamps = np.zeros((n_det, stamp_size, stamp_size), dtype=large_image.dtype)
    offsets = np.zeros((n_det, 2), dtype=np.float64)

    for i in range(n_det):
        y = det_cat["y"][i]
        x = det_cat["x"][i]

        # Integer pixel center (round to nearest integer for slicing)
        yc = int(np.floor(y + 0.5))
        xc = int(np.floor(x + 0.5))

        # Subpixel offset
        offsets[i, 0] = y - yc
        offsets[i, 1] = x - xc

        y1 = yc - half
        y2 = yc + half
        x1 = xc - half
        x2 = xc + half

        # Clamp to canvas bounds; pad with zeros if outside
        cy1, cy2 = max(y1, 0), min(y2, canvas_h)
        cx1, cx2 = max(x1, 0), min(x2, canvas_w)

        if cy2 > cy1 and cx2 > cx1:
            # Place the valid region into the stamp
            sy1 = cy1 - y1
            sy2 = stamp_size - (y2 - cy2)
            sx1 = cx1 - x1
            sx2 = stamp_size - (x2 - cx2)
            stamps[i, sy1:sy2, sx1:sx2] = large_image[cy1:cy2, cx1:cx2]

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
