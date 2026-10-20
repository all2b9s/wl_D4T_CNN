"""FITS helpers derived from src/blended_run.py; kept torch-free here."""
import numpy as np
import pandas as pd
from astropy.io import fits
from src.datasets.dataset_toolkit import get_weighted_e

def native_column(array):
    """Convert FITS arrays to native byte order for pandas."""
    array = np.asarray(array)
    array = array.astype(array.dtype.newbyteorder("="), copy=False)
    return array if array.ndim == 1 else list(array)


def load_blended_exposure(
    fits_path: str,
    truth_path: str,
    cat_ref: np.ndarray,
    pixel_scale: float = 0.2,
    noise_mode: str = "from_variance",
    fixed_noise: float = None,
):
    """
    Load a DC1 blended simulation FITS exposure and truth catalog.

    Parameters
    ----------
    fits_path : str
        Path to ``exp-i-NNNNN.fits`` (HDU[1]=image, HDU[3]=variance_map).
    truth_path : str
        Path to ``truth-NNNNN.fits`` (HDU[1]=truth table).
    cat_ref : ndarray
        Reference catalog (OneDegSq.fits) for i_ab, hlr, etc.
    pixel_scale : float
        Pixel scale in arcsec/pixel.
    noise_mode : str
        ``"from_variance"`` — estimate sigma from variance_map median.
        ``"fixed"`` — use ``fixed_noise`` value.
    fixed_noise : float or None
        Fixed noise sigma when ``noise_mode="fixed"``.

    Returns
    -------
    large_image : ndarray (H, W)
        Science image (float64).
    noise_sigma : float
        Estimated or fixed noise standard deviation.
    truth_df : pd.DataFrame
        Truth catalog filtered to objects inside the image.
        Contains ``y``, ``x``, ``e1``, ``e2``, ``gamma1``, ``gamma2``,
        ``i_ab``, ``hlr``, etc.
    positions : ndarray (N, 2)
        Truth (y, x) positions for matching.
    """
    # ---- Load exposure ----
    with fits.open(fits_path, memmap=False) as hdul:
        large_image = np.asarray(hdul[1].data, dtype=np.float64)
        variance_map = np.asarray(hdul[3].data, dtype=np.float64)

    ny, nx = large_image.shape

    # ---- Load truth ----
    with fits.open(truth_path, memmap=False) as hdul:
        truth_raw = hdul[1].data.copy()

    truth_df = pd.DataFrame({
        name: native_column(truth_raw[name])
        for name in truth_raw.dtype.names
    })

    # ---- Convert dx/dy to image pixels (fixed offset: nx/2 + dx/0.2 + 1500) ----
    x_all = nx / 2 + truth_df["dx"].to_numpy(dtype=float) / pixel_scale + 1500
    y_all = ny / 2 + truth_df["dy"].to_numpy(dtype=float) / pixel_scale + 1500

    inside = (
        (x_all >= 32) & (x_all < nx-32)
        & (y_all >= 32) & (y_all < ny-32)
    )

    cat = truth_df.loc[inside].reset_index(drop=True)
    cat["x"] = x_all[inside]
    cat["y"] = y_all[inside]
    cat["e1"] = cat["gamma1"]  # alias for merge_and_save_catalog
    cat["e2"] = cat["gamma2"]

    # ---- Match cat_ref (i_ab, hlr, etc.) via truth indices ----
    ref_indices = cat["indices"].to_numpy(dtype=np.int64)

    if len(ref_indices) > 0 and ref_indices.min() >= 0 and ref_indices.max() < len(cat_ref):
        cat_ref_selected = cat_ref[ref_indices]
        for name in cat_ref_selected.dtype.names:
            output_name = name if name not in cat.columns else f"{name}_ref"
            cat[output_name] = native_column(cat_ref_selected[name])

    # ---- Compute component-weighted truth shape (bulge+disk) ----
    _weighted_cols = ["a_b", "b_b", "pa_bulge", "fluxnorm_bulge",
                      "a_d", "b_d", "pa_disk", "fluxnorm_disk"]
    if all(c in cat.columns for c in _weighted_cols) and "angles" in cat.columns:
        _we = get_weighted_e(
            cat["a_b"].to_numpy(dtype=float),
            cat["b_b"].to_numpy(dtype=float),
            cat["pa_bulge"].to_numpy(dtype=float),
            cat["fluxnorm_bulge"].to_numpy(dtype=float),
            cat["a_d"].to_numpy(dtype=float),
            cat["b_d"].to_numpy(dtype=float),
            cat["pa_disk"].to_numpy(dtype=float),
            cat["fluxnorm_disk"].to_numpy(dtype=float),
            cat["angles"].to_numpy(dtype=float),
        )
        cat["truth_e1_weighted"] = _we[0]
        cat["truth_e2_weighted"] = _we[1]

    positions = cat[["y", "x"]].to_numpy(dtype=np.float64)

    # ---- Noise sigma ----
    if noise_mode == "from_variance":
        valid = variance_map[np.isfinite(variance_map) & (variance_map > 0)]
        noise_sigma = float(np.sqrt(np.median(valid))) if len(valid) > 0 else 0.3
    else:
        noise_sigma = float(fixed_noise) if fixed_noise is not None else 0.3

    return large_image, noise_sigma, cat, positions
