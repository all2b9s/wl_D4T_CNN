"""Detection helpers derived from the existing blended pipeline."""
import numpy as np
import pandas as pd
from src.anacal.detection_ml_pipeline import _empty_det_cat, match_detections_to_truth

_FAKE_DET_DTYPE = np.dtype([
    ("det_id", np.int32),
    ("y", np.float64), ("x", np.float64),
    ("fpfs_e1", np.float64), ("fpfs_e2", np.float64),
    ("fpfs_R11", np.float64), ("fpfs_R22", np.float64),
    ("fpfs_w", np.float64),
    ("fpfs_dw_dg1", np.float64), ("fpfs_dw_dg2", np.float64),
    ("fpfs_m00", np.float64),
    ("fpfs_dm00_dg1", np.float64), ("fpfs_dm00_dg2", np.float64),
])


def _build_truth_detection_catalog(truth_df: pd.DataFrame, mag_cut: float | None) -> np.ndarray:
    """Create calibration-compatible detections from truth positions for test mode."""
    if mag_cut is not None and "i_ab" in truth_df.columns:
        truth_df = truth_df.loc[truth_df["i_ab"].to_numpy(dtype=float) < mag_cut]

    det_cat = np.zeros(len(truth_df), dtype=_FAKE_DET_DTYPE)
    det_cat["det_id"] = np.arange(len(truth_df), dtype=np.int32)
    det_cat["y"] = truth_df["y"].to_numpy(dtype=np.float64)
    det_cat["x"] = truth_df["x"].to_numpy(dtype=np.float64)
    det_cat["fpfs_w"] = 1.0
    return det_cat


def _recenter_on_truth(
    det_cat: np.ndarray,
    positions: np.ndarray,
    match_threshold: float,
) -> np.ndarray:
    """Crossmatch test: re-center detections on the matched truth positions.

    Detections without a truth match (farther than ``match_threshold``
    pixels) are DROPPED; the remaining detections keep their weights but
    their stamp cut centres move to the matched truth coordinates.
    """
    if len(det_cat) == 0:
        return det_cat

    truth_idx, _ = match_detections_to_truth(
        det_cat, positions, match_threshold=match_threshold,
    )
    matched = truth_idx >= 0
    n_total = len(det_cat)
    n_match = int(np.sum(matched))
    det_cat = det_cat[matched]
    if n_match > 0:
        det_cat["y"] = positions[truth_idx[matched], 0]
        det_cat["x"] = positions[truth_idx[matched], 1]
    print(f"    [center_on_truth] {n_match}/{n_total} matched & re-centered on truth "
          f"(thr={match_threshold} px, dropped {n_total - n_match})")
    return det_cat
