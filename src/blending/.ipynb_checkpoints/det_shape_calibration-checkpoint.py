"""
Detection + Selection + Shape Calibration Pipeline
====================================================

Jointly calibrates detection response, selection response, and shape
measurement multiplicative/additive biases using NPZ outputs from
``detection_ml_pipeline.py``.

Based on ``cal_biases.py`` architecture. Supports FPFS / ML shape
estimators via ``shape_mode``.

Usage
-----
    python src/blending/det_shape_calibration.py \
        --noise 0.594 --job_name xlens_shift --fname p85

Or programmatic::

    from src.blending.det_shape_calibration import get_det_calibration_biases
    result = get_det_calibration_biases(noise_level=0.594, job_name='xlens_shift')
"""

import datetime
import numpy as np
import os
import re
import time
import argparse
from tqdm import tqdm
import multiprocessing as mp

from src.anacal.cal_toolkit import selection_response, format_number


# ===========================================================================
# Default paths (override via function parameters or env vars)
# ===========================================================================

DEFAULT_BASE_DIR = os.environ.get(
    "DET_CALI_BASE_DIR",
    "/work/hdd/bfmo/wenyinli/measurement/det_sims",
)
DEFAULT_OUTPUT_DIR = os.environ.get(
    "DET_CALI_OUTPUT_DIR",
    "/work/hdd/bfmo/wenyinli/datasets/det_cali_results",
)
DEFAULT_LOG_CSV = os.environ.get(
    "DET_CALI_LOG_CSV",
    "./logs/det_calibration_result.csv",
)


# ===========================================================================
# Path helpers
# ===========================================================================

def _resolve_det_npz_path(base_dir, shear_comp, shear_mode, fname, index):
    """Resolve an NPZ file path for one shear-condition × batch-index.

    Tries several directory layouts to accommodate different run
    configurations of ``detection_ml_pipeline`` and
    ``fpfsdet_MLshape_run.py`` (which saves as ``catalog_{index}.npz``).

    Parameters
    ----------
    base_dir : str
        Root directory, e.g. ``/.../det_sims/n0594/``.
    shear_comp : {'g1', 'g2'}
    shear_mode : {0, 1}
    fname : str
        Base filename prefix.
    index : int
        Batch index.

    Returns
    -------
    path : str or None
    """
    candidates = [
        f"{base_dir}/{shear_comp}_{shear_mode}/{fname}_{index}.npz",
        f"{base_dir}/{shear_comp}_{shear_mode}/{fname}/{fname}_{index}.npz",
        f"{base_dir}/{shear_comp}_{shear_mode}/{fname}/catalog_{index}.npz",
        f"{base_dir}/{fname}/{shear_comp}_{shear_mode}/{fname}_{index}.npz",
        f"{base_dir}/{fname}/{shear_comp}_{shear_mode}/catalog_{index}.npz",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


# ===========================================================================
# Parallel NPZ loader
# ===========================================================================

def _load_det_npz_task(args):
    """Load a single NPZ file and return parsed components.

    Designed to be called via ``mp.Pool.imap_unordered``.

    Parameters
    ----------
    args : tuple
        ``(i, shear_mode, shear_comp, index, base_dir, fname, shape_mode)``
        where *i* is the shear-condition slot (0-3).

    Returns
    -------
    result : tuple or None
        ``(i, index, e, R, w, dw, m00, dm00, match_idx, n_det)``

        - **e**: (N, 2)  — shape estimates (e1, e2)
        - **R**: (N, 2, 2) — full response matrix (R11, R12, R21, R22)
        - **w**: (N,) — detection weight
        - **dw**: (N, 2) — dw/dg1, dw/dg2
        - **m00**: (N,) — flux / monopole moment
        - **dm00**: (N, 2) — dm00/dg1, dm00/dg2
        - **match_idx**: (N,) — truth match index (int32, -1 = unmatched)
        - **n_det**: int — number of detections

        Returns ``None`` if the file does not exist.
    """
    i, shear_mode, shear_comp, index, base_dir, fname, shape_mode = args

    path = _resolve_det_npz_path(base_dir, shear_comp, shear_mode, fname, index)
    if path is None:
        return None

    data = np.load(path)

    n = len(data["fpfs_e1"])
    # --- shapes (depends on estimator) ---
    if shape_mode == "fpfs":
        e = np.column_stack([data["fpfs_e1"], data["fpfs_e2"]]).astype(np.float64)
        R = np.zeros((n, 2, 2), dtype=np.float64)
        R[:, 0, 0] = data["fpfs_R11"].astype(np.float64)
        R[:, 1, 1] = data["fpfs_R22"].astype(np.float64)
    elif shape_mode == "ml":
        e = np.column_stack([data["ml_e1"], data["ml_e2"]]).astype(np.float64)
        R = np.zeros((n, 2, 2), dtype=np.float64)
        R[:, 0, 0] = data["ml_R11"].astype(np.float64)
        R[:, 1, 1] = data["ml_R22"].astype(np.float64)
    else:
        raise ValueError(f"Unknown shape_mode: {shape_mode}")

    # --- detection-level quantities (always from FPFS) ---
    w = data["fpfs_w"].astype(np.float64)
    dw = np.column_stack([
        data["fpfs_dw_dg1"].astype(np.float64),
        data["fpfs_dw_dg2"].astype(np.float64),
    ])
    m00 = data["fpfs_m00"].astype(np.float64)
    dm00 = np.column_stack([
        data["fpfs_dm00_dg1"].astype(np.float64),
        data["fpfs_dm00_dg2"].astype(np.float64),
    ])
    match_idx = data["match_idx"].astype(np.int32)
    n_det = len(e)

    return i, index, e, R, w, dw, m00, dm00, match_idx, n_det


# ===========================================================================
# Direct calibration (unequal per-condition N — no truncation needed)
# ===========================================================================

def _direct_calibrate(g_cond, R_cond, mask_cond,
                      shear_value=0.02, bs_times=100,
                      is_twin=True, n_jobs=4, base_seed=12345,
                      shear_comps=None):
    """Calibrate biases directly from per-condition means.

    Each condition *i* has its own sample of N_i galaxies (possibly
    unequal).  We compute per-condition means of the (already weighted)
    signal and response, then plug into the bias formula.

    Bootstrap: resample each condition independently and recompute m/c.

    Parameters
    ----------
    g_cond : list of 4 ndarrays shape (N_i, 2)
        Per-condition signals (w*e if detection response enabled).
        Missing shear components use empty arrays (len=0).
    R_cond : list of 4 ndarrays shape (N_i, 2, 2)
        Per-condition total responses.
    mask_cond : list of 4 ndarrays (N_i,) or None
        Per-condition magnitude-cut masks.
    shear_value : float
        Applied shear.
    bs_times : int
        Bootstrap iterations.
    is_twin : bool
        If True, split each condition's sample in half for twin bootstrap.
    n_jobs : int
        Not used (kept for signature compatibility).
    base_seed : int
        RNG seed.
    shear_comps : list of str or None
        Shear components to calibrate, e.g. ``["g1"]`` or ``["g1","g2"]``.
        Default ``["g1", "g2"]`` for backward compatibility.

    Returns
    -------
    m_mean : ndarray (2,)
    c_mean : ndarray (2,)
    m_std : ndarray (2,)
    c_std : ndarray (2,)
        Unavailable components are set to NaN.
    """
    if shear_comps is None:
        shear_comps = ["g1", "g2"]

    def _compute_mc(g_list, R_list, m_list):
        """Compute m, c from per-condition means.

        Condition order: 0=g1+, 1=g1-, 2=g2+, 3=g2-.
        Handles missing shear components (empty condition arrays)
        by returning NaN for the unavailable component.
        """
        # Per-condition means (masked) — start with NaN for missing conditions
        g_means = np.full((4, 2), np.nan, dtype=np.float64)
        R_means = np.full((4, 2, 2), np.nan, dtype=np.float64)
        for i in range(4):
            gi = g_list[i]
            Ri = R_list[i]
            if len(gi) == 0:
                continue
            if m_list is not None and m_list[i] is not None:
                ok = m_list[i]
                gi = gi[ok]
                Ri = Ri[ok]
            if len(gi) == 0:
                continue
            g_means[i] = gi.mean(axis=0)
            R_means[i] = Ri.mean(axis=0)

        m = np.full(2, np.nan, dtype=np.float64)
        c = np.full(2, np.nan, dtype=np.float64)

        # g1 (conds 0, 1) — component index 0
        if "g1" in shear_comps and len(g_list[0]) > 0 and len(g_list[1]) > 0:
            shear_1p = g_means[0, 0]
            shear_1m = g_means[1, 0]
            resp_1p = R_means[0, 0, 0]
            resp_1m = R_means[1, 0, 0]
            denom1 = resp_1p + resp_1m
            if np.isfinite(denom1) and denom1 != 0:
                m[0] = (shear_1p - shear_1m) / denom1 / shear_value - 1.0
                c[0] = (shear_1p + shear_1m) / denom1

        # g2 (conds 2, 3) — component index 1
        if "g2" in shear_comps and len(g_list[2]) > 0 and len(g_list[3]) > 0:
            shear_2p = g_means[2, 1]
            shear_2m = g_means[3, 1]
            resp_2p = R_means[2, 1, 1]
            resp_2m = R_means[3, 1, 1]
            denom2 = resp_2p + resp_2m
            if np.isfinite(denom2) and denom2 != 0:
                m[1] = (shear_2p - shear_2m) / denom2 / shear_value - 1.0
                c[1] = (shear_2p + shear_2m) / denom2

        return m, c

    # Point estimate
    m_mean, c_mean = _compute_mc(g_cond, R_cond, mask_cond)

    # Bootstrap
    rng = np.random.default_rng(base_seed)
    m_bs = np.zeros((bs_times, 2), dtype=np.float64)
    c_bs = np.zeros((bs_times, 2), dtype=np.float64)

    for b in tqdm(range(bs_times), desc="Bootstrap", leave=False):
        bs_g = []
        bs_R = []
        for i in range(4):
            n_i = g_cond[i].shape[0]
            if n_i == 0:
                bs_g.append(g_cond[i])
                bs_R.append(R_cond[i])
                continue
            if is_twin:
                n_eff = n_i // 2
                bs_size = n_eff
            else:
                n_eff = n_i
                bs_size = n_i
            if bs_size < 2:
                bs_g.append(g_cond[i])
                bs_R.append(R_cond[i])
            else:
                idx = rng.integers(0, n_eff, size=bs_size)
                if is_twin:
                    idx = np.concatenate([idx, idx + n_eff])
                bs_g.append(g_cond[i][idx])
                bs_R.append(R_cond[i][idx])
        m_bs[b], c_bs[b] = _compute_mc(bs_g, bs_R, mask_cond)

    m_std = np.nanstd(m_bs, axis=0)
    c_std = np.nanstd(c_bs, axis=0)
    return m_mean, c_mean, m_std, c_std


# ===========================================================================
# Main calibration entry point
# ===========================================================================

def get_det_calibration_biases(
    noise_level,
    job_name="xlens_shift",
    fname="p85",
    index_range=(0, 4000),
    shape_mode="both",
    mag_cut=None,
    include_detection_response=True,
    bs_times=100,
    is_twin=True,
    workers=64,
    base_dir=None,
    output_dir=None,
    log_csv=None,
    shear_comps=None,
):
    """Run joint detection+selection+shape calibration on NPZ data.

    Parameters
    ----------
    noise_level : float
        Noise standard deviation.  Converted to folder name via
        ``format_number`` (e.g. 0.594 → "n0594").
    job_name : str
        Dataset identifier.  The NPZ files are expected under
        ``{base_dir}/{format_number(noise_level)}/``.
    fname : str
        Filename prefix used in the NPZ files.
    index_range : tuple (start, end)
        Batch indices to process, inclusive of start, exclusive of end.
    shape_mode : {'fpfs', 'ml', 'both'}
        Which shape estimator to calibrate.
    mag_cut : float or None
        If given, apply a magnitude cut and compute the associated
        selection response.
    include_detection_response : bool
        If True, use per-galaxy detection-weighted signal and response:
        signal = w * e, response = w * R + dw/dg * e.
        If False, use original unweighted e and R_shape (ignore detection weight).
    bs_times : int
        Number of bootstrap iterations.
    is_twin : bool
        Split the sample in half for independent calibration.
    workers : int
        Number of parallel workers for loading and bootstrap.
    base_dir : str or None
        Root directory for NPZ files.  Defaults to
        ``DEFAULT_BASE_DIR / {folder_name}``.
    output_dir : str or None
        Directory for saving ``g_all.npy`` / ``R_all.npy``.
    log_csv : str or None
        CSV file to append calibration results to.
    shear_comps : list of str or None
        Shear components to calibrate, e.g. ``["g1"]`` for single-component
        data (blended sims) or ``["g1", "g2"]`` (default) for full
        two-component calibration.  When ``None``, defaults to
        ``["g1", "g2"]``.

    Returns
    -------
    result : dict or tuple
        If ``shape_mode='both'``, returns a dict
        ``{'fpfs': (m,c,m_err,c_err), 'ml': (m,c,m_err,c_err)}``.
        Otherwise returns the tuple ``(m, c, m_err, c_err)`` directly.
        Unavailable components are set to NaN.
    """
    # ---- resolve directories ----
    folder_name = format_number(noise_level)
    if base_dir is None:
        base_dir = os.path.join(DEFAULT_BASE_DIR, job_name, folder_name)
    if output_dir is None:
        output_dir = os.path.join(DEFAULT_OUTPUT_DIR, job_name, folder_name)
    if log_csv is None:
        log_csv = DEFAULT_LOG_CSV

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_csv) or ".", exist_ok=True)

    img_nB = index_range[1] - index_range[0]
    print(f"[det_calibration] noise={noise_level} ({folder_name}), "
          f"job={job_name}, fname={fname}, "
          f"batches={index_range[0]}..{index_range[1]} ({img_nB}), "
          f"shape_mode={shape_mode}")
    print(f"[det_calibration] base_dir={base_dir}")

    # ---- determine which modes to run ----
    if shape_mode == "both":
        modes = ["fpfs", "ml"]
    else:
        modes = [shape_mode]

    # ---- resolve shear_comps default ----
    if shear_comps is None:
        shear_comps = ["g1", "g2"]

    # ---- shear tasks (built dynamically from shear_comps) ----
    # i=0: g1+ (shear_mode=1, shear_comp='g1')
    # i=1: g1- (shear_mode=0, shear_comp='g1')
    # i=2: g2+ (shear_mode=1, shear_comp='g2')
    # i=3: g2- (shear_mode=0, shear_comp='g2')
    # When shear_comps=["g1"], only the first two tasks are generated.
    # `_direct_calibrate` always receives 4-condition lists; missing
    # components are padded with empty arrays and calibrate to NaN.
    shear_tasks = []
    for comp in shear_comps:
        shear_tasks.append((1, comp))   # positive shear
        shear_tasks.append((0, comp))   # negative shear
    n_conds = len(shear_tasks)  # 2 or 4

    all_results = {}  # mode → calibration result

    for mode in modes:
        print(f"\n{'='*60}\n  Running calibration for shape_mode='{mode}'\n{'='*60}")
        print(f"  shear_comps={shear_comps}  (n_conds={n_conds})")

        # ---- Phase 1: parallel load ----
        tasks = []
        for i, (shear_mode, shear_comp) in enumerate(shear_tasks):
            for index in range(index_range[0], index_range[1]):
                tasks.append((
                    i, shear_mode, shear_comp, index,
                    base_dir, fname, mode,
                ))

        print(f"[load] Dispatching {len(tasks)} NPZ loads with {workers} workers...")
        t0 = time.time()

        # Collect results per batch index
        # batch_data[index] = list of n_conds result tuples (or None placeholders)
        batch_data = {}
        n_missing = 0
        n_loaded = 0

        with mp.Pool(processes=workers) as pool:
            for result in tqdm(
                pool.imap_unordered(_load_det_npz_task, tasks),
                total=len(tasks),
                desc=f"Loading NPZ ({mode})",
            ):
                if result is None:
                    n_missing += 1
                    continue
                i, index, e, R, w, dw, m00, dm00, match_idx, n_det = result
                n_loaded += 1

                if index not in batch_data:
                    batch_data[index] = [None] * n_conds
                batch_data[index][i] = (
                    e, R, w, dw, m00, dm00, match_idx, n_det
                )

        t1 = time.time()
        print(f"[load] Loaded {n_loaded} files, {n_missing} missing "
              f"({100 * n_missing / max(len(tasks), 1):.1f}%), "
              f"in {t1 - t0:.1f}s")

        if n_loaded == 0:
            print("[load] ERROR: no NPZ files loaded.  Check base_dir and fname.")
            all_results[mode] = (np.full(2, np.nan), np.full(2, np.nan),
                                 np.full(2, np.nan), np.full(2, np.nan))
            continue

        # ---- Phase 2: assemble per condition, then stack ----
        # Each shear condition is an independent sample of different galaxies.
        # We collect all galaxies per condition.
        cond_data = {i: [] for i in range(n_conds)}

        for index in sorted(batch_data.keys()):
            results = batch_data[index]
            # Only require the conditions we actually asked for
            if any(results[i] is None for i in range(n_conds)):
                continue
            for i in range(n_conds):
                e_i, R_i, w_i, dw_i, m00_i, dm00_i, _match, _ndet = results[i]
                cond_data[i].append((e_i, R_i, w_i, dw_i, m00_i, dm00_i))

        # Concatenate per condition (n_conds → pad to 4 for _direct_calibrate)
        g_concat = []
        R_concat = []
        w_concat = []
        dw_concat = []
        m00_concat = []
        dm00_concat = []

        for i in range(n_conds):
            if not cond_data[i]:
                print(f"[assemble] ERROR: no data for condition {i}.")
                break
            batch_arrays = list(zip(*cond_data[i]))
            g_concat.append(np.concatenate(batch_arrays[0], axis=0))
            R_concat.append(np.concatenate(batch_arrays[1], axis=0))
            w_concat.append(np.concatenate(batch_arrays[2], axis=0))
            dw_concat.append(np.concatenate(batch_arrays[3], axis=0))
            m00_concat.append(np.concatenate(batch_arrays[4], axis=0))
            dm00_concat.append(np.concatenate(batch_arrays[5], axis=0))

        if len(g_concat) < n_conds:
            all_results[mode] = (np.full(2, np.nan), np.full(2, np.nan),
                                 np.full(2, np.nan), np.full(2, np.nan))
            continue

        # Pad to 4 conditions with empty arrays for missing shear components.
        # _direct_calibrate always expects 4-condition lists.
        _empty_e = np.empty((0, 2), dtype=np.float64)
        _empty_R = np.empty((0, 2, 2), dtype=np.float64)
        _empty_1d = np.empty((0,), dtype=np.float64)
        _empty_2d = np.empty((0, 2), dtype=np.float64)
        while len(g_concat) < 4:
            g_concat.append(_empty_e.copy())
            R_concat.append(_empty_R.copy())
            w_concat.append(_empty_1d.copy())
            dw_concat.append(_empty_2d.copy())
            m00_concat.append(_empty_1d.copy())
            dm00_concat.append(_empty_2d.copy())

        N_per_cond = [g.shape[0] for g in g_concat]
        print(f"[assemble] Per-condition N: {N_per_cond}  (shear_comps={shear_comps})")

        # ---- Phase 3: detection response (per-condition) ----
        # Each condition i independently:
        #   signal_i = w_i * e_i
        #   response_i = w_i * R_i + dw_i/dg * e_i
        # Empty conditions are no-ops because the arrays have shape (0, ...).
        if include_detection_response:
            for i in range(n_conds):
                g_raw = g_concat[i]
                g_concat[i] = w_concat[i][:, None] * g_raw              # w * e
                # R_total[j,k] = w * R_jk + dw/dg_k * e_j  =  w*R  +  e⊗dw
                R_concat[i] = (R_concat[i] * w_concat[i][:, None, None]
                               + g_raw[:, :, None] * dw_concat[i][:, None, :])
                # R_concat[i] shape: (N_i, 2, 2)  —  w*R + (dw ⊗ e)
            print(f"[detection_response] Per-condition: signal = w*e, response = w*R + dw/dg*e")

        # ---- Phase 4: selection response (magnitude cut) ----
        if mag_cut is not None:
            # Compute selection response per condition, average, then add to each cond's R
            R_sel_sum = np.zeros(2, dtype=np.float64); n_valid = 0
            for i in range(n_conds):
                if len(m00_concat[i]) == 0:
                    continue
                try:
                    r = selection_response(
                        flux=m00_concat[i], R_flux=dm00_concat[i],
                        shapes=g_concat[i], zero_point=30.0, mag_cut=mag_cut)
                    if np.all(np.isfinite(r)):
                        R_sel_sum += r; n_valid += 1
                except Exception: pass
            R_sel = R_sel_sum / n_valid if n_valid > 0 else np.zeros(2)
            print(f"[selection_response] R_sel = [{R_sel[0]:.6f}, {R_sel[1]:.6f}]")

            # Build mask & apply R_sel to each condition
            mask_list = []
            for i in range(4):
                if len(m00_concat[i]) == 0:
                    mask_list.append(None)
                    continue
                R_concat[i][:, 0, 0] += R_sel[0]
                R_concat[i][:, 1, 1] += R_sel[1]
                mag_i = 30.0 - 2.5 * np.log10(np.maximum(m00_concat[i], 1e-30))
                mask_list.append(mag_i < mag_cut)
                n_pass_i = np.sum(mask_list[-1])
                print(f"  cond {i}: {n_pass_i}/{len(mag_i)} pass mag<{mag_cut}")
        else:
            mask_list = None

        # ---- Phase 5: direct mean-based calibration ----
        # Compute per-condition means, then plug into the bias formula.
        # Bootstrap resamples each condition independently.
        # Missing shear components (empty arrays) → NaN in the result.
        cal_result = _direct_calibrate(
            g_concat, R_concat, mask_list,
            shear_value=0.02, bs_times=bs_times,
            is_twin=is_twin, n_jobs=workers, base_seed=12345,
            shear_comps=shear_comps,
        )
        m_mean, c_mean, m_std, c_std = cal_result
        print(f"[result] m = {m_mean}, c = {c_mean}")
        print(f"[result] m_err = {m_std}, c_err = {c_std}")

        all_results[mode] = cal_result

        # ---- Save per-condition arrays ----
        for i in range(n_conds):
            np.save(os.path.join(output_dir, f"{fname}_{mode}_g_c{i}.npy"), g_concat[i])
            np.save(os.path.join(output_dir, f"{fname}_{mode}_R_c{i}.npy"), R_concat[i])

        # ---- Log to CSV ----
        _append_csv(log_csv, noise_level, fname, mode, mag_cut, cal_result, shear_comps)

    # ---- return ----
    if shape_mode == "both":
        return all_results
    else:
        return all_results[modes[0]]


# ===========================================================================
# CSV logging
# ===========================================================================

def _append_csv(log_csv, noise_level, fname, mode, mag_cut, cal_result, shear_comps=None):
    """Append one line to the calibration log CSV."""
    m, c, m_err, c_err = cal_result
    tag = fname
    if mag_cut is not None:
        tag += f"_mag{mag_cut}"
    if shear_comps is not None:
        tag += f"_{''.join(shear_comps)}"
    try:
        with open(log_csv, "a") as f:
            line = (
                f"{noise_level},{mode},{tag},"
                f"{m[0]:.8f},{m[1]:.8f},"
                f"{m_err[0]:.8f},{m_err[1]:.8f},"
                f"{c[0]:.8f},{c[1]:.8f},"
                f"{c_err[0]:.8f},{c_err[1]:.8f},"
                f"{datetime.datetime.now().isoformat()}\n"
            )
            f.write(line)
    except Exception as e:
        print(f"[log] Failed to write CSV: {e}")


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    start_time = time.time()

    p = argparse.ArgumentParser(
        description="Joint detection + selection + shape calibration"
    )
    p.add_argument("--noise", type=float, required=True,
                   help="Noise level (e.g. 0.594)")
    p.add_argument("--job_name", type=str, default="xlens_shift",
                   help="Dataset identifier")
    p.add_argument("--fname", type=str, default="p85",
                   help="Filename prefix")
    p.add_argument("--index_start", type=int, default=0,
                   help="First batch index (inclusive)")
    p.add_argument("--index_end", type=int, default=4000,
                   help="Last batch index (exclusive)")
    p.add_argument("--shape_mode", type=str, default="both",
                   choices=["fpfs", "ml", "both"],
                   help="Shape estimator to calibrate")
    p.add_argument("--mag_cut", type=float, default=None,
                   help="Magnitude cut for selection response")
    p.add_argument("--no_detection_response", action="store_true",
                   help="Disable detection response correction")
    p.add_argument("--bs_times", type=int, default=100,
                   help="Bootstrap iterations")
    p.add_argument("--no_twin", action="store_true",
                   help="Disable twin-sample split")
    p.add_argument("--workers", type=int, default=64,
                   help="Number of parallel workers")
    p.add_argument("--base_dir", type=str, default=None,
                   help="Root NPZ directory (default: env or built-in)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory for intermediate arrays")
    p.add_argument("--log_csv", type=str, default=None,
                   help="CSV log file path")
    p.add_argument("--shear_comps", nargs="+", type=str,
                   default=["g1", "g2"],
                   help="Shear components to calibrate (default: g1 g2). "
                        "Use 'g1' for blended single-shear data.")

    args = p.parse_args()

    print(f"=== Detection + Shape Calibration ===")
    print(f"    noise={args.noise}, job={args.job_name}, fname={args.fname}")
    print(f"    index_range=[{args.index_start}, {args.index_end})")
    print(f"    shape_mode={args.shape_mode}, mag_cut={args.mag_cut}")
    print(f"    shear_comps={args.shear_comps}")

    result = get_det_calibration_biases(
        noise_level=args.noise,
        job_name=args.job_name,
        fname=args.fname,
        index_range=(args.index_start, args.index_end),
        shape_mode=args.shape_mode,
        mag_cut=args.mag_cut,
        include_detection_response=not args.no_detection_response,
        bs_times=args.bs_times,
        is_twin=not args.no_twin,
        workers=args.workers,
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        log_csv=args.log_csv,
        shear_comps=args.shear_comps,
    )

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

    # Print summary
    if args.shape_mode == "both":
        for mode, (m, c, m_err, c_err) in result.items():
            print(f"\n[{mode}] m = {m}")
            print(f"[{mode}] c = {c}")
            print(f"[{mode}] m_err = {m_err}")
            print(f"[{mode}] c_err = {c_err}")
    else:
        m, c, m_err, c_err = result
        print(f"\nm = {m}")
        print(f"c = {c}")
        print(f"m_err = {m_err}")
        print(f"c_err = {c_err}")
