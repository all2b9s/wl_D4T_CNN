"""
FPFS Detection + ML Shape Measurement — Blended Image Pipeline
===================================================================

End-to-end batch processing for DC1 blended simulation FITS exposures:
  1. Load FITS exposure + truth catalog, extract noise sigma
  2. Generate single renoise canvas (shared by detection & ML)
  3. FPFS detect → filter by weight (fpfs_w >= 1e-6)
  4. Cut stamps + noise stamps
  5. Batch ML inference on GPU (all stamps in chunk)
  6. Merge & save per-exposure catalogs (NPZ + CSV)

Output structure compatible with ``get_det_calibration_biases()``.

Usage:
    python blended_run.py --psf_path ./logs/blended_test_psf.npy \
        --cat_ref_path /projects/bdsp/wenyinli/codes/data/catsim-v4/OneDegSq.fits \
        --range 0 100

References:
    - blended_test.ipynb  (FITS loading reference)
    - src/anacal/detection_ml_pipeline.py  (Phase 2-4 functions)
    - src/blending/ml_shape_measurement.py (ml_shape_measure)
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import anacal
import argparse
import time
import threading
import subprocess
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from astropy.io import fits
import fitsio

from src.anacal.batch_calibration import Calibrator
from src.anacal.detection_ml_pipeline import (
    run_detection_and_fpfs,
    cut_stamps_from_large,
    merge_and_save_catalog,
    _build_task_from_fpfs_config,
    _build_cell_list,
    match_detections_to_truth,
)
from src.blending.ml_shape_measurement import ml_shape_measure
from src.architecture.forward8_CNN import Forward8_fixW_CNN
from src.architecture.quad_Gauss import QuadGauss
from src.datasets.dataset_toolkit import get_weighted_e

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

FPFS_CONFIG = anacal.fpfs.FpfsConfig(
    sigma_shapelets1=0.45,
    sigma_shapelets2=0.55,
)

BLENDED_DIR = "/taiga/illinois/las/astro/xinliuxl/DC1_sim_blended"
OUTPUT_ROOT = "/work/hdd/bfmo/wenyinli/measurement/blended_sims"
MODEL_PATH = "./models/F8_fpfs_l5c32r01_50ep.pth"
PSF_PATH = "./logs/blended_test_psf.npy"
CAT_REF_PATH = "/projects/bdsp/wenyinli/codes/data/catsim-v4/OneDegSq.fits"

# sim_mode → (shear_comp, shear_mode) for calibration-compatible output paths
# sim_mode0: g1 = -0.02  →  g1_0
# sim_mode40: g1 = +0.02  →  g1_1
SIM_MODE_MAP = {
    "sim_mode0": ("g1", 0),
    "sim_mode40": ("g1", 1),
}

PIXEL_SCALE = 0.2
WEIGHT_THRESHOLD = 1e-6
# Drop detections closer than this many pixels to the image border
# (normal detection mode only) so the full stamp always fits the exposure.
EDGE_MARGIN = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# GPU utilization monitor (background thread)
# ---------------------------------------------------------------------------

def _gpu_util_monitor(interval, results):
    """Background thread: poll nvidia-smi every ``interval`` seconds."""
    try:
        while not results.get("stop"):
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                timeout=2, text=True,
            )
            for line in out.strip().splitlines():
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 4:
                    continue
                try:
                    gpu_u = float(parts[0])
                    mem_u = float(parts[1])
                    mem_used = float(parts[2].split()[0])
                    mem_total = float(parts[3].split()[0])
                except (ValueError, IndexError):
                    continue
                results["util_samples"].append(gpu_u)
                results["mem_util_samples"].append(mem_u)
                results["mem_frac_samples"].append(
                    100.0 * mem_used / mem_total if mem_total > 0 else 0.0
                )
            time.sleep(interval)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CPU preparation workers
# ---------------------------------------------------------------------------

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


@dataclass(frozen=True)
class _CpuPreparationConfig:
    """Immutable dependencies shared by CPU chunk and exposure workers."""

    blended_dir: str
    cat_ref: np.ndarray
    psf_image: np.ndarray
    noise_mode: str
    noise_level: float | None
    mag_zero: float
    pixel_scale: float
    stamp_size: int
    skip_detection: bool
    mag_cut: float | None
    parity_centering: bool
    center_on_truth: bool
    match_threshold: float


def _empty_stamps(stamp_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return empty science and noise stamp arrays with the expected shape."""
    shape = (0, stamp_size, stamp_size)
    return np.empty(shape, dtype=np.float64), np.empty(shape, dtype=np.float64)


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


def _cut_signal_and_noise_stamps(
    large_img: np.ndarray,
    noise_canvas: np.ndarray,
    det_cat: np.ndarray,
    stamp_size: int,
    parity_centering: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Cut matched science and renoise stamps, including the empty case."""
    if len(det_cat) == 0:
        return _empty_stamps(stamp_size)

    stamps, _ = cut_stamps_from_large(
        large_img, det_cat, stamp_size=stamp_size, parity_centering=parity_centering,
    )
    noise_stamps, _ = cut_stamps_from_large(
        noise_canvas, det_cat, stamp_size=stamp_size, parity_centering=parity_centering,
    )
    return stamps, noise_stamps


def _process_one_exposure(
    i_local: int,
    exp_start: int,
    sim_mode: str,
    config: _CpuPreparationConfig,
) -> tuple:
    """Load, detect, filter, and cut stamps for one exposure."""
    exp_idx = exp_start + i_local
    sim_dir = os.path.join(config.blended_dir, sim_mode)
    fits_path = os.path.join(sim_dir, f"exp-i-{exp_idx:05d}.fits")
    truth_path = os.path.join(sim_dir, f"truth-{exp_idx:05d}.fits")

    t1 = time.perf_counter()
    large_img, noise_sigma, truth_df, positions = load_blended_exposure(
        fits_path, truth_path, config.cat_ref, config.pixel_scale,
        config.noise_mode, config.noise_level,
    )
    t2 = time.perf_counter()
    noise_canvas = np.random.default_rng(19491001 + exp_idx).normal(
        0, noise_sigma, large_img.shape,
    ).astype(np.float32)
    noise_variance = noise_sigma ** 2 if noise_sigma > 0 else 1e-4
    t3 = time.perf_counter()

    if config.skip_detection:
        det_cat = _build_truth_detection_catalog(truth_df, config.mag_cut)
        stamps, noise_stamps = _cut_signal_and_noise_stamps(
            large_img, noise_canvas, det_cat, config.stamp_size, False,
        )
        t4 = time.perf_counter()
        timings = {
            "load": t2 - t1, "noise_gen": t3 - t2, "build_task": 0.0,
            "detect": 0.0, "weight_filter": 0.0, "cut": t4 - t3,
            "n_before": len(truth_df), "n_after": len(det_cat),
            "noise_sigma": noise_sigma, "test_mode": True,
        }
    else:
        task = _build_task_from_fpfs_config(mag_zero=config.mag_zero, pixel_scale=config.pixel_scale)
        cells = _build_cell_list(*large_img.shape, pixel_scale=config.pixel_scale)
        t4 = time.perf_counter()
        det_cat = run_detection_and_fpfs(
            large_image=large_img, psf_image=config.psf_image, fpfs_config=FPFS_CONFIG,
            mag_zero=config.mag_zero, pixel_scale=config.pixel_scale,
            noise_variance=noise_variance, noise_array=noise_canvas, task=task, cells=cells,
        )
        t5 = time.perf_counter()
        n_before = len(det_cat)
        det_cat = det_cat[det_cat["fpfs_w"] >= WEIGHT_THRESHOLD]
        n_after_w = len(det_cat)
        img_ny, img_nx = large_img.shape
        det_cat = det_cat[
            (det_cat["y"] >= EDGE_MARGIN) & (det_cat["y"] < img_ny - EDGE_MARGIN)
            & (det_cat["x"] >= EDGE_MARGIN) & (det_cat["x"] < img_nx - EDGE_MARGIN)
        ]
        # Crossmatch test: re-center stamps on the matched truth positions
        # (same detection-selected sample & weights), isolating the stamp-
        # centering effect from sample/selection effects.
        if config.center_on_truth:
            det_cat = _recenter_on_truth(det_cat, positions, config.match_threshold)
        t6 = time.perf_counter()
        stamps, noise_stamps = _cut_signal_and_noise_stamps(
            large_img, noise_canvas, det_cat, config.stamp_size, config.parity_centering,
        )
        t7 = time.perf_counter()
        timings = {
            "load": t2 - t1, "noise_gen": t3 - t2, "build_task": t4 - t3,
            "detect": t5 - t4, "weight_filter": t6 - t5, "cut": t7 - t6,
            "n_before": n_before, "n_after": len(det_cat), "n_after_w": n_after_w,
            "noise_sigma": noise_sigma, "test_mode": False,
        }

    return i_local, exp_idx, large_img, noise_sigma, truth_df, positions, det_cat, stamps, noise_stamps, timings


def _prepare_cpu_chunk(
    exp_start: int, exp_end: int, sim_mode: str, config: _CpuPreparationConfig,
) -> dict:
    """Prepare a chunk in parallel: load exposures, run FPFS, and cut stamps."""
    chunk_n = exp_end - exp_start
    started = time.time()
    results = [None] * chunk_n
    with ThreadPoolExecutor(max_workers=min(chunk_n, 32)) as executor:
        futures = [executor.submit(_process_one_exposure, i, exp_start, sim_mode, config)
                   for i in range(chunk_n)]
        for future in as_completed(futures):
            result = future.result()
            results[result[0]] = result
            _, exp_idx, _, noise_sigma, _, _, _, _, _, timings = result
            if timings["test_mode"]:
                mag_info = f"  mag_cut<{config.mag_cut}" if config.mag_cut is not None else "  (no mag cut)"
                print(f"    [{sim_mode}] exp-{exp_idx:05d}: noise_sigma={noise_sigma:.4f}  "
                      f"[TEST] truth objects: {timings['n_before']} → {timings['n_after']}{mag_info}")
            else:
                print(f"    [{sim_mode}] exp-{exp_idx:05d}: noise_sigma={noise_sigma:.4f}  "
                      f"detections: {timings['n_before']} → {timings['n_after_w']} → {timings['n_after']} "
                      f"(w<{WEIGHT_THRESHOLD}, edge<{EDGE_MARGIN}px)")

    fields = list(zip(*results))
    total_stamps = sum(len(stamps) for stamps in fields[7])
    elapsed = time.time() - started
    mode_tag = "[TEST]" if config.skip_detection else ""
    print(f"  Chunk [{exp_start}:{exp_end}) {chunk_n} exposures, {total_stamps} stamps {mode_tag}  [{elapsed:.1f}s]")
    return {
        "exp_start": exp_start, "exp_end": exp_end, "chunk_n": chunk_n,
        "t_cpu": elapsed, "total_stamps": total_stamps,
        "all_large_images": list(fields[2]), "all_noise_sigmas": list(fields[3]),
        "all_truth_dfs": list(fields[4]), "all_positions": list(fields[5]),
        "all_det_cats": list(fields[6]), "all_stamps_list": list(fields[7]),
        "all_noise_stamps_list": list(fields[8]),
    }


# ===========================================================================
# Main pipeline
# ===========================================================================

def run_detection_ml_pipeline(
    blended_dir: str = BLENDED_DIR,
    output_root: str = OUTPUT_ROOT,
    psf_path: str = PSF_PATH,
    cat_ref_path: str = CAT_REF_PATH,
    sim_modes: list = None,
    exp_range: tuple = (0, 100),
    noise_level: float = None,
    io_chunk_size: int = 1,
    ml_batch_size: int = 800,
    n_workers: int = 32,
    mag_zero: float = 30.0,
    pixel_scale: float = 0.2,
    psf_fwhm: float = 0.85,
    stamp_size: int = 64,
    fname: str = "det_ml",
    model_path: str = MODEL_PATH,
    model_type: str = "cnn",
    gauss_sigma: float = 4.0,
    save_npz: bool = True,
    save_csv: bool = True,
    match_threshold: float = 3.0,
    dtype: str = "float32",
    skip_detection: bool = False,
    mag_cut: float = None,
    parity_centering: bool = False,
    center_on_truth: bool = False,
):
    if sim_modes is None:
        sim_modes = list(SIM_MODE_MAP.keys())

    noise_mode = "from_variance" if noise_level is None else "fixed"

    # ---- Device & Model ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[pipeline] Using device: {device}")
    print(f"[pipeline] Model type: {model_type}")

    if model_type == "gauss":
        model = QuadGauss(sigma=gauss_sigma).to(device)
        model.eval()
        print(f"[pipeline] QuadGauss model created (sigma={gauss_sigma}, no checkpoint needed)")
    elif model_type == "cnn":
        model = Forward8_fixW_CNN(num_layers=5, base_channels=32, res_factor=0.1,
                                  nl_sigma=4, gw_sigma=4).to(device)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        print(f"[pipeline] CNN model loaded from {model_path}")
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'cnn' or 'gauss'.")

    calibrator = Calibrator(model, device=device, dtype=dtype)

    # ---- Load PSF & cat_ref once ----
    psf_image = np.load(psf_path).astype(np.float64)
    if psf_image.ndim == 3:
        psf_image = psf_image[0]
    # Trim border pixel to ensure even dimensions (FFT requires even ny/nx)
    # The loaded PSF is 49×49; trimming [:-1, :-1] gives 48×48.
    if psf_image.shape[0] % 2 == 1 or psf_image.shape[1] % 2 == 1:
        psf_image = psf_image[:-1, :-1]
    print(f"[pipeline] PSF shape: {psf_image.shape}  from {psf_path}")

    cat_ref = fitsio.read(cat_ref_path)
    print(f"[pipeline] cat_ref loaded from {cat_ref_path}  ({len(cat_ref)} objects)")

    # ---- Summary ----
    print(f"[pipeline] Noise mode: {noise_mode}"
          + (f"  sigma={noise_level}" if noise_level is not None else "  (from variance_map)"))
    print(f"[pipeline] I/O chunk: {io_chunk_size} exposures, ML batch: {ml_batch_size}")
    print(f"[pipeline] sim_modes: {sim_modes}, exposure range: {exp_range}")
    print(f"[pipeline] Save npz={save_npz}, csv={save_csv}")
    print(f"[pipeline] Weight threshold: fpfs_w >= {WEIGHT_THRESHOLD}")
    print(f"[pipeline] Test mode: skip_detection={skip_detection}"
          + (f"  mag_cut={mag_cut}" if mag_cut is not None else "  (no magnitude cut)"))
    print(f"[pipeline] Parity centering (detection only): {parity_centering}")
    print(f"[pipeline] Center stamps on matched truth (crossmatch test): {center_on_truth}"
          + (f"  (match_threshold={match_threshold} px)" if center_on_truth else ""))

    # ---- Timing accumulators ----
    is_cuda = (device == "cuda")
    t_total_io = 0.0
    t_total_cpu = 0.0
    t_total_gpu = 0.0
    t_total_save = 0.0
    n_chunks_total = 0

    cpu_config = _CpuPreparationConfig(
        blended_dir=blended_dir,
        cat_ref=cat_ref,
        psf_image=psf_image,
        noise_mode=noise_mode,
        noise_level=noise_level,
        mag_zero=mag_zero,
        pixel_scale=pixel_scale,
        stamp_size=stamp_size,
        skip_detection=skip_detection,
        mag_cut=mag_cut,
        parity_centering=parity_centering,
        center_on_truth=center_on_truth,
        match_threshold=match_threshold,
    )

    # ==================================================================
    # Pipeline loop: for each sim_mode, process exposure chunks
    # ==================================================================

    for sim_mode in sim_modes:
        shear_comp, shear_mode = SIM_MODE_MAP[sim_mode]
        print(f"\n{'='*60}")
        print(f"[pipeline] sim_mode={sim_mode}  →  output: {shear_comp}_{shear_mode}")
        print(f"{'='*60}")

        exp_indices = list(range(exp_range[0], exp_range[1], io_chunk_size))

        with ThreadPoolExecutor(max_workers=1) as prefetch_exec:
            cpu_future = None

            for i_chunk, exp_start in enumerate(exp_indices):
                exp_end = min(exp_start + io_chunk_size, exp_range[1])
                chunk_n = exp_end - exp_start
                t_chunk = time.time()

                # --------------------------------------------------
                # Obtain CPU data (or consume prefetched)
                # --------------------------------------------------
                if cpu_future is not None:
                    cpu_data = cpu_future.result()
                    cpu_future = None
                else:
                    cpu_data = _prepare_cpu_chunk(
                        exp_start, exp_end, sim_mode, cpu_config,
                    )

                t_cpu = cpu_data["t_cpu"]
                total_stamps = cpu_data["total_stamps"]
                all_det_cats = cpu_data["all_det_cats"]
                all_truth_dfs = cpu_data["all_truth_dfs"]
                all_positions = cpu_data["all_positions"]
                all_stamps_list = cpu_data["all_stamps_list"]
                all_noise_stamps_list = cpu_data["all_noise_stamps_list"]

                # --------------------------------------------------
                # Prefetch NEXT chunk (CPU work in background)
                # --------------------------------------------------
                if i_chunk + 1 < len(exp_indices):
                    _next_start = exp_indices[i_chunk + 1]
                    _next_end = min(_next_start + io_chunk_size, exp_range[1])
                    cpu_future = prefetch_exec.submit(
                        _prepare_cpu_chunk, _next_start, _next_end, sim_mode, cpu_config,
                    )

                # --------------------------------------------------
                # Step C: Big-batch ML inference on GPU
                # --------------------------------------------------
                if total_stamps > 0:
                    print(f"[Step C] ML measurement on {total_stamps} stamps "
                          f"(batch_size={ml_batch_size}) ...")
                    if is_cuda:
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                    t_gpu_start = time.time()

                    stamps_all = np.concatenate(
                        [s for s in all_stamps_list if len(s) > 0], axis=0
                    )
                    noise_stamps_all = np.concatenate(
                        [n for n in all_noise_stamps_list if len(n) > 0], axis=0
                    )

                    # GPU utilization monitor
                    gpu_mon = {"util_samples": [], "mem_util_samples": [],
                               "mem_frac_samples": [], "stop": False}
                    gpu_thread = None
                    if is_cuda:
                        gpu_thread = threading.Thread(
                            target=_gpu_util_monitor, args=(0.5, gpu_mon), daemon=True,
                        )
                        gpu_thread.start()

                    ml_shapes_all, ml_R_all = ml_shape_measure(
                        stamps=stamps_all,
                        psf_image=psf_image,
                        calibrator=calibrator,
                        pixel_scale=pixel_scale,
                        psf_fwhm=psf_fwhm,
                        noise_arrays=noise_stamps_all,
                        n_workers=n_workers,
                        dtype=dtype,
                        ml_batch_size=ml_batch_size,
                        seed=20020620 + exp_start,
                    )

                    if is_cuda:
                        torch.cuda.synchronize()
                    t_gpu = time.time() - t_gpu_start

                    gpu_mon["stop"] = True
                    if gpu_thread is not None:
                        gpu_thread.join(timeout=2)
                    gpu_util = np.mean(gpu_mon["util_samples"]) if gpu_mon["util_samples"] else 0.0
                    gpu_mem_util = np.mean(gpu_mon["mem_util_samples"]) if gpu_mon["mem_util_samples"] else 0.0
                    gpu_mem_frac = np.mean(gpu_mon["mem_frac_samples"]) if gpu_mon["mem_frac_samples"] else 0.0
                    gpu_mem = ""
                    if is_cuda:
                        gpu_mem = (f", GPU mem: "
                                   f"{torch.cuda.max_memory_allocated()/1024**3:.2f} GB peak, "
                                   f"{torch.cuda.memory_allocated()/1024**3:.2f} GB current")
                        torch.cuda.reset_peak_memory_stats()

                    print(f"  ML shapes: {ml_shapes_all.shape}, R: {ml_R_all.shape}")
                    print(f"  [Step C] GPU: {t_gpu:.1f}s  "
                          f"GPU util: {gpu_util:.0f}%, "
                          f"GPU mem util: {gpu_mem_util:.0f}%, "
                          f"GPU mem cap: {gpu_mem_frac:.0f}%{gpu_mem}")
                else:
                    ml_shapes_all = np.empty((0, 2))
                    ml_R_all = np.empty((0, 1, 2, 2))
                    t_gpu = 0.0

                # --------------------------------------------------
                # Step D: Split ML results & save per-exposure
                # --------------------------------------------------
                t_save_start = time.time()
                ml_offset = 0
                n_total_saved = 0

                save_root = os.path.join(
                    output_root,
                    f"{shear_comp}_{shear_mode}",
                    fname,
                )

                for i_local in range(chunk_n):
                    exp_idx = exp_start + i_local
                    det_i = all_det_cats[i_local]
                    n_det_i = len(det_i)

                    if n_det_i > 0:
                        ml_s_i = ml_shapes_all[ml_offset:ml_offset + n_det_i]
                        ml_R_i = ml_R_all[ml_offset:ml_offset + n_det_i]
                        ml_offset += n_det_i
                        n_total_saved += n_det_i
                    else:
                        ml_s_i = np.empty((0, 2))
                        ml_R_i = np.empty((0, 1, 2, 2))

                    merge_and_save_catalog(
                        det_cat=det_i,
                        ml_shapes=ml_s_i,
                        ml_R=ml_R_i,
                        positions=all_positions[i_local],
                        output_dir=save_root,
                        truth_cat=all_truth_dfs[i_local],
                        fname=f"catalog_{exp_idx}",
                        match_threshold=match_threshold,
                        save_npz=save_npz,
                        save_csv=save_csv,
                        verbose=False,
                    )

                t_save = time.time() - t_save_start

                # --- Cleanup ---
                del all_det_cats, all_truth_dfs, all_positions
                del all_stamps_list, all_noise_stamps_list
                if total_stamps > 0:
                    del stamps_all, ml_shapes_all, ml_R_all
                    if is_cuda:
                        torch.cuda.empty_cache()

                # --- Accumulate ---
                t_total_io += 0.0  # I/O included in t_cpu for blended
                t_total_cpu += t_cpu
                t_total_gpu += t_gpu
                t_total_save += t_save
                n_chunks_total += 1

                t_elapsed = time.time() - t_chunk
                print(f"[pipeline] Chunk done — {n_total_saved} stamps saved in {t_save:.1f}s  "
                      f"| wall {t_elapsed:.1f}s  "
                      f"CPU={t_cpu:.1f}s  GPU={t_gpu:.1f}s  Save={t_save:.1f}s")
                import sys
                sys.stdout.flush()

    # ---- Grand total ----
    t_total = t_total_cpu + t_total_gpu + t_total_save
    print(f"\n{'='*60}")
    print(f"[pipeline] All batches finished.")
    print(f"[pipeline] === Timing Summary ({n_chunks_total} chunks) ===")
    print(f"  Total wall time: {t_total:.1f}s ({t_total/60:.1f} min)")
    print(f"  CPU (load+detect+cut): {t_total_cpu:8.1f}s  ({100*t_total_cpu/max(t_total,1):5.1f}%)")
    print(f"  GPU (ML inference):    {t_total_gpu:8.1f}s  ({100*t_total_gpu/max(t_total,1):5.1f}%)")
    print(f"  Save:                  {t_total_save:8.1f}s  ({100*t_total_save/max(t_total,1):5.1f}%)")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    start_time = time.time()

    p = argparse.ArgumentParser(
        description="FPFS Detection + ML Shape Measurement — Blended Image Pipeline"
    )
    p.add_argument("--blended_dir", type=str, default=BLENDED_DIR,
                   help="Root directory of DC1 blended simulations")
    p.add_argument("--output_root", type=str, default=OUTPUT_ROOT,
                   help="Root directory for output NPZ/CSV catalogs")
    p.add_argument("--psf_path", type=str, default=PSF_PATH,
                   help="Path to PSF .npy file")
    p.add_argument("--cat_ref_path", type=str, default=CAT_REF_PATH,
                   help="Path to OneDegSq.fits reference catalog")
    p.add_argument("--sim_modes", nargs="+", type=str,
                   default=list(SIM_MODE_MAP.keys()),
                   help="Simulation mode names (default: sim_mode0 sim_mode40)")
    p.add_argument("--noise", type=float, default=None,
                   help="Fixed noise level; if not set, estimated from variance_map")
    p.add_argument("--fname", type=str, default="det_ml",
                   help="Output subdirectory name")
    p.add_argument("--range", nargs=2, type=int, default=[0, 100],
                   help="Exposure index range [start, end)")
    p.add_argument("--model", type=str, default=MODEL_PATH,
                   help="Path to trained ML model .pth (ignored when --model_type=gauss)")
    p.add_argument("--model_type", type=str, default="cnn", choices=["cnn", "gauss"],
                   help="Model type: 'cnn' (Forward8_fixW_CNN) or 'gauss' (QuadGauss)")
    p.add_argument("--gauss_sigma", type=float, default=4.0,
                   help="Gaussian weight sigma in pixels (only for --model_type=gauss)")
    p.add_argument("--workers", type=int, default=32,
                   help="Number of parallel workers for ML Q-image prep")
    p.add_argument("--io_chunk", type=int, default=1,
                   help="Exposures per I/O chunk (default 1)")
    p.add_argument("--ml_batch", type=int, default=600,
                   help="ML inference batch size")
    p.add_argument("--no_csv", action="store_true",
                   help="Skip CSV output")
    p.add_argument("--no_npz", action="store_true",
                   help="Skip NPZ output")
    p.add_argument("--skip_detection", action="store_true",
                   help="Skip FPFS detection; use truth positions directly as cutout centers")
    p.add_argument("--parity_centering", action="store_true",
                   help="Place odd-parity detection peaks at stamp pixel 31 "
                        "(even at 32) to restore symmetry around the geometric "
                        "centre for even-sized stamps. Detection mode only; "
                        "ignored with --skip_detection.")
    p.add_argument("--center_on_truth", action="store_true",
                   help="Crossmatch detections to the truth catalog and cut "
                        "postage stamps centered on the matched TRUTH positions "
                        "instead of the detection positions. Keeps the same "
                        "detection-selected sample and weights -- isolates the "
                        "stamp-centering effect from sample/selection effects. "
                        "Detection mode only; ignored with --skip_detection. "
                        "Use a distinct --fname (e.g. det_ml_ct) for the output.")
    p.add_argument("--mag_cut", type=float, default=None,
                   help="When --skip_detection is set, only use truth objects with i_ab < mag_cut "
                        "(e.g. 24.5). Has no effect in normal detection mode.")

    args = p.parse_args()

    run_detection_ml_pipeline(
        blended_dir=args.blended_dir,
        output_root=args.output_root,
        psf_path=args.psf_path,
        cat_ref_path=args.cat_ref_path,
        sim_modes=args.sim_modes,
        exp_range=tuple(args.range),
        noise_level=args.noise,
        io_chunk_size=args.io_chunk,
        ml_batch_size=args.ml_batch,
        n_workers=args.workers,
        fname=args.fname,
        model_path=args.model,
        model_type=args.model_type,
        gauss_sigma=args.gauss_sigma,
        save_npz=not args.no_npz,
        save_csv=not args.no_csv,
        skip_detection=args.skip_detection,
        mag_cut=args.mag_cut,
        parity_centering=args.parity_centering,
        center_on_truth=args.center_on_truth,
    )

    print(f"[main] Total time: {time.time() - start_time:.1f}s "
          f"({(time.time() - start_time)/60:.1f} min)")
