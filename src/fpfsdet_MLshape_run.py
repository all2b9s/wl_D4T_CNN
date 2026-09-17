"""
FPFS Detection + ML Shape Measurement — Batch Pipeline (Optimized)
===================================================================

Optimized end-to-end batch processing:
  1. Parallel I/O via build_dataset() — load many directories at once
  2. Per-directory: stitch 100 cutouts → FPFS detect → collect stamps
  3. Concatenate all stamps → single large-batch ML inference on GPU
  4. Split results → merge & save per-directory catalogs

Usage:
    python src/fpfsdet_MLshape_run.py --noise 0.594 --fname det_ml --range 0 4000

References:
    - src/shape_measurement.py  (argparse / loop structure)
    - src/anacal/detection_ml_pipeline.py  (Phase 1-4 functions)
    - src/blending/ml_shape_measurement.py (ml_shape_measure)
"""

import os

# Must be set before importing numpy / torch / anacal so outer threading
# can parallelize Step B instead of competing with inner BLAS/OpenMP pools.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
import anacal
import argparse
import time
import threading
import subprocess
import re
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.anacal.cal_toolkit import format_number, build_dataset
from src.anacal.batch_calibration import Calibrator
from src.anacal.detection_ml_pipeline import (
    generate_grid_positions,
    stitch_large_image,
    run_detection_and_fpfs,
    cut_stamps_from_large,
    merge_and_save_catalog,
    match_detections_to_truth,
    _build_task_from_fpfs_config,
    _build_cell_list,
)
from src.blending.ml_shape_measurement import ml_shape_measure
from src.architecture.forward8_CNN import Forward8_fixW_CNN

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

FPFS_CONFIG = anacal.fpfs.FpfsConfig(
    sigma_shapelets=0.52,
    sigma_shapelets1=0.45,
    sigma_shapelets2=0.55,
)

# Repository root = parent of ``src/``; keeps the model default CWD-independent.
REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = "/work/nvme/bfmo/wenyinli/datasets/xlens_shift"
OUTPUT_ROOT = "/work/hdd/bfmo/wenyinli/measurement/xlens_sims"
MODEL_PATH = str(REPO_ROOT / "models" / "F8_fpfs_l5c32r01_50ep.pth")

DEFAULT_SHEAR_TASKS = [
    (0, "g1"),
    (1, "g1"),
    (0, "g2"),
    (1, "g2"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_shear_tasks(task_list):
    """Parse shear tasks like ['0_g1', '1_g2'] -> [(0, 'g1'), (1, 'g2')]"""
    parsed = []
    for t in task_list:
        try:
            idx, shear = t.split("_")
            parsed.append((int(idx), shear))
        except ValueError:
            raise ValueError(
                f"Invalid shear task format '{t}', expected like '0_g1' or '1_g2'"
            )
    return parsed


# ---------------------------------------------------------------------------
# Main batch loop (optimized)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# GPU utilization monitor (background thread)
# ---------------------------------------------------------------------------

def _gpu_util_monitor(interval, results):
    """Background thread: poll nvidia-smi every `interval` seconds.

    Collects:
      - results["util_samples"]: average GPU utilization (%) per poll
      - results["mem_util_samples"]: average memory utilization (%) per poll
      - results["mem_frac_samples"]: average memory capacity used (%) per poll
    """
    try:
        while not results.get("stop"):
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                timeout=2, text=True
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
                results["mem_frac_samples"].append(100.0 * mem_used / mem_total if mem_total > 0 else 0.0)
            time.sleep(interval)
    except Exception:
        pass


def run_detection_ml_pipeline(
    dir: str = DATA_DIR,
    output_root: str = OUTPUT_ROOT,
    img_nB_range: tuple = (0, 100),
    noise_level: float = 0.594,
    noise_mode: str = "fixed",
    io_chunk_size: int = 8,
    ml_batch_size: int = 800,
    n_workers: int = 32,
    mag_zero: float = 30.0,
    pixel_scale: float = 0.2,
    psf_fwhm: float = 0.85,
    stamp_size: int = 64,
    grid_sep: int = 80,
    grid_margin: int = 40,
    shear_tasks: list = None,
    fname: str = "det_ml",
    model_path: str = MODEL_PATH,
    save_npz: bool = True,
    save_csv: bool = True,
    match_threshold: float = 3.0,
    dtype: str = "float32",
):
    if shear_tasks is None:
        shear_tasks = DEFAULT_SHEAR_TASKS

    # ---- Device & Model ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[pipeline] Using device: {device}")

    model = Forward8_fixW_CNN(num_layers=5, base_channels=32, res_factor=0.1).to(device)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print(f"[pipeline] Model loaded from {model_path}")

    calibrator = Calibrator(model, device=device, dtype=dtype)

    # ---- Noise ----
    if noise_level == "adaptive":
        folder_name = "nada"
    else:
        folder_name = format_number(noise_level)

    noise_variance = noise_level**2 if isinstance(noise_level, (int, float)) and noise_level > 0 else 1e-4

    print(f"[pipeline] Noise: {noise_level}  ->  folder: {folder_name}")
    print(f"[pipeline] FPFS noise_variance: {noise_variance:.2e}")
    print(f"[pipeline] I/O chunk: {io_chunk_size} dirs, ML batch: {ml_batch_size}")
    print(f"[pipeline] Shear tasks: {shear_tasks}, range: {img_nB_range}")
    print(f"[pipeline] Save npz={save_npz}, csv={save_csv}")

    # ---- Pre-compute grid ----
    positions_template, canvas_size = generate_grid_positions(
        nx=10, ny=10, sep=grid_sep, margin=grid_margin, stamp_size=stamp_size,
    )

    # ---- Timing accumulators ----
    is_cuda = (device == "cuda")
    t_total_io = 0.0
    t_total_cpu = 0.0
    t_total_gpu = 0.0
    t_total_save = 0.0
    n_chunks_total = 0

    # ---- Main loop with CPU-GPU pipelining ----
    #
    # Strategy: while chunk N's GPU is working, chunk N+1's CPU work
    # (IO + stitch + noise + detect/cut) runs in a background thread.
    # This overlaps CPU time with GPU time, reducing total wall time
    # by ~GPU_time_per_chunk.

    def _chunk_cpu_work(cpu_args):
        """All CPU work for one chunk: Step A (IO+stitch+noise) + Step B
        (detect + cut stamps).  Returns a dict with all data needed by the
        GPU / Save stages, plus timing info.

        Runs in a background thread during CPU-GPU pipelining.
        """
        _outer_start = cpu_args["outer_start"]
        _outer_end = cpu_args["outer_end"]
        _chunk_nB = cpu_args["chunk_nB"]
        _shear_comp = cpu_args["shear_comp"]
        _shear_mode = cpu_args["shear_mode"]
        _dir = cpu_args["dir"]
        _noise_level = cpu_args["noise_level"]
        _noise_mode = cpu_args["noise_mode"]
        _n_workers = cpu_args["n_workers"]

        _t0 = time.time()

        # ------------------------------------------------------
        # Step A: Parallel I/O
        # ------------------------------------------------------
        print(f"\n{'='*60}")
        print(f"[pipeline] I/O chunk: dirs [{_outer_start}, {_outer_end})  "
              f"({_chunk_nB} dirs, ~{_chunk_nB*100} galaxies)")
        print(f"{'='*60}")
        print("[Step A] Parallel loading via build_dataset ...")
        _t_io_start = time.time()
        _all_cutouts, _all_psfs, _all_cats, _all_noise = build_dataset(
            shear_comp=_shear_comp,
            shear_mode=_shear_mode,
            abs_shear_value=0.02,
            noise_level=_noise_level,
            noise_mode=_noise_mode,
            index_range=[_outer_start, _outer_end],
            directory=_dir,
            workers=_n_workers,
            use_threads=True,
        )
        _t_io = time.time() - _t_io_start
        _n_gal_total = _all_cutouts.shape[0]
        print(f"  Loaded {_n_gal_total} cutouts  ({_all_cutouts.shape})  [{_t_io:.1f}s]")

        # Ensure contiguous memory layout for cache-friendly stitch access
        _all_cutouts = np.ascontiguousarray(_all_cutouts)

        # Pre-stitch all large images serially to avoid memory-bandwidth contention
        _gal_per_dir = 100
        _all_large_images = []
        for _i in range(_chunk_nB):
            _s, _e = _i * _gal_per_dir, (_i + 1) * _gal_per_dir
            _all_large_images.append(
                stitch_large_image(_all_cutouts[_s:_e], positions_template, canvas_size)
            )

        # Batch noise generation: vectorised across all dirs at once (float32)
        if noise_level > 0:
            _rng_batch = np.random.default_rng(20020620 + _outer_start)
            _all_noise_canvases = _rng_batch.normal(
                0, noise_level,
                (_chunk_nB, canvas_size, canvas_size),
            ).astype(np.float32)
        else:
            _all_noise_canvases = None

        # ------------------------------------------------------
        # Step B: Per-directory detect + cut stamps (parallel)
        # ------------------------------------------------------
        _t_cpu_start = time.time()

        def _process_one_dir(i_dir):
            t0 = time.perf_counter()
            s = i_dir * _gal_per_dir
            e = s + _gal_per_dir
            cat_i = _all_cats.iloc[s:e]

            psf_i = _all_psfs[s]
            if psf_i.ndim == 3:
                psf_i = psf_i[0]

            large_i = _all_large_images[i_dir]
            t1 = time.perf_counter()

            if _all_noise_canvases is not None:
                noise_canvas_i = _all_noise_canvases[i_dir]
            else:
                noise_canvas_i = None
            t2 = time.perf_counter()

            task_i = _build_task_from_fpfs_config(
                mag_zero=mag_zero,
                pixel_scale=pixel_scale,
                sigma_arcsec=0.40,
            )
            cells_i = _build_cell_list(
                img_height=canvas_size,
                img_width=canvas_size,
                pixel_scale=pixel_scale,
            )
            t3 = time.perf_counter()

            det_i = run_detection_and_fpfs(
                large_image=large_i,
                psf_image=psf_i,
                fpfs_config=FPFS_CONFIG,
                mag_zero=mag_zero,
                pixel_scale=pixel_scale,
                noise_variance=noise_variance,
                noise_array=noise_canvas_i,
                task=task_i,
                cells=cells_i,
            )
            t4 = time.perf_counter()

            if len(det_i) == 0:
                empty_stamps = np.empty((0, stamp_size, stamp_size))
                timings = {
                    "stitch": t1 - t0, "noise": t2 - t1,
                    "build_task": t3 - t2, "detect": t4 - t3,
                    "cut": 0.0, "n_det": 0,
                }
                return i_dir, det_i, positions_template, cat_i, empty_stamps, empty_stamps, timings

            n_det = len(det_i)
            _yy_buf = np.empty((n_det, stamp_size, stamp_size), dtype=np.int32)
            _xx_buf = np.empty((n_det, stamp_size, stamp_size), dtype=np.int32)
            _sig_out = np.empty((n_det, stamp_size, stamp_size), dtype=large_i.dtype)
            _noi_out = np.empty((n_det, stamp_size, stamp_size), dtype=large_i.dtype)

            stamps_i, _ = cut_stamps_from_large(
                large_i, det_i, stamp_size=stamp_size,
                stamps_out=_sig_out, yy_buf=_yy_buf, xx_buf=_xx_buf,
            )

            if noise_canvas_i is not None:
                noise_stamps_i, _ = cut_stamps_from_large(
                    noise_canvas_i, det_i, stamp_size=stamp_size,
                    stamps_out=_noi_out, yy_buf=_yy_buf, xx_buf=_xx_buf,
                )
            else:
                noise_stamps_i = np.zeros_like(stamps_i)
            t5 = time.perf_counter()

            timings = {
                "stitch": t1 - t0, "noise": t2 - t1,
                "build_task": t3 - t2, "detect": t4 - t3,
                "cut": t5 - t4, "n_det": len(det_i),
            }
            return i_dir, det_i, positions_template, cat_i, stamps_i, noise_stamps_i, timings

        _all_det_cats = [None] * _chunk_nB
        _all_positions_list = [None] * _chunk_nB
        _all_cats_chunks = [None] * _chunk_nB
        _all_stamps_list = [None] * _chunk_nB
        _all_noise_stamps_list = [None] * _chunk_nB
        _all_timings = [None] * _chunk_nB

        with ThreadPoolExecutor(max_workers=min(_chunk_nB, 32)) as ex:
            futures = {ex.submit(_process_one_dir, i_dir): i_dir for i_dir in range(_chunk_nB)}
            for future in as_completed(futures):
                result = future.result()
                i_dir, det_i, pos_i, cat_i, stamps_i, noise_stamps_i = result[:6]
                timings_i = result[6] if len(result) > 6 else {}
                _all_det_cats[i_dir] = det_i
                _all_positions_list[i_dir] = pos_i
                _all_cats_chunks[i_dir] = cat_i
                _all_stamps_list[i_dir] = stamps_i
                _all_noise_stamps_list[i_dir] = noise_stamps_i
                _all_timings[i_dir] = timings_i

        _total_stamps = sum(len(s) for s in _all_stamps_list if s is not None)
        _t_cpu = time.time() - _t_cpu_start

        # ---- Per-directory timing summary ----
        keys = ["stitch", "noise", "build_task", "detect", "cut"]
        for k in keys:
            vals = [t[k] for t in _all_timings if t is not None and k in t]
            if vals:
                print(f"  StepB [{k:>11s}]   min={min(vals):.3f}s  "
                      f"mean={np.mean(vals):.3f}s  max={max(vals):.3f}s  "
                      f"(n={len(vals)})")
        n_det_vals = [t["n_det"] for t in _all_timings if t is not None and "n_det" in t]
        if n_det_vals:
            print(f"  StepB [     n_det]   min={min(n_det_vals):.0f}  "
                  f"mean={np.mean(n_det_vals):.1f}  max={max(n_det_vals):.0f}")
        print(f"  StepB total: {_chunk_nB} dirs, "
              f"{_total_stamps} stamps  [{_t_cpu:.1f}s]")

        _t_elapsed_cpu = time.time() - _t0
        print(f"  [CPU prep done in {_t_elapsed_cpu:.1f}s, "
              f"waiting for GPU pipeline slot]")

        return {
            "outer_start": _outer_start,
            "outer_end": _outer_end,
            "chunk_nB": _chunk_nB,
            "t_io": _t_io,
            "t_cpu": _t_cpu,
            "total_stamps": _total_stamps,
            "all_stamps_list": _all_stamps_list,
            "all_noise_stamps_list": _all_noise_stamps_list,
            "all_det_cats": _all_det_cats,
            "all_positions_list": _all_positions_list,
            "all_cats_chunks": _all_cats_chunks,
            "all_psfs": _all_psfs,
            "all_cats": _all_cats,
            "all_cutouts": _all_cutouts,
            "all_noise": _all_noise,
        }

    # ------------------------------------------------------------------
    # Pipeline loop: for each shear task, process chunks with CPU-GPU overlap
    # ------------------------------------------------------------------
    for shear_mode, shear_comp in shear_tasks:
        chunk_starts = list(range(img_nB_range[0], img_nB_range[1], io_chunk_size))

        with ThreadPoolExecutor(max_workers=1) as prefetch_exec:
            cpu_future = None  # Future holding NEXT chunk's pre-fetched CPU data

            for i_chunk, outer_start in enumerate(chunk_starts):
                outer_end = min(outer_start + io_chunk_size, img_nB_range[1])
                chunk_nB = outer_end - outer_start
                t_chunk = time.time()

                # ====================================================
                # Step A+B: Obtain CPU data for this chunk
                # ====================================================
                if cpu_future is not None:
                    # Consume pre-fetched CPU data from background thread
                    cpu_data = cpu_future.result()
                    outer_start = cpu_data["outer_start"]
                    outer_end = cpu_data["outer_end"]
                    chunk_nB = cpu_data["chunk_nB"]
                    t_io = cpu_data["t_io"]
                    t_cpu = cpu_data["t_cpu"]
                    total_stamps = cpu_data["total_stamps"]
                    all_stamps_list = cpu_data["all_stamps_list"]
                    all_noise_stamps_list = cpu_data["all_noise_stamps_list"]
                    all_det_cats = cpu_data["all_det_cats"]
                    all_positions_list = cpu_data["all_positions_list"]
                    all_cats_chunks = cpu_data["all_cats_chunks"]
                    all_psfs = cpu_data["all_psfs"]
                    all_cats = cpu_data["all_cats"]
                    all_cutouts = cpu_data["all_cutouts"]
                    all_noise = cpu_data["all_noise"]
                    cpu_future = None
                else:
                    # First chunk: compute CPU inline
                    cpu_data = _chunk_cpu_work({
                        "outer_start": outer_start,
                        "outer_end": outer_end,
                        "chunk_nB": chunk_nB,
                        "shear_comp": shear_comp,
                        "shear_mode": shear_mode,
                        "dir": dir,
                        "noise_level": noise_level,
                        "noise_mode": noise_mode,
                        "n_workers": n_workers,
                    })
                    t_io = cpu_data["t_io"]
                    t_cpu = cpu_data["t_cpu"]
                    total_stamps = cpu_data["total_stamps"]
                    all_stamps_list = cpu_data["all_stamps_list"]
                    all_noise_stamps_list = cpu_data["all_noise_stamps_list"]
                    all_det_cats = cpu_data["all_det_cats"]
                    all_positions_list = cpu_data["all_positions_list"]
                    all_cats_chunks = cpu_data["all_cats_chunks"]
                    all_psfs = cpu_data["all_psfs"]
                    all_cats = cpu_data["all_cats"]
                    all_cutouts = cpu_data["all_cutouts"]
                    all_noise = cpu_data["all_noise"]

                # ====================================================
                # Launch CPU pre-fetch for NEXT chunk (if any)
                # This runs in background WHILE current chunk's GPU works
                # ====================================================
                if i_chunk + 1 < len(chunk_starts):
                    _next_start = chunk_starts[i_chunk + 1]
                    _next_end = min(_next_start + io_chunk_size, img_nB_range[1])
                    _next_nB = _next_end - _next_start
                    cpu_future = prefetch_exec.submit(_chunk_cpu_work, {
                        "outer_start": _next_start,
                        "outer_end": _next_end,
                        "chunk_nB": _next_nB,
                        "shear_comp": shear_comp,
                        "shear_mode": shear_mode,
                        "dir": dir,
                        "noise_level": noise_level,
                        "noise_mode": noise_mode,
                        "n_workers": n_workers,
                    })

                # ====================================================
                # Step C: Big-batch ML inference
                # ====================================================
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

                    psf_ref = all_psfs[0]
                    if psf_ref.ndim == 3:
                        psf_ref = psf_ref[0]

                    # Concatenate pre-generated noise stamps (same as detection used)
                    noise_stamps_all = np.concatenate(
                        [n for n in all_noise_stamps_list if len(n) > 0], axis=0
                    )

                    # GPU utilization monitor (background polling)
                    gpu_mon = {"util_samples": [], "mem_util_samples": [],
                               "mem_frac_samples": [], "stop": False}
                    gpu_thread = None
                    if is_cuda:
                        gpu_thread = threading.Thread(
                            target=_gpu_util_monitor, args=(0.5, gpu_mon), daemon=True
                        )
                        gpu_thread.start()

                    ml_shapes_all, ml_R_all = ml_shape_measure(
                        stamps=stamps_all,
                        psf_image=psf_ref,
                        calibrator=calibrator,
                        pixel_scale=pixel_scale,
                        psf_fwhm=psf_fwhm,
                        noise_arrays=noise_stamps_all,
                        n_workers=n_workers,
                        dtype=dtype,
                        ml_batch_size=ml_batch_size,
                        seed=20020620 + outer_start,
                    )
                    if is_cuda:
                        torch.cuda.synchronize()
                    t_ml_total = time.time() - t_gpu_start
                    # Stop monitor
                    gpu_mon["stop"] = True
                    if gpu_thread is not None:
                        gpu_thread.join(timeout=2)
                    gpu_util = np.mean(gpu_mon["util_samples"]) if gpu_mon["util_samples"] else 0.0
                    gpu_mem_util = np.mean(gpu_mon["mem_util_samples"]) if gpu_mon["mem_util_samples"] else 0.0
                    gpu_mem_frac = np.mean(gpu_mon["mem_frac_samples"]) if gpu_mon["mem_frac_samples"] else 0.0
                    # GPU memory stats from PyTorch
                    gpu_mem = ""
                    if is_cuda:
                        gpu_mem = (f", GPU mem: "
                                   f"{torch.cuda.max_memory_allocated()/1024**3:.2f} GB peak, "
                                   f"{torch.cuda.memory_allocated()/1024**3:.2f} GB current")
                        torch.cuda.reset_peak_memory_stats()
                    print(f"  ML shapes: {ml_shapes_all.shape}, R: {ml_R_all.shape}")
                    print(f"  [Step C] ML total: {t_ml_total:.1f}s  "
                          f"(see breakdown above ^^^)  "
                          f"GPU util: {gpu_util:.0f}%, "
                          f"GPU mem util: {gpu_mem_util:.0f}%, "
                          f"GPU mem cap: {gpu_mem_frac:.0f}%{gpu_mem}")
                    # Rename for accumulator (still used in final summary)
                    t_gpu = t_ml_total
                else:
                    ml_shapes_all = np.empty((0, 2))
                    ml_R_all = np.empty((0, 1, 2, 2))
                    t_gpu = 0.0

                # ====================================================
                # Step D: Split & save per-directory catalogs
                # ====================================================
                t_save_start = time.time()
                ml_offset = 0
                n_total_saved = 0

                for i_dir in range(chunk_nB):
                    dir_idx = outer_start + i_dir
                    det_i = all_det_cats[i_dir]
                    n_det_i = len(det_i)

                    if n_det_i > 0:
                        ml_s_i = ml_shapes_all[ml_offset:ml_offset + n_det_i]
                        ml_R_i = ml_R_all[ml_offset:ml_offset + n_det_i]
                        ml_offset += n_det_i
                        n_total_saved += n_det_i
                    else:
                        ml_s_i = np.empty((0, 2))
                        ml_R_i = np.empty((0, 1, 2, 2))

                    save_dir = os.path.join(
                        output_root, folder_name,
                        f"{shear_comp}_{shear_mode}", fname,
                    )

                    merge_and_save_catalog(
                        det_cat=det_i,
                        ml_shapes=ml_s_i,
                        ml_R=ml_R_i,
                        positions=all_positions_list[i_dir],
                        output_dir=save_dir,
                        truth_cat=all_cats_chunks[i_dir],
                        fname=f"catalog_{dir_idx}",
                        match_threshold=match_threshold,
                        save_npz=save_npz,
                        save_csv=save_csv,
                        verbose=False,
                    )

                # --- Cleanup ---
                t_save = time.time() - t_save_start
                del all_cutouts, all_psfs, all_cats, all_noise
                del all_det_cats, all_positions_list, all_cats_chunks
                del all_stamps_list, all_noise_stamps_list
                if total_stamps > 0:
                    del stamps_all, ml_shapes_all, ml_R_all

                # --- Accumulate timings ---
                t_total_io += t_io
                t_total_cpu += t_cpu
                t_total_gpu += t_gpu
                t_total_save += t_save
                n_chunks_total += 1

                t_elapsed = time.time() - t_chunk
                print(f"[pipeline] Chunk done — {n_total_saved} stamps saved in {t_save:.1f}s  "
                      f"| wall {t_elapsed:.1f}s  "
                      f"I/O={t_io:.1f}s  CPU={t_cpu:.1f}s  ML_total={t_gpu:.1f}s  Save={t_save:.1f}s")

                # Flush print buffer so log tail is current
                import sys; sys.stdout.flush()

    # ---- Grand total ----
    t_total = t_total_io + t_total_cpu + t_total_gpu + t_total_save
    print(f"\n{'='*60}")
    print(f"[pipeline] All batches finished.")
    print(f"[pipeline] === Timing Summary ({n_chunks_total} chunks) ===")
    print(f"  Total wall time: {t_total:.1f}s ({t_total/60:.1f} min)")
    print(f"  I/O:      {t_total_io:8.1f}s  ({100*t_total_io/max(t_total,1):5.1f}%)")
    print(f"  CPU:      {t_total_cpu:8.1f}s  ({100*t_total_cpu/max(t_total,1):5.1f}%)")
    print(f"  ML total: {t_total_gpu:8.1f}s  ({100*t_total_gpu/max(t_total,1):5.1f}%)")
    print(f"  Save:     {t_total_save:8.1f}s  ({100*t_total_save/max(t_total,1):5.1f}%)")
    if n_chunks_total > 0 and t_total_gpu > 0:
        print(f"  Note: 'ML total' includes prepare_q_images (CPU) + shape_measure (GPU).")
        print(f"  See '[ml_shape_measure timing]' above for the per-chunk breakdown.")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    start_time = time.time()

    p = argparse.ArgumentParser(
        description="FPFS Detection + ML Shape Measurement — Optimized Batch Pipeline"
    )
    p.add_argument("--noise", type=float, required=True)
    p.add_argument("--fname", type=str, default="det_ml")
    p.add_argument("--shear_tasks", nargs="+", type=str,
                   default=["0_g1", "1_g1", "0_g2", "1_g2"])
    p.add_argument("--range", nargs=2, type=int, default=[0, 100])
    p.add_argument("--model", type=str, default=MODEL_PATH)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--io_chunk", type=int, default=8,
                   help="Directories per I/O chunk (default 8 → ~800 galaxies)")
    p.add_argument("--ml_batch", type=int, default=800,
                   help="ML inference batch size")
    p.add_argument("--grid_sep", type=int, default=80)
    p.add_argument("--grid_margin", type=int, default=40,
                   help="Canvas margin (pixels). 40 makes 864 canvas, more FFT-friendly.")
    p.add_argument("--no_csv", action="store_true")
    p.add_argument("--no_npz", action="store_true")

    args = p.parse_args()

    run_detection_ml_pipeline(
        dir=DATA_DIR,
        output_root=OUTPUT_ROOT,
        img_nB_range=tuple(args.range),
        noise_level=args.noise,
        n_workers=args.workers,
        io_chunk_size=args.io_chunk,
        ml_batch_size=args.ml_batch,
        shear_tasks=parse_shear_tasks(args.shear_tasks),
        fname=args.fname,
        model_path=args.model,
        grid_sep=args.grid_sep,
        grid_margin=args.grid_margin,
        save_npz=not args.no_npz,
        save_csv=not args.no_csv,
    )

    print(f"[main] Total time: {time.time() - start_time:.1f}s "
          f"({(time.time() - start_time)/60:.1f} min)")
