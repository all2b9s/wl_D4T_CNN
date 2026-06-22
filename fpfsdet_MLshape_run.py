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
    - shape_measurement.py  (argparse / loop structure)
    - src/anacal/detection_ml_pipeline.py  (Phase 1-4 functions)
    - src/blending/ml_shape_measurement.py (ml_shape_measure)
"""

import numpy as np
import torch
import anacal
import argparse
import os
import time
import threading
import subprocess
import re
from tqdm import tqdm

from src.anacal.cal_toolkit import format_number, build_dataset
from src.anacal.batch_calibration import Calibrator
from src.anacal.detection_ml_pipeline import (
    generate_grid_positions,
    stitch_large_image,
    run_fpfs_detection,
    cut_stamps_from_large,
    merge_and_save_catalog,
    match_detections_to_truth,
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

DATA_DIR = "/work/nvme/bfmo/wenyinli/datasets/xlens_shift"
OUTPUT_ROOT = "/work/hdd/bfmo/wenyinli/measurement/xlens_sims"
MODEL_PATH = "./models/F8_fpfs_l5c32r01_50ep.pth"

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
    grid_margin: int = 32,
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

    # ---- Main loop ----
    for shear_mode, shear_comp in shear_tasks:
        for outer_start in range(img_nB_range[0], img_nB_range[1], io_chunk_size):
            outer_end = min(outer_start + io_chunk_size, img_nB_range[1])
            chunk_nB = outer_end - outer_start
            t_chunk = time.time()

            print(f"\n{'='*60}")
            print(f"[pipeline] I/O chunk: dirs [{outer_start}, {outer_end})  "
                  f"({chunk_nB} dirs, ~{chunk_nB*100} galaxies)")
            print(f"{'='*60}")

            # ========================================================
            # Step A: Parallel I/O
            # ========================================================
            print("[Step A] Parallel loading via build_dataset ...")
            t_io_start = time.time()
            all_cutouts, all_psfs, all_cats, all_noise = build_dataset(
                shear_comp=shear_comp,
                shear_mode=shear_mode,
                abs_shear_value=0.02,
                noise_level=noise_level,
                noise_mode=noise_mode,
                index_range=[outer_start, outer_end],
                directory=dir,
                workers=n_workers,
                use_threads=False,
            )
            t_io = time.time() - t_io_start
            n_gal_total = all_cutouts.shape[0]
            print(f"  Loaded {n_gal_total} cutouts  ({all_cutouts.shape})  [{t_io:.1f}s]")

            # ========================================================
            # Step B: Per-directory stitch + detect + cut stamps
            # ========================================================
            gal_per_dir = 100
            t_cpu_start = time.time()

            all_det_cats = []
            all_positions_list = []
            all_cats_chunks = []
            all_stamps_list = []
            all_noise_levels_list = []  # ← per-stamp noise sigma

            for i_dir in range(chunk_nB):
                s = i_dir * gal_per_dir
                e = s + gal_per_dir
                cutouts_i = all_cutouts[s:e]
                cat_i = all_cats.iloc[s:e]
                noise_i = all_noise[s:e]       # per-galaxy noise levels in this dir

                psf_i = all_psfs[s]
                if psf_i.ndim == 3:
                    psf_i = psf_i[0]

                large_i = stitch_large_image(cutouts_i, positions_template, canvas_size)

                det_i = run_fpfs_detection(
                    large_image=large_i,
                    psf_image=psf_i,
                    fpfs_config=FPFS_CONFIG,
                    mag_zero=mag_zero,
                    pixel_scale=pixel_scale,
                    noise_variance=noise_variance,
                    noise_array=None,
                )

                if len(det_i) == 0:
                    all_det_cats.append(det_i)
                    all_positions_list.append(positions_template)
                    all_cats_chunks.append(cat_i)
                    all_stamps_list.append(np.empty((0, stamp_size, stamp_size)))
                    all_noise_levels_list.append(np.empty((0,)))
                    del large_i
                    continue

                stamps_i, offsets_i = cut_stamps_from_large(
                    large_i, det_i, stamp_size=stamp_size,
                )

                # Match detections to truth → per-stamp noise levels
                truth_idx_i, _ = match_detections_to_truth(
                    det_i, positions_template, cat_i,
                    match_threshold=match_threshold,
                )
                # For matched detections, use the source galaxy's noise;
                # for unmatched, use the median noise of this directory.
                n_det_i = len(det_i)
                stamp_noise_i = np.full(n_det_i, np.median(noise_i), dtype=np.float64)
                matched = truth_idx_i >= 0
                if np.any(matched):
                    stamp_noise_i[matched] = noise_i[truth_idx_i[matched]]

                all_det_cats.append(det_i)
                all_positions_list.append(positions_template)
                all_cats_chunks.append(cat_i)
                all_stamps_list.append(stamps_i)
                all_noise_levels_list.append(stamp_noise_i)

                del large_i

            total_stamps = sum(len(s) for s in all_stamps_list)
            n_nonempty = sum(1 for s in all_stamps_list if len(s) > 0)
            t_cpu = time.time() - t_cpu_start
            print(f"  Stitched & detected {chunk_nB} dirs: "
                  f"{total_stamps} stamps ({n_nonempty} non-empty)  [{t_cpu:.1f}s]")

            # ========================================================
            # Step C: Big-batch ML inference
            # ========================================================
            if total_stamps > 0:
                print(f"[Step C] ML measurement on {total_stamps} stamps "
                      f"(batch_size={ml_batch_size}) ...")
                if is_cuda:
                    torch.cuda.synchronize()
                t_gpu_start = time.time()

                stamps_all = np.concatenate(
                    [s for s in all_stamps_list if len(s) > 0], axis=0
                )

                psf_ref = all_psfs[0]
                if psf_ref.ndim == 3:
                    psf_ref = psf_ref[0]

                # Concatenate per-stamp noise levels
                noise_levels_all = np.concatenate(
                    [n for n in all_noise_levels_list if len(n) > 0], axis=0
                )

                # GPU utilization monitor (background polling) — start BEFORE ML inference
                gpu_mon = {"util_samples": [], "mem_util_samples": [], "mem_frac_samples": [], "stop": False}
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
                    noise_levels=noise_levels_all,
                    n_workers=n_workers,
                    dtype=dtype,
                    ml_batch_size=ml_batch_size,
                    seed=20020620 + outer_start,
                )
                print(f"  ML shapes: {ml_shapes_all.shape}, R: {ml_R_all.shape}")
                if is_cuda:
                    torch.cuda.synchronize()
                t_gpu = time.time() - t_gpu_start
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
                print(f"  GPU time: {t_gpu:.1f}s, util: {gpu_util:.0f}%, "
                      f"mem util: {gpu_mem_util:.0f}%, "
                      f"mem cap: {gpu_mem_frac:.0f}%{gpu_mem}")
            else:
                ml_shapes_all = np.empty((0, 2))
                ml_R_all = np.empty((0, 1, 2, 2))
                t_gpu = 0.0

            # ========================================================
            # Step D: Split & save per-directory catalogs
            # ========================================================
            print("[Step D] Merging & saving catalogs ...")
            t_save_start = time.time()
            ml_offset = 0

            for i_dir in range(chunk_nB):
                dir_idx = outer_start + i_dir
                det_i = all_det_cats[i_dir]
                n_det_i = len(det_i)

                if n_det_i > 0:
                    ml_s_i = ml_shapes_all[ml_offset:ml_offset + n_det_i]
                    ml_R_i = ml_R_all[ml_offset:ml_offset + n_det_i]
                    ml_offset += n_det_i
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
                )

            # --- Cleanup ---
            t_save = time.time() - t_save_start
            del all_cutouts, all_psfs, all_cats, all_noise
            del all_det_cats, all_positions_list, all_cats_chunks, all_stamps_list, all_noise_levels_list
            if total_stamps > 0:
                del stamps_all, ml_shapes_all, ml_R_all

            # --- Accumulate timings ---
            t_total_io += t_io
            t_total_cpu += t_cpu
            t_total_gpu += t_gpu
            t_total_save += t_save
            n_chunks_total += 1

            t_elapsed = time.time() - t_chunk
            print(f"[pipeline] I/O chunk done in {t_elapsed:.1f}s "
                  f"({t_elapsed/60:.1f} min)")
            print(f"  Breakdown: I/O={t_io:.1f}s ({100*t_io/max(t_elapsed,0.001):.0f}%)  "
                  f"CPU={t_cpu:.1f}s ({100*t_cpu/max(t_elapsed,0.001):.0f}%)  "
                  f"GPU={t_gpu:.1f}s ({100*t_gpu/max(t_elapsed,0.001):.0f}%)  "
                  f"Save={t_save:.1f}s ({100*t_save/max(t_elapsed,0.001):.0f}%)")

    # ---- Grand total ----
    t_total = t_total_io + t_total_cpu + t_total_gpu + t_total_save
    print(f"\n{'='*60}")
    print(f"[pipeline] All batches finished.")
    print(f"[pipeline] === Timing Summary ({n_chunks_total} chunks) ===")
    print(f"  Total wall time: {t_total:.1f}s ({t_total/60:.1f} min)")
    print(f"  I/O:     {t_total_io:8.1f}s  ({100*t_total_io/max(t_total,1):5.1f}%)")
    print(f"  CPU:     {t_total_cpu:8.1f}s  ({100*t_total_cpu/max(t_total,1):5.1f}%)")
    print(f"  GPU:     {t_total_gpu:8.1f}s  ({100*t_total_gpu/max(t_total,1):5.1f}%)")
    print(f"  Save:    {t_total_save:8.1f}s  ({100*t_total_save/max(t_total,1):5.1f}%)")
    if n_chunks_total > 0 and t_total_gpu > 0:
        print(f"  Avg GPU utilization (GPU time / wall time): "
              f"{100*t_total_gpu/t_total:.1f}%")
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
    p.add_argument("--grid_margin", type=int, default=32)
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
