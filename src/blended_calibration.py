"""
Blended Calibration — Standalone Single-Shear (g1) Calibration
===================================================================

Runs the detection + selection + shape calibration on the per-exposure
NPZ catalogs produced by ``src/blended_run.py`` (structure::

    {output_root}/{shear_comp}_{shear_mode}/{fname}/catalog_{idx}.npz
    e.g. .../blended_sims/g1_0/blended_det_fpfs_truth_s4/catalog_0.npz

Only g1 is available in the blended sims, so ``shear_comps=["g1"]`` is
used; the g2 component of the returned biases is NaN.

Run as an INDEPENDENT job AFTER the measurement pipeline jobs finish
(needs both ``g1_0`` and ``g1_1`` outputs present under ``output_root``).
``blended_run.sh`` submits this job with an ``afterok`` dependency on the
per-mode measurement jobs.

Usage (run from the repository root so that ``src`` is importable)::

    python src/blended_calibration.py \
        --output_root /work/hdd/bfmo/wenyinli/measurement/blended_sims \
        --fname blended_det_fpfs_truth_s4 \
        --range 0 10000 \
        --shape_mode ml --bs_times 100
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
import argparse

# Default output root (same as src/blended_run.py OUTPUT_ROOT)
OUTPUT_ROOT = "/work/hdd/bfmo/wenyinli/measurement/blended_sims"


def run_calibration(
    output_root: str = OUTPUT_ROOT,
    fname: str = "det_ml",
    index_range: tuple = (0, 100),
    shape_mode: str = "ml",
    mag_cut: float = None,
    include_detection_response: bool = True,
    bs_times: int = 100,
    workers: int = 64,
    output_dir: str = None,
    log_csv: str = None,
):
    """Run single-shear (g1-only) detection+selection+shape calibration.

    Reads the per-exposure NPZ catalogs produced by ``src/blended_run.py``.

    Only g1 is available in the blended sims, so ``shear_comps=["g1"]``
    is used; the g2 component of the returned biases is NaN.
    ``base_dir`` points directly at ``output_root`` (no noise subfolder
    for blended data), so the noise level only feeds log lines / default
    ``output_dir`` naming (hardcoded 0.0) and ``is_twin=False`` is fixed
    to match the single-shear setup.

    Returns
    -------
    result : tuple or dict
        ``(m, c, m_err, c_err)`` for a single ``shape_mode``, or
        ``{'fpfs': ..., 'ml': ...}`` for ``shape_mode="both"``.
    """
    from src.blending.det_shape_calibration import get_det_calibration_biases

    print(f"\n{'='*70}")
    print(f"[calibration] Blended calibration (single-shear g1)")
    print(f"{'='*70}")
    print(f"  output_root   = {output_root}")
    print(f"  fname         = {fname}")
    print(f"  index_range   = {index_range}")
    print(f"  shape_mode    = {shape_mode}")
    print(f"  mag_cut       = {mag_cut}")
    print(f"  bs_times      = {bs_times}")
    print(f"  shear_comps   = ['g1']  (base_dir bypasses noise subfolder)")

    t0 = time.time()

    result = get_det_calibration_biases(
        # noise_level / is_twin are not used for blended data:
        #   - base_dir bypasses the noise-level folder, so noise_level only
        #     shows up in log lines and default output_dir naming (dummy 0.0).
        #   - is_twin=False matches the single-shear (g1-only) setup.
        noise_level=0.0,
        fname=fname,
        index_range=index_range,
        shape_mode=shape_mode,
        mag_cut=mag_cut,
        include_detection_response=include_detection_response,
        bs_times=bs_times,
        is_twin=False,
        workers=workers,
        base_dir=output_root,        # blended sims root (no noise subfolder)
        output_dir=output_dir,
        log_csv=log_csv,
        shear_comps=["g1"],         # <-- key: only g1 is available
    )

    def _print_single(label, m, c, m_err, c_err):
        print(f"\n[{label}] calibration result (ML, g1-only, {index_range}):")
        print(f"  m     = [{m[0]:.6f}, {m[1]:.6f}]   (m[1] = NaN: no g2 data)")
        print(f"  c     = [{c[0]:.6f}, {c[1]:.6f}]   (c[1] = NaN: no g2 data)")
        print(f"  m_err = [{m_err[0]:.6f}, {m_err[1]:.6f}]")
        print(f"  c_err = [{c_err[0]:.6f}, {c_err[1]:.6f}]")

    if shape_mode == "both":
        for label, res in result.items():
            m, c, m_err, c_err = res
            _print_single(label, m, c, m_err, c_err)
    else:
        m, c, m_err, c_err = result
        _print_single(shape_mode, m, c, m_err, c_err)

    print(f"\n[calibration] Total time: {time.time() - t0:.1f}s")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    start_time = time.time()

    p = argparse.ArgumentParser(
        description="Blended single-shear (g1) calibration — standalone job"
    )
    p.add_argument("--output_root", type=str, default=OUTPUT_ROOT,
                   help="Root directory of measurement NPZ catalogs")
    p.add_argument("--fname", type=str, default="det_ml",
                   help="Output subdirectory name (same as src/blended_run.py --fname)")
    p.add_argument("--range", nargs=2, type=int, default=[0, 100],
                   help="Exposure index range [start, end)")
    p.add_argument("--shape_mode", type=str, default="ml",
                   choices=["fpfs", "ml", "both"],
                   help="Shape estimator to calibrate (default: ml)")
    p.add_argument("--mag_cut", type=float, default=None,
                   help="Magnitude cut for selection response (e.g. 24.5); None = no cut")
    p.add_argument("--bs_times", type=int, default=100,
                   help="Bootstrap iterations for calibration (default 100)")
    p.add_argument("--workers", type=int, default=64,
                   help="Number of parallel workers for loading and bootstrap")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory for calibration g_all/R_all npy files")
    p.add_argument("--log_csv", type=str, default=None,
                   help="CSV file to append calibration results to")

    args = p.parse_args()

    run_calibration(
        output_root=args.output_root,
        fname=args.fname,
        index_range=tuple(args.range),
        shape_mode=args.shape_mode,
        mag_cut=args.mag_cut,
        bs_times=args.bs_times,
        workers=args.workers,
        output_dir=args.output_dir,
        log_csv=args.log_csv,
    )

    print(f"[main] Calibration total time: {time.time() - start_time:.1f}s "
          f"({(time.time() - start_time)/60:.1f} min)")
