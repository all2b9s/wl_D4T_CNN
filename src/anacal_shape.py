#!/usr/bin/env python3
"""FPFS measurement entry point.

The original mode measures the historical single-galaxy simulation layout.
The ``--isolated-dir`` mode measures the noiseless matched-isolated shards
written by ``render_matched_isolated.py`` at their known stamp centres.  The
latter deliberately does *not* run detection again: it is the same detected
object that supplied the blended postage stamp.
"""

import argparse
import os
from pathlib import Path
from time import perf_counter

# These must be set before numpy/scipy/anacal are imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import anacal
import numpy as np


FPFS_CONFIG = anacal.fpfs.FpfsConfig(
    sigma_shapelets1=0.45,
    sigma_shapelets2=0.55,
)


def parse_shear_tasks(task_list):
    """Parse ``['0_g1', '1_g2']`` into ``[(0, 'g1'), (1, 'g2')]``."""
    parsed = []
    for task in task_list:
        try:
            index, shear = task.split("_")
            parsed.append((int(index), shear))
        except ValueError as error:
            raise ValueError(f"Invalid shear task '{task}', expected '0_g1'") from error
    return parsed


def _fractional_pixel_coordinate(coordinate: np.ndarray) -> np.ndarray:
    """Return the offset from the nearest integer pixel centre.

    The isolated renderer centres each stamp on the matched detection's
    *integer* pixel, but retains the truth catalogue's sub-pixel position.
    Using this offset for forced FPFS is essential: measuring every stamp at
    (32, 32) would introduce a coherent pixel-phase error.
    """
    coordinate = np.asarray(coordinate, dtype=np.float64)
    return coordinate - np.floor(coordinate + 0.5)


def _measure_grid_batch(
    cutouts: np.ndarray,
    frac_y: np.ndarray,
    frac_x: np.ndarray,
    psf: np.ndarray,
    noise_variance: float,
    grid_spacing: int,
) -> np.ndarray:
    """Forced-measure a sparse grid of stamps in one current-AnaCal call.

    ``fpfs.process_image`` in the current pipeline accepts many forced
    detection coordinates per image.  Packing stamps with a blank moat avoids
    cross-talk while replacing thousands of Python/C++ calls per tract with a
    handful of vectorised calls.
    """
    n_source, height, width = cutouts.shape
    if height != FPFS_CONFIG.npix or width != FPFS_CONFIG.npix:
        raise ValueError(
            f"Expected {FPFS_CONFIG.npix}x{FPFS_CONFIG.npix} stamps, got "
            f"{height}x{width}"
        )
    if grid_spacing < max(height, width):
        raise ValueError("--grid-spacing must be at least the stamp width")

    ncol = int(np.ceil(np.sqrt(n_source)))
    nrow = int(np.ceil(n_source / ncol))
    margin = FPFS_CONFIG.npix
    canvas = np.zeros(
        (2 * margin + nrow * grid_spacing, 2 * margin + ncol * grid_spacing),
        dtype=np.float64,
    )
    detection = np.zeros(n_source, dtype=[("y", "f8"), ("x", "f8")])
    half_y, half_x = height // 2, width // 2

    for index in range(n_source):
        row, col = divmod(index, ncol)
        centre_y = margin + half_y + row * grid_spacing
        centre_x = margin + half_x + col * grid_spacing
        canvas[
            centre_y - half_y : centre_y + half_y,
            centre_x - half_x : centre_x + half_x,
        ] = cutouts[index]
        detection["y"][index] = centre_y + frac_y[index]
        detection["x"][index] = centre_x + frac_x[index]

    output = anacal.fpfs.process_image(
        fpfs_config=FPFS_CONFIG,
        mag_zero=30.0,
        gal_array=np.ascontiguousarray(canvas),
        psf_array=psf,
        pixel_scale=0.2,
        # These are noiseless label images.  ``noise_variance`` is retained
        # only for FPFS normalisation; no random noise realization is supplied.
        noise_variance=noise_variance,
        noise_array=None,
        detection=detection,
    )
    if output is None or len(output) != n_source:
        output_size = 0 if output is None else len(output)
        raise RuntimeError(
            f"Forced FPFS returned {output_size} rows for {n_source} detections"
        )
    return output


def _find_archives(input_dir: Path, exposure_range):
    paths = sorted(input_dir.glob("isolated_*.npz"))
    if exposure_range is None:
        return paths
    start, stop = exposure_range
    selected = []
    for path in paths:
        try:
            exp_id = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        if start <= exp_id < stop:
            selected.append(path)
    return selected


def _output_path(output_dir: Path, archive_path: Path) -> Path:
    exp_id = archive_path.stem.split("_")[-1]
    return output_dir / f"fpfs_isolated_{exp_id}.npz"


def measure_isolated_archive(
    archive_path: Path,
    output_dir: Path,
    noise_variance: float,
    sources_per_batch: int,
    grid_spacing: int,
    overwrite: bool,
) -> None:
    """Forced-measure one isolated render shard and preserve all FPFS columns."""
    output_path = _output_path(output_dir, archive_path)
    if output_path.exists() and not overwrite:
        print(f"[{archive_path.name}] complete; skipping")
        return

    with np.load(archive_path) as data:
        required = {"isolated_stamps", "sample_id", "psf"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{archive_path} is missing {sorted(missing)}")
        cutouts = np.asarray(data["isolated_stamps"], dtype=np.float32)
        sample_id = np.asarray(data["sample_id"], dtype=np.int64)
        psf = np.asarray(data["psf"], dtype=np.float64)
        provenance = {
            key: np.asarray(data[key])
            for key in (
                "exp_id", "det_id", "catsim_index", "det_x", "det_y",
                "truth_x", "truth_y", "match_dist_px", "i_ab", "hlr",
            )
            if key in data
        }
    if len(cutouts) != len(sample_id):
        raise ValueError(f"{archive_path}: stamps/sample_id length mismatch")
    if not len(cutouts) or cutouts.ndim != 3:
        raise ValueError(f"{archive_path}: expected a nonempty [N,H,W] stamp array, got {cutouts.shape}")

    # AnaCal's forced routine expects its configured npix.  This matches the
    # resize convention in the blended detector pipeline.
    target_shape = (FPFS_CONFIG.npix, FPFS_CONFIG.npix)
    if psf.shape != target_shape:
        psf = anacal.utils.resize_array(psf, target_shape)
    psf = np.ascontiguousarray(psf, dtype=np.float64)
    psf_sum = psf.sum()
    if not np.isfinite(psf_sum) or psf_sum <= 0:
        raise ValueError(f"{archive_path}: invalid PSF normalisation {psf_sum}")
    psf /= psf_sum

    # ``truth_x/y`` are written by render_matched_isolated.py.  Archives made
    # before that change fall back to the integer matched-detection position.
    truth_x = provenance.get("truth_x", provenance.get("det_x"))
    truth_y = provenance.get("truth_y", provenance.get("det_y"))
    if truth_x is None or truth_y is None:
        raise KeyError(f"{archive_path}: need truth_x/truth_y or det_x/det_y")
    if len(truth_x) != len(cutouts) or len(truth_y) != len(cutouts):
        raise ValueError(f"{archive_path}: truth-coordinate length mismatch")
    frac_x = _fractional_pixel_coordinate(truth_x)
    frac_y = _fractional_pixel_coordinate(truth_y)

    batches = []
    for start in range(0, len(cutouts), sources_per_batch):
        stop = min(start + sources_per_batch, len(cutouts))
        output = _measure_grid_batch(
            cutouts=np.asarray(cutouts[start:stop], dtype=np.float64),
            frac_y=frac_y[start:stop],
            frac_x=frac_x[start:stop],
            psf=psf,
            noise_variance=noise_variance,
            grid_spacing=grid_spacing,
        )
        batches.append(output)
    records = np.concatenate(batches)
    names = records.dtype.names
    if names is None:
        raise TypeError(f"{archive_path}: AnaCal returned a non-structured result")
    finite_columns = [
        name for name in ("fpfs1_e1", "fpfs1_e2", "fpfs1_de1_dg1", "fpfs1_de2_dg2")
        if name in names
    ]
    valid = np.ones(len(records), dtype=bool)
    for name in finite_columns:
        valid &= np.isfinite(records[name])

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_id": sample_id,
        "fpfs": records,
        "fpfs_valid": valid,
        "measurement_kind": np.asarray("forced_fpfs1_sparse_grid"),
        "noise_variance": np.asarray(noise_variance, dtype=np.float64),
        "pixel_scale": np.asarray(0.2, dtype=np.float32),
        **provenance,
    }
    # Expose every raw AnaCal column as an ordinary key too.  That keeps RDSM
    # code simple and retains future measurement-response information.
    payload.update({name: records[name] for name in names})
    # Stable aliases used by the downstream RDSM/training scripts.
    aliases = {
        "fpfs_e1": "fpfs1_e1",
        "fpfs_e2": "fpfs1_e2",
        "fpfs_R11": "fpfs1_de1_dg1",
        "fpfs_R22": "fpfs1_de2_dg2",
        "fpfs_m00": "fpfs1_m00",
    }
    payload.update({alias: records[source] for alias, source in aliases.items() if source in names})
    temporary = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(output_path)
    print(f"[{archive_path.name}] FPFS valid={valid.sum()}/{len(valid)} -> {output_path}")


def run_isolated_mode(args: argparse.Namespace) -> None:
    output_dir = args.output_dir or args.isolated_dir / "fpfs"
    paths = _find_archives(args.isolated_dir, args.range)
    if not paths:
        raise FileNotFoundError(f"No isolated_*.npz archives found in {args.isolated_dir}")
    for path in paths:
        measure_isolated_archive(
            path,
            output_dir,
            args.noise_variance,
            args.sources_per_batch,
            args.grid_spacing,
            args.overwrite,
        )


def run_legacy_mode(args: argparse.Namespace) -> None:
    """Keep the former one-off calibration invocation available."""
    if args.noise is None:
        raise ValueError("--noise is required unless --isolated-dir is supplied")
    from src.anacal.fpfs import fpfs_measure
    from src.anacal.cal_toolkit import bootstrap_e_std

    shear_task = parse_shear_tasks(args.shear_tasks)[0]
    started = perf_counter()
    e_fpfs, response, _, _, weight, dweight = fpfs_measure(
        dir=args.legacy_dir, img_nB_range=[0, 2], imgBs=100,
        noise_level=args.noise, n_workers=args.workers, center=[32, 32],
        shear_task=shear_task, fname=f"fpfs_calibration_{args.noise}",
        if_return=True, do_detection=True,
    )
    np.save(f"fpfs_calibration_{args.noise}_bootstrap.npy", (e_fpfs, response))
    np.save(f"fpfs_calibration_{args.noise}_w.npy", weight)
    np.save(f"fpfs_calibration_{args.noise}_dw.npy", dweight)
    mean, error, _ = bootstrap_e_std(np.asarray(e_fpfs), np.asarray(response), n_boot=100)
    elapsed = perf_counter() - started
    print(f"FPFS mean ellipticity: {mean}, std: {error} for noise={args.noise}")
    print(f"Total runtime: {elapsed:.2f}s ({elapsed / 60:.2f}min)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isolated-dir", type=Path, help="Directory containing isolated_XXXXX.npz")
    parser.add_argument("--output-dir", type=Path, help="Default: <isolated-dir>/fpfs")
    parser.add_argument("--range", type=int, nargs=2, metavar=("START", "STOP"))
    parser.add_argument(
        "--workers", type=int, default=32,
        help="Legacy calibration worker count (isolated sparse-grid mode is single-process).",
    )
    parser.add_argument(
        "--sources-per-batch", type=int, default=256,
        help="Number of isolated stamps packed into each forced-FPFS grid.",
    )
    parser.add_argument(
        "--grid-spacing", type=int, default=96,
        help="Centre-to-centre spacing (pixels) of stamps in a sparse FPFS grid.",
    )
    parser.add_argument(
        "--noise-variance", type=float, default=1e-4,
        help="Finite variance required by FPFS normalisation; no noise is added to labels.",
    )
    parser.add_argument("--overwrite", action="store_true")
    # Legacy arguments.
    parser.add_argument("--noise", type=float, help="Legacy calibration noise level")
    parser.add_argument("--legacy-dir", default="/work/nvme/bfmo/wenyinli/datasets/xlens_shift")
    parser.add_argument("--shear-tasks", nargs="+", default=["0_g1", "1_g1", "0_g2", "1_g2"])
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.noise_variance <= 0:
        parser.error("--noise-variance must be positive")
    if args.sources_per_batch <= 0:
        parser.error("--sources-per-batch must be positive")
    if args.grid_spacing < FPFS_CONFIG.npix:
        parser.error(f"--grid-spacing must be at least {FPFS_CONFIG.npix}")
    if args.range is not None and (args.range[0] < 0 or args.range[1] <= args.range[0]):
        parser.error("--range must satisfy 0 <= START < STOP")
    if args.isolated_dir is not None:
        run_isolated_mode(args)
    else:
        run_legacy_mode(args)


if __name__ == "__main__":
    main()
