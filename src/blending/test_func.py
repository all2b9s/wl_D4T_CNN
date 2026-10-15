"""
Sigma-sweep test utilities (``Forward8_fixW_CNN`` ``nl_sigma`` / ``gw_sigma``)
=============================================================================

Three groups of tools, all driven by ``scripts/sigma_test/run_sigma_test.py``:

1. Sigma-aware measurement
   ``build_cnn_from_run_dir`` rebuilds the *exact* architecture that
   ``src/train_blended_rdsm.py`` trained (reading ``config.json``), so a
   checkpoint can never be measured with the wrong ``nl_sigma``/``gw_sigma``.
   (``src/blended_run.py`` hardcodes ``nl_sigma=gw_sigma=4`` at model build
   time, which silently mismatches any other training run.)  The
   ``measure_catalogs_with_run`` driver then replays the blended detection +
   ML measurement sequence with that model, reusing
   ``src.blended_run._process_one_exposure`` for the CPU half (exposure load,
   renoise canvas, FPFS detection, weight/edge selection, stamp cut) so the
   per-exposure catalogs stay layout- and value-compatible with
   ``src/blended_run.py`` output:

       {output_root}/{shear_comp}_{shear_mode}/{fname}/catalog_{index}.npz

2. Shape noise
   ``accumulate_shape_noise`` -> per-exposure sufficient statistics;
   ``shape_noise_from_stats`` -> ``std(e) / |mean(R)|`` (the statistic used in
   ``blending_set_test.ipynb``); ``bootstrap_shape_noise`` -> exposure
   bootstrap; ``compare_shape_noise`` -> multi-model table with a paired
   ML/FPFS ratio.

3. Pair ellipticity correlation
   ``pair_ellipticity_correlation`` bins the pair products ``e_a(i) e_b(j)`` by
   separation inside every exposure (``scipy.spatial.cKDTree``), including one
   un-binned all-pairs average per exposure; ``summarize_pair_correlation``
   turns the per-exposure sums into ``xi(r)`` with an exposure-scatter error
   bar.  Correlating ML against the intrinsic (truth) ellipticities measures
   the blending-induced *measurement* correlation, since distinct galaxies'
   intrinsic shapes are uncorrelated.

Conventions
-----------
* ``truth_e1``/``truth_e2`` in the catalogs are the *applied shear*
  (``gamma1``/``gamma2`` aliases), NOT an intrinsic ellipticity.  The intrinsic
  bulge+disk ellipticity is ``truth_e1_weighted``/``truth_e2_weighted`` (NaN
  for unmatched detections), which is the reference used here.
* ``fpfs_w`` is the detection/selection weight; the weighted variant of a
  signal is ``w * e`` and of the response ``w * R + (dw/dg) * e``, matching
  ``blending_set_test.ipynb`` and ``src/blending/det_shape_calibration.py``.
  Pair weights are therefore the products ``w_i * w_j``.
* Importing ``src.blended_run`` requires the directory holding ``src`` on
  ``sys.path``; ``ensure_code_dir_on_path`` takes care of that.

All lengths are in image pixels unless a name says ``arcsec``.
"""

from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

# src/blending/test_func.py -> parents[2] = codes/single_galaxies, [4] = repo root
CODE_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_PSF_PATH = CODE_DIR / "logs/blended_test_psf.npy"
DEFAULT_CAT_REF_PATH = PROJECT_ROOT / "codes/data/catsim-v4/OneDegSq.fits"
DEFAULT_BLENDED_DIR = "/work/hdd/bfcn/wenyinli/blended_shapenoise_fullIMG"
DEFAULT_OUTPUT_ROOT = "/work/hdd/bfmo/wenyinli/measurement/blended_sims"

#: Job-name prefix used by ``scripts/sigma_test/shape_noise_sigma_sweep.sh``.
SWEEP_JOB_PREFIX = "blended_det_rdsm_sweep_s"

#: ``config.json`` keys forwarded to ``Forward8_fixW_CNN``.
MODEL_KWARG_KEYS = ("in_dim", "base_channels", "head_hidden", "out_dim",
                    "num_layers", "res_factor", "nl_sigma", "gw_sigma", "shared_head")
#: Keys that must be present, otherwise the architecture cannot be verified.
REQUIRED_MODEL_KEYS = ("nl_sigma", "gw_sigma")

#: Catalog columns per shape method.  ``R=None`` means no response is stored.
METHOD_COLUMNS = {
    "ml": {"e": ("ml_e1", "ml_e2"), "R": ("ml_R11", "ml_R22")},
    "fpfs": {"e": ("fpfs_e1", "fpfs_e2"), "R": ("fpfs_R11", "fpfs_R22")},
    "truth": {"e": ("truth_e1_weighted", "truth_e2_weighted"), "R": None},
}
DEFAULT_METHODS = ("ml", "fpfs")
DEFAULT_CORRELATION_METHODS = ("ml", "fpfs", "truth")

#: Pair-product components accumulated by ``pair_ellipticity_correlation``.
PAIR_COMPONENTS = ("e1e1", "e2e2", "e1e2")

#: Default separation bins in pixels (0.2 arcsec/pixel -> up to 12.8 arcsec).
DEFAULT_BINS_PX = np.array([0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0], dtype=np.float64)


def ensure_code_dir_on_path(code_dir: Path | str = CODE_DIR) -> str:
    """Put ``codes/single_galaxies`` first on ``sys.path`` so ``src`` imports work."""
    code_dir = str(Path(code_dir).resolve())
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    return code_dir


# ===========================================================================
# 1. Sigma-aware model construction
# ===========================================================================

def read_run_config(run_dir: Path | str) -> dict:
    """Load ``{run_dir}/config.json`` written by ``src/train_blended_rdsm.py``."""
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"No config.json in run dir: {run_dir}")
    return json.loads(config_path.read_text())


def model_kwargs_from_config(config: Mapping, source: str = "config.json") -> dict:
    """Extract the ``Forward8_fixW_CNN`` kwargs, requiring the sigma keys.

    Raises ``ValueError`` when ``nl_sigma``/``gw_sigma`` are absent: such a run
    predates sigma support, so guessing the architecture would silently produce
    a wrong measurement (exactly the ``blended_run.py`` failure mode).
    """
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise ValueError(f"{source}: missing 'model' block")
    model_class = config.get("model_class", "Forward8_fixW_CNN")
    if model_class not in ("Forward8_fixW_CNN", "Forward8_poolmix_CNN",
                            "Forward8_AdaSigma_CNN", "Forward8_AttnW_CNN"):
        raise ValueError(f"Unsupported model class: {model_class}")
    poolmix = model_class == "Forward8_poolmix_CNN"
    adaptive_keys = {
        "Forward8_AdaSigma_CNN": ("sigma_init", "sigma_hidden"),
        "Forward8_AttnW_CNN": ("attn_grid_size", "attn_dim", "attn_heads"),
    }
    required = (("nl_sigma", "suppress", "r_cut", "suppress_order") + adaptive_keys[model_class]
                if model_class in adaptive_keys else
                ("nl_sigma", "pool_sigmas") if poolmix else REQUIRED_MODEL_KEYS)
    missing = [key for key in required if key not in model]
    if missing:
        raise ValueError(
            f"{source} does not record {missing}; the trained architecture cannot be "
            "verified. Retrain with src/train_blended_rdsm.py --nl-sigma/--gw-sigma, "
            "which writes both sigmas into config.json."
        )
    keys = tuple(k for k in MODEL_KWARG_KEYS if k != "gw_sigma") + ("pool_sigmas", "suppress", "r_cut", "suppress_order", "adaptive_pool", "pool_gate_hidden") if poolmix else MODEL_KWARG_KEYS
    if model_class in adaptive_keys:
        keys = tuple(k for k in MODEL_KWARG_KEYS if k != "gw_sigma") + required
    if model_class == "Forward8_fixW_CNN":
        keys = tuple(keys) + ("suppress",)
    if model_class == 'Forward8_AttnW_CNN':
        keys = tuple(keys) + ('head_num_hidden_layers',)
    return {key: model[key] for key in keys if key in model}


def build_cnn_from_run_dir(
    run_dir: Path | str,
    weights: str = "best.pth",
    device: str | None = None,
) -> Tuple["object", dict, Path]:
    """Build the trained CNN exactly as recorded in ``{run_dir}/config.json``.

    Returns ``(model, model_kwargs, checkpoint_path)`` with the model in eval
    mode on ``device``.
    """
    ensure_code_dir_on_path()
    import torch

    from src.architecture.forward8_CNN import (Forward8_fixW_CNN, Forward8_poolmix_CNN,
                                                Forward8_AdaSigma_CNN, Forward8_AttnW_CNN)

    run_dir = Path(run_dir)
    config = read_run_config(run_dir)
    model_kwargs = model_kwargs_from_config(config, source=str(run_dir / "config.json"))
    checkpoint = run_dir / weights
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    classes = {cls.__name__: cls for cls in (Forward8_fixW_CNN, Forward8_poolmix_CNN,
                                            Forward8_AdaSigma_CNN, Forward8_AttnW_CNN)}
    model_class = classes[config.get("model_class", "Forward8_fixW_CNN")]
    model = model_class(**model_kwargs)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, model_kwargs, checkpoint


# ===========================================================================
# 2. Measurement driver
# ===========================================================================

def load_psf_image(psf_path: Path | str) -> np.ndarray:
    """Load the PSF exactly as ``src/blended_run.py`` does (even-sized trim)."""
    psf_image = np.load(psf_path).astype(np.float64)
    if psf_image.ndim == 3:
        psf_image = psf_image[0]
    if psf_image.shape[0] % 2 == 1 or psf_image.shape[1] % 2 == 1:
        psf_image = psf_image[:-1, :-1]
    return psf_image


def measure_catalogs_with_run(
    run_dir: Path | str,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    fname: Optional[str] = None,
    blended_dir: Path | str = DEFAULT_BLENDED_DIR,
    sim_mode: str = "sim_mode80",
    index_range: Tuple[int, int] = (0, 1000),
    psf_path: Path | str = DEFAULT_PSF_PATH,
    cat_ref_path: Path | str = DEFAULT_CAT_REF_PATH,
    noise_level: Optional[float] = None,
    weights: str = "best.pth",
    stamp_size: int = 64,
    pixel_scale: float = 0.2,
    psf_fwhm: float = 0.85,
    mag_zero: float = 30.0,
    snr_cut: float = 5.0,
    ml_batch_size: int = 600,
    n_workers: int = 64,
    dtype: str = "float32",
    device: Optional[str] = None,
    save_csv: bool = False,
    resume: bool = True,
    prefetch_workers: int = 4,
    progress_every: int = 25,
    verbose: bool = True,
) -> dict:
    """Measure one trained run on a blended simulation, one exposure at a time.

    Detection, weighting and stamp cutting are delegated to
    ``src.blended_run._process_one_exposure`` so every selection constant stays
    identical to ``blended_run.py``; only the model and its sigmas come from
    ``run_dir``.  The renoise canvas uses the same seed as ``blended_run.py``
    (``19491001 + exp_idx``), so ML/FPFS results are directly comparable and
    reproducible.

    Existing ``catalog_{index}.npz`` files are skipped when ``resume`` is set,
    which makes a re-submitted sbatch job cheap to restart.

    Returns a summary dict with the model kwargs, output directory, per-exposure
    manifest (as a :class:`pandas.DataFrame`) and timing totals.

    ``prefetch_workers`` runs that many exposures' CPU preparation (load, FPFS
    detection, stamp cut) concurrently in a thread pool while the GPU measures
    the exposure that just finished, mirroring the CPU/GPU overlap of
    ``blended_run.py``.  Set it to 1 for a strictly serial, easiest-to-debug run.
    """
    ensure_code_dir_on_path()
    import fitsio
    import torch

    from src.anacal.batch_calibration import Calibrator
    from src.anacal.detection_ml_pipeline import merge_and_save_catalog
    from src.blended_run import (
        EDGE_MARGIN,
        SIM_MODE_MAP,
        WEIGHT_THRESHOLD,
        _CpuPreparationConfig,
        _process_one_exposure,
    )
    from src.blending.ml_shape_measurement import ml_shape_measure

    start, stop = int(index_range[0]), int(index_range[1])
    if stop <= start:
        raise ValueError(f"Empty index range: {index_range}")
    if sim_mode not in SIM_MODE_MAP:
        raise ValueError(f"Unknown sim_mode {sim_mode!r}; known: {sorted(SIM_MODE_MAP)}")

    model, model_kwargs, checkpoint = build_cnn_from_run_dir(run_dir, weights=weights, device=device)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    sigma = float(model_kwargs["nl_sigma"])
    if fname is None:
        fname = f"{SWEEP_JOB_PREFIX}{sigma:g}"

    shear_comp, shear_mode = SIM_MODE_MAP[sim_mode]
    mode_label = f"{shear_comp}_{shear_mode}"
    out_dir = Path(output_root) / mode_label / fname
    out_dir.mkdir(parents=True, exist_ok=True)

    cat_ref = fitsio.read(str(cat_ref_path))
    psf_image = load_psf_image(psf_path)
    cpu_config = _CpuPreparationConfig(
        blended_dir=str(blended_dir),
        cat_ref=cat_ref,
        psf_image=psf_image,
        noise_mode="from_variance" if noise_level is None else "fixed",
        noise_level=noise_level,
        mag_zero=mag_zero,
        pixel_scale=pixel_scale,
        stamp_size=stamp_size,
        skip_detection=False,
        mag_cut=None,
        snr_cut=snr_cut,
        center_on_truth=False,
        match_threshold=3.0,
    )

    calibrator = Calibrator(model, device=device, dtype=dtype)

    if verbose:
        print(f"[test_func] run dir        : {Path(run_dir).resolve()}")
        print(f"[test_func] checkpoint     : {checkpoint}")
        print(f"[test_func] model kwargs   : {model_kwargs}")
        print(f"[test_func] device/dtype   : {device} / {dtype}")
        print(f"[test_func] input          : {blended_dir}/{sim_mode}")
        print(f"[test_func] exposures      : {start}..{stop}")
        print(f"[test_func] output         : {out_dir}")
        print(f"[test_func] selection      : fpfs_w >= {WEIGHT_THRESHOLD}, "
              f"edge margin {EDGE_MARGIN} px, snr_peak_min {snr_cut}, "
              f"mag_zero {mag_zero}, stamp {stamp_size} px")

    manifest: List[dict] = []
    n_new, n_stamps_total = 0, 0
    t_measure_total = 0.0
    t_wall_start = time.time()

    todo = [idx for idx in range(start, stop)
            if not (resume and (out_dir / f"catalog_{idx}.npz").is_file())]
    n_skipped = (stop - start) - len(todo)
    if verbose:
        print(f"[test_func] {len(todo)} exposures to measure "
              f"({n_skipped} already present), prefetch_workers={max(1, prefetch_workers)}")

    def prepared_exposures():
        """Yield ``(exp_idx, result)`` in ascending order, preparing ahead in threads."""
        workers = max(1, int(prefetch_workers))
        if not todo:
            return
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures: Dict[int, "object"] = {}

            def submit(position: int):
                exp = todo[position]
                return exp, pool.submit(_process_one_exposure, exp - start, start,
                                        sim_mode, cpu_config)

            cursor = 0
            while cursor < len(todo) and len(futures) < workers:
                exp, future = submit(cursor)
                futures[exp] = future
                cursor += 1
            for exp in todo:
                future = futures.pop(exp)
                yield exp, future.result()
                # Top the window back up while the caller uses the GPU.
                if cursor < len(todo):
                    new_exp, new_future = submit(cursor)
                    futures[new_exp] = new_future
                    cursor += 1

    for exp_idx, result in prepared_exposures():
        (_, exp_out, _, noise_sigma, truth_df, positions,
         det_cat, stamps, noise_stamps, timings) = result
        if exp_out != exp_idx:
            raise RuntimeError(f"Exposure bookkeeping mismatch: {exp_out} != {exp_idx}")

        n_det = len(det_cat)
        t_start = time.perf_counter()
        if n_det > 0:
            ml_shapes, ml_R = ml_shape_measure(
                stamps=np.asarray(stamps),
                psf_image=psf_image,
                calibrator=calibrator,
                pixel_scale=pixel_scale,
                psf_fwhm=psf_fwhm,
                noise_arrays=np.asarray(noise_stamps),
                n_workers=n_workers,
                dtype=dtype,
                ml_batch_size=ml_batch_size,
                seed=20020620 + exp_idx,
                verbose=False,
            )
        else:
            ml_shapes = np.empty((0, 2))
            ml_R = np.empty((0, 1, 2, 2))
        t_measure = time.perf_counter() - t_start

        merge_and_save_catalog(
            det_cat=det_cat,
            ml_shapes=ml_shapes,
            ml_R=ml_R,
            positions=positions,
            output_dir=str(out_dir),
            truth_cat=truth_df,
            fname=f"catalog_{exp_idx}",
            match_threshold=3.0,
            save_npz=True,
            save_csv=save_csv,
            verbose=False,
        )

        n_new += 1
        n_stamps_total += n_det
        t_measure_total += t_measure
        manifest.append(dict(
            exp_id=exp_idx, n_det=n_det, noise_sigma=noise_sigma,
            t_load=timings["load"], t_detect=timings["detect"], t_measure=t_measure,
        ))
        if verbose and (n_new % max(1, progress_every) == 0 or exp_idx == stop - 1):
            elapsed = time.time() - t_wall_start
            print(f"[test_func]   exp {exp_idx:05d}: {n_det:5d} detections, "
                  f"measure {t_measure:5.2f}s | done {n_new}, elapsed {elapsed/60:.1f} min")

    manifest_df = pd.DataFrame(manifest)
    if not manifest_df.empty:
        manifest_df.to_csv(out_dir / "measurement_manifest.csv", index=False)

    summary = dict(
        run_dir=str(Path(run_dir).resolve()),
        checkpoint=str(checkpoint),
        model_kwargs=model_kwargs,
        sigma=sigma,
        output_dir=str(out_dir),
        sim_mode=sim_mode,
        index_range=[start, stop],
        n_written=n_new,
        n_skipped=n_skipped,
        n_detections=n_stamps_total,
        t_measure_s=t_measure_total,
        t_wall_s=time.time() - t_wall_start,
        selection=dict(weight_threshold=WEIGHT_THRESHOLD, edge_margin=EDGE_MARGIN,
                       snr_cut=snr_cut, mag_zero=mag_zero, stamp_size=stamp_size),
    )
    (out_dir / "measurement_config.json").write_text(json.dumps(
        {k: v for k, v in summary.items() if k not in ("t_wall_s",)}, indent=2, default=str))

    if verbose:
        print(f"[test_func] wrote {n_new} catalogs ({n_skipped} skipped) "
              f"with {n_stamps_total} detections in {summary['t_wall_s']/60:.1f} min")
    return summary


# ===========================================================================
# 3. Catalog reading
# ===========================================================================

def _read_columns(saved, names: Sequence[str], path: Path) -> np.ndarray:
    missing = [name for name in names if name not in saved.files]
    if missing:
        raise KeyError(f"{path.name}: missing columns {missing}; available={sorted(saved.files)}")
    return np.column_stack([np.asarray(saved[name], dtype=np.float64) for name in names])


def truth_magnitude(
    path: Path | str,
    truth_dir: Path | str,
    sim_mode: str = "sim_mode80",
    column: str = "i_ab",
    pixel_scale: float = 0.2,
    edge_margin: float = 32.0,
    index: Optional[int] = None,
) -> np.ndarray:
    """Truth-catalog magnitude of the objects matched to one exposure's detections.

    Reproduces the truth loading of ``src/blended_run.load_blended_exposure``:
    the truth table is converted to image pixels with
    ``x = nx / 2 + dx / pixel_scale + 1500`` (``y`` likewise), objects closer
    than ``edge_margin`` px to the border are dropped, and the catalog's
    ``match_idx`` indexes rows of that inside-selected table (``-1`` =
    unmatched).  A matched detection therefore gets the magnitude of its own
    truth row; unmatched detections get NaN.

    ``column`` is the magnitude column of the truth catalog (``i_ab`` is the
    i-band AB magnitude written by ``src/blended_sim.py``).
    """
    import fitsio

    path = Path(path)
    index = int(index if index is not None else path.stem.split("_")[-1])
    sim_dir = Path(truth_dir) / sim_mode
    header = fitsio.read_header(str(sim_dir / f"exp-i-{index:05d}.fits"), ext=1)
    ny, nx = int(header["NAXIS2"]), int(header["NAXIS1"])
    truth = fitsio.read(str(sim_dir / f"truth-{index:05d}.fits"),
                        columns=["dx", "dy", column])
    x_all = nx / 2 + np.asarray(truth["dx"], dtype=np.float64) / pixel_scale + 1500.0
    y_all = ny / 2 + np.asarray(truth["dy"], dtype=np.float64) / pixel_scale + 1500.0
    inside = ((x_all >= edge_margin) & (x_all < nx - edge_margin)
              & (y_all >= edge_margin) & (y_all < ny - edge_margin))
    values = np.asarray(truth[column], dtype=np.float64)[inside]

    with np.load(path) as saved:
        match_idx = np.asarray(saved["match_idx"])
    magnitude = np.full(match_idx.shape, np.nan)
    matched = match_idx >= 0
    if matched.any():
        if int(match_idx[matched].max()) >= values.size:
            raise IndexError(
                f"{path.name}: match_idx {int(match_idx[matched].max())} outside the "
                f"{values.size} inside-selected truth rows of {sim_dir.name}")
        magnitude[matched] = values[match_idx[matched]]
    return magnitude


def load_exposure_arrays(
    path: Path | str,
    methods: Sequence[str] = DEFAULT_METHODS,
    weight_key: str = "fpfs_w",
    require_positions: bool = True,
    mag_cut: Optional[Mapping] = None,
) -> dict:
    """Read one per-exposure catalog into aligned, masked arrays.

    Returns a dict with ``path``, ``n_total``, ``mask`` (common finite mask over
    every requested method, the weight and the positions), ``det_id``, ``y``,
    ``x``, ``w``, ``match_idx``, ``match_dist`` and ``methods`` mapping each
    method name to ``{"e": (N, 2), "R": (N, 2) or None}``.

    ``mag_cut`` (a mapping with ``max`` plus the keyword arguments of
    :func:`truth_magnitude`, i.e. ``truth_dir``/``sim_mode``/``column``/...)
    additionally requires the matched truth magnitude to be finite and below
    that limit; the per-detection magnitudes are returned as ``magnitude``.
    """
    path = Path(path)
    with np.load(path) as saved:
        files = set(saved.files)
        reference = "det_id" if "det_id" in files else next(iter(saved.files))
        n_total = int(len(np.asarray(saved[reference])))

        det_id = np.asarray(saved["det_id"]) if "det_id" in files else np.arange(n_total)
        y = np.asarray(saved["y"], dtype=np.float64) if "y" in files else None
        x = np.asarray(saved["x"], dtype=np.float64) if "x" in files else None
        w = np.asarray(saved[weight_key], dtype=np.float64) if weight_key in files else None
        match_idx = np.asarray(saved["match_idx"]) if "match_idx" in files else None
        match_dist = (np.asarray(saved["match_dist"], dtype=np.float64)
                      if "match_dist" in files else None)

        mask = np.ones(n_total, dtype=bool)
        method_data: Dict[str, dict] = {}
        for name in methods:
            if name not in METHOD_COLUMNS:
                raise KeyError(f"Unknown method {name!r}; known: {sorted(METHOD_COLUMNS)}")
            columns = METHOD_COLUMNS[name]
            e = _read_columns(saved, columns["e"], path)
            if len(e) != n_total:
                raise ValueError(f"{path.name}: inconsistent length for method {name!r}")
            mask &= np.isfinite(e).all(axis=1)
            R = None
            if columns["R"] is not None:
                R = _read_columns(saved, columns["R"], path)
                mask &= np.isfinite(R).all(axis=1)
            method_data[name] = {"e": e, "R": R}

        if require_positions:
            if y is None or x is None:
                raise KeyError(f"{path.name}: positions 'y'/'x' are required")
            mask &= np.isfinite(y) & np.isfinite(x)
        if w is not None:
            mask &= np.isfinite(w)

    magnitude = None
    if mag_cut is not None:
        settings = dict(mag_cut)
        limit = float(settings.pop("max"))
        magnitude = truth_magnitude(path, **settings)
        mask &= np.isfinite(magnitude) & (magnitude < limit)

    return dict(path=path, n_total=n_total, mask=mask, det_id=det_id, y=y, x=x, w=w,
                match_idx=match_idx, match_dist=match_dist, methods=method_data,
                magnitude=magnitude)


def catalog_paths(directory: Path | str, index_range: Optional[Tuple[int, int]] = None,
                  pattern: str = "catalog_*.npz") -> List[Path]:
    """Sorted ``catalog_*.npz`` paths in ``directory``, optionally index-filtered."""
    directory = Path(directory)
    paths = sorted(directory.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No {pattern} files in {directory}")
    if index_range is None:
        return paths
    start, stop = int(index_range[0]), int(index_range[1])
    selected = [p for p in paths if start <= int(p.stem.split("_")[-1]) < stop]
    if not selected:
        raise FileNotFoundError(f"No {pattern} files in {directory} for {index_range}")
    return selected


def common_mask(catalogs: Sequence[dict]) -> np.ndarray:
    """Intersection of the per-catalog masks (same detections assumed)."""
    if not catalogs:
        raise ValueError("No catalogs given")
    mask = catalogs[0]["mask"].copy()
    for catalog in catalogs[1:]:
        if len(catalog["mask"]) != len(mask):
            raise ValueError("Catalogs have different lengths; cannot share a mask")
        mask &= catalog["mask"]
    return mask


# ===========================================================================
# 4. Shape noise
# ===========================================================================

def _shape_signal(catalog: dict, method: str, mask: np.ndarray,
                  detection_weighted: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Return the (signal, response) pair used for the shape-noise statistic."""
    entry = catalog["methods"][method]
    e = entry["e"][mask]
    R = entry["R"]
    if R is None:
        raise ValueError(f"Method {method!r} has no response column; shape noise is undefined")
    R = R[mask]
    if not detection_weighted:
        return e, R
    if catalog["w"] is None:
        raise ValueError("detection_weighted=True needs the fpfs_w column")
    # Matches blending_set_test.ipynb / det_shape_calibration:
    #   signal = w * e,  response = w * R + (dw/dg) * e
    with np.load(catalog["path"]) as saved:
        if "fpfs_dw_dg1" not in saved.files or "fpfs_dw_dg2" not in saved.files:
            raise KeyError(f"{Path(catalog['path']).name}: missing fpfs_dw_dg1/2")
        dw = np.column_stack([np.asarray(saved["fpfs_dw_dg1"], dtype=np.float64),
                              np.asarray(saved["fpfs_dw_dg2"], dtype=np.float64)])[mask]
    w = catalog["w"][mask]
    return w[:, None] * e, w[:, None] * R + dw * e


def accumulate_shape_noise(
    paths: Sequence[Path | str],
    methods: Sequence[str] = DEFAULT_METHODS,
    detection_weighted: bool = False,
    mag_cut: Optional[Mapping] = None,
    progress_every: int = 200,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Per-exposure sufficient statistics for ``std(e) / |mean(R)|``.

    Each exposure contributes ``[n, sum(e), sum(e^2), sum(R)]`` per method under
    a common finite mask across all requested methods.

    Returns ``(statistics, counts, manifest)`` where ``statistics`` has shape
    ``(n_exposure, n_method, 4, 2)`` and ``counts`` is
    ``[n_detections, n_finite_per_method, n_common]``.
    """
    statistics, manifest = [], []
    counts = np.zeros(3, dtype=np.int64)
    for i, path in enumerate(paths):
        catalog = load_exposure_arrays(path, methods=methods, mag_cut=mag_cut)
        mask = catalog["mask"]
        counts += [catalog["n_total"], int(mask.sum()), int(mask.sum())]
        per_method = []
        for method in methods:
            signal, response = _shape_signal(catalog, method, mask, detection_weighted)
            per_method.append(np.stack([
                np.full(2, len(signal), dtype=np.float64),
                signal.sum(axis=0),
                np.square(signal).sum(axis=0),
                response.sum(axis=0),
            ]))
        statistics.append(per_method)
        manifest.append(dict(exposure=int(Path(path).stem.split("_")[-1]), path=str(path),
                             n_det=catalog["n_total"], n_used=int(mask.sum())))
        if verbose and (i + 1) % max(1, progress_every) == 0:
            print(f"[test_func]   shape-noise stats: {i + 1}/{len(paths)} exposures")
    return np.asarray(statistics), counts, pd.DataFrame(manifest)


def shape_noise_from_stats(total: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``std(e) / |mean(R)|`` from accumulated ``[n, sum_e, sum_e2, sum_R]``.

    Same estimator as ``blending_set_test.ipynb::sn_from_stats``; works for any
    leading shape, e.g. ``(n_exposure, n_method, 4, 2)`` or ``(n_method, 4, 2)``.
    """
    total = np.asarray(total, dtype=np.float64)
    n = total[..., 0, :]
    sum_e, sum_e2, sum_r = total[..., 1, :], total[..., 2, :], total[..., 3, :]
    if np.any(n < 2):
        raise ValueError("Too few objects to compute a standard deviation")
    mean_r = sum_r / n
    if np.any(np.abs(mean_r) < 1e-12):
        raise ValueError("Mean response is zero; shape noise is undefined")
    std = np.sqrt(np.maximum((sum_e2 - np.square(sum_e) / n) / (n - 1), 0.0))
    return std, mean_r, std / np.abs(mean_r)


def bootstrap_shape_noise(
    total: np.ndarray,
    n_boot: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """Exposure bootstrap of the shape-noise statistic.

    Exposures are resampled with replacement (paired across methods, since every
    method sees the same mask).  Passing the same ``seed`` for different models
    pairs the bootstrap draws across models as well, which is what
    ``compare_shape_noise`` relies on for ratios.

    Returns ``(n_boot, ...)`` draws with the trailing shape of ``total`` minus
    the last two axes, plus the two shape components, e.g.
    ``(n_boot, n_method, 2)`` for a ``(n_exposure, n_method, 4, 2)`` input.
    """
    total = np.asarray(total, dtype=np.float64)
    if len(total) < 2:
        raise ValueError("At least two exposures are needed to bootstrap")
    rng = np.random.default_rng(seed)
    out_shape = (n_boot,) + tuple(total.shape[1:-2]) + (2,)
    draws = np.empty(out_shape, dtype=np.float64)
    for b in range(n_boot):
        index = rng.integers(0, len(total), size=len(total))
        draws[b] = shape_noise_from_stats(total[index].sum(axis=0))[2]
    return draws


def compare_shape_noise(
    sweep: Mapping[str, Sequence[Path | str]],
    methods: Sequence[str] = DEFAULT_METHODS,
    detection_weighted: bool = False,
    mag_cut: Optional[Mapping] = None,
    n_boot: int = 200,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Shape noise for several models (and the FPFS baseline) side by side.

    ``sweep`` maps a model label to its catalog paths.  Labels are also used to
    order the output, so ``"blended_det_rdsm_sweep_s4"`` style names sort by the
    trailing sigma.

    Returns ``(table, ratios, stats)``:

    * ``table``: one row per (label, method, component) with ``std_e``,
      ``mean_response``, ``shape_noise`` and its exposure-bootstrap error.
    * ``ratios``: one row per (label, component) with ``ml_over_fpfs`` and its
      paired bootstrap error.
    * ``stats``: raw per-model arrays needed for plots or further analysis.
    """
    if len(methods) < 2:
        raise ValueError("compare_shape_noise needs at least two methods for a ratio")
    stats: Dict[str, dict] = {}
    rows, ratio_rows = [], []

    for label, paths in sweep.items():
        total, counts, manifest = accumulate_shape_noise(
            paths, methods=methods, detection_weighted=detection_weighted,
            mag_cut=mag_cut, verbose=verbose,
        )
        std, mean_r, sn = shape_noise_from_stats(total.sum(axis=0))
        draws = bootstrap_shape_noise(total, n_boot=n_boot, seed=seed)
        # Per-component absolute error too (not just the ratio) for completeness.
        sn_draws = draws
        stats[label] = dict(total=total, counts=counts,
                            std=std, mean_response=mean_r, shape_noise=sn,
                            bootstrap=sn_draws)
        for i, method in enumerate(methods):
            for j in range(2):
                rows.append(dict(
                    label=label, method=method, component=f"e{j + 1}",
                    n_common=int(counts[2]),
                    std_e=float(std[i, j]),
                    mean_response=float(mean_r[i, j]),
                    shape_noise=float(sn[i, j]),
                    bootstrap_error=float(sn_draws[:, i, j].std(ddof=1)),
                ))
        if "ml" in methods and "fpfs" in methods:
            i_ml, i_fpfs = methods.index("ml"), methods.index("fpfs")
            ratio = sn[i_ml] / sn[i_fpfs]
            ratio_draws = sn_draws[:, i_ml, :] / sn_draws[:, i_fpfs, :]
            stats[label]["ratio"] = ratio
            stats[label]["ratio_bootstrap"] = ratio_draws
            for j in range(2):
                ratio_rows.append(dict(
                    label=label, component=f"e{j + 1}",
                    ml_over_fpfs=float(ratio[j]),
                    bootstrap_error=float(ratio_draws[:, j].std(ddof=1)),
                ))

    return pd.DataFrame(rows), pd.DataFrame(ratio_rows), stats


# ===========================================================================
# 5. Pair ellipticity correlation
# ===========================================================================

def unique_match_mask(
    match_idx: Optional[np.ndarray],
    match_dist: Optional[np.ndarray],
    mask: np.ndarray,
) -> np.ndarray:
    """Keep a single (nearest) detection per matched truth object.

    Blended exposures often detect the same galaxy twice; those pairs sit at
    almost zero separation and would inflate ``xi(r)``.  Unmatched detections
    (``match_idx < 0``) are kept as-is.
    """
    if match_idx is None:
        return mask
    keep = mask.copy()
    idx = np.asarray(match_idx)
    matched = np.flatnonzero(keep & (idx >= 0))
    if len(matched) == 0:
        return keep
    if match_dist is None:
        order = matched
    else:
        order = matched[np.argsort(np.asarray(match_dist)[matched], kind="stable")]
    _, first = np.unique(idx[order], return_index=True)
    keep[order] = False
    keep[order[first]] = True
    return keep


def signal_magnitude(
    paths: Sequence[Path | str],
    methods: Sequence[str] = DEFAULT_CORRELATION_METHODS,
    weight_key: str = "fpfs_w",
    detection_weighted: bool = False,
    unique_match: bool = True,
    mag_cut: Optional[Mapping] = None,
    progress_every: int = 200,
    verbose: bool = True,
) -> pd.DataFrame:
    """Mean modulus ``<|e_c|>`` of each method/component, exposure by exposure.

    Uses exactly the sample of :func:`pair_ellipticity_correlation` (same
    ``unique_match`` mask), so ``xi(r) / <|e_c|>`` compares the leaked
    correlation with the typical amplitude of the *measured* ellipticity itself
    -- no truth information enters the normalisation.  With
    ``detection_weighted=True`` the reported mean is the weight-averaged
    ``sum(w |e|) / sum(w)``, matching the ``w_i w_j`` pair weights.

    Returns a long table, one row per exposure and method/component:
    ``exposure``, ``path``, ``method``, ``component``, ``n``, ``mean_abs_e``,
    ``wmean_abs_e``.  Average over exposures to get the same exposure weighting
    the correlation uses.
    """
    rows = []
    for i, path in enumerate(paths):
        catalog = load_exposure_arrays(path, methods=methods, weight_key=weight_key,
                                       mag_cut=mag_cut)
        mask = (unique_match_mask(catalog["match_idx"], catalog["match_dist"],
                                 catalog["mask"])
                if unique_match else catalog["mask"])
        n_used = int(mask.sum())
        w = catalog["w"][mask] if catalog["w"] is not None else np.ones(n_used)
        for name in methods:
            magnitude = np.abs(catalog["methods"][name]["e"][mask])
            for c, component in enumerate(("e1", "e2")):
                rows.append(dict(
                    exposure=int(Path(path).stem.split("_")[-1]), path=str(path),
                    method=name, component=component, n=n_used,
                    mean_abs_e=float(magnitude[:, c].mean()) if n_used else np.nan,
                    wmean_abs_e=(float((w * magnitude[:, c]).sum() / w.sum())
                                 if (detection_weighted and n_used and w.sum() > 0)
                                 else np.nan),
                ))
        if verbose and (i + 1) % max(1, progress_every) == 0:
            print(f"[test_func]   |e|: {i + 1}/{len(paths)} exposures")
    return pd.DataFrame(rows)


def component_norm(norm_e1: float, norm_e2: float, component: str) -> float:
    """Scale of a :data:`PAIR_COMPONENTS` entry from the two single-component means.

    ``e1e1`` -> ``<|e1|>``, ``e2e2`` -> ``<|e2|>``, ``e1e2`` -> the geometric
    mean ``sqrt(<|e1|> <|e2|>)``.
    """
    if component == "e1e1":
        return float(norm_e1)
    if component == "e2e2":
        return float(norm_e2)
    if component == "e1e2":
        return float(np.sqrt(norm_e1 * norm_e2))
    raise ValueError(f"Unknown component {component!r}")


def _pair_indices(
    pairs: np.ndarray,
    xy: np.ndarray,
    min_separation_px: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pair indices and separations, dropping pairs closer than the cut."""
    if len(pairs) == 0:
        empty_index = np.empty(0, dtype=np.int64)
        return empty_index, empty_index, np.empty(0, dtype=np.float64)
    left, right = pairs[:, 0], pairs[:, 1]
    separation = np.hypot(xy[left, 0] - xy[right, 0], xy[left, 1] - xy[right, 1])
    keep = separation >= min_separation_px
    return left[keep], right[keep], separation[keep]


def _exposure_accumulation(
    index: int,
    path: Path | str,
    methods: Sequence[str],
    bins: np.ndarray,
    min_separation_px: float,
    unique_match: bool,
    all_pairs_max_px: Optional[float],
    max_all_pairs: int,
    weight_key: str,
    mag_cut: Optional[Mapping] = None,
) -> tuple:
    """Binned + all-pairs accumulators of *one* exposure.

    Module-level so it can be submitted to a process pool (a closure would not
    be picklable).  The returned arrays carry no exposure axis; the caller
    writes them into ``[index]``, which keeps the pooled result identical to
    the serial one.
    """
    from scipy.spatial import cKDTree

    n_bins = bins.size - 1
    n_methods = len(methods)
    n_components = len(PAIR_COMPONENTS)
    binned_shape = (n_methods, n_methods, n_components, n_bins)
    all_shape = (n_methods, n_methods, n_components)
    counts = np.zeros(binned_shape)
    sums = np.zeros(binned_shape)
    wcounts = np.zeros(binned_shape)
    wsums = np.zeros(binned_shape)
    all_counts = np.zeros(all_shape)
    all_sums = np.zeros(all_shape)
    all_wcounts = np.zeros(all_shape)
    all_wsums = np.zeros(all_shape)

    catalog = load_exposure_arrays(path, methods=methods, weight_key=weight_key,
                                   mag_cut=mag_cut)
    mask = (unique_match_mask(catalog["match_idx"], catalog["match_dist"], catalog["mask"])
            if unique_match else catalog["mask"])
    n_used = int(mask.sum())
    manifest = dict(exposure=int(Path(path).stem.split("_")[-1]), path=str(path),
                    n_det=catalog["n_total"], n_used=n_used)
    n_pairs_binned = 0

    if n_used >= 2:
        xy = np.column_stack([catalog["y"][mask], catalog["x"][mask]])
        w = catalog["w"][mask] if catalog["w"] is not None else np.ones(n_used)
        signals = {name: catalog["methods"][name]["e"][mask] for name in methods}
        tree = cKDTree(xy)

        def accumulate(left: np.ndarray, right: np.ndarray, bin_index: np.ndarray,
                       binned: bool) -> None:
            """Fill the binned accumulators, or the single all-pairs slot."""
            pair_weight = w[left] * w[right]
            if binned:
                bin_counts = np.bincount(bin_index, minlength=n_bins).astype(np.float64)
                bin_wcounts = np.bincount(bin_index, weights=pair_weight, minlength=n_bins)
            for a_index, a_name in enumerate(methods):
                ea = signals[a_name][left]
                for b_index, b_name in enumerate(methods):
                    eb = signals[b_name][right]
                    for c, (ca, cb) in enumerate(((0, 0), (1, 1), (0, 1))):
                        product = ea[:, ca] * eb[:, cb]
                        weighted = product * pair_weight
                        if binned:
                            sums[a_index, b_index, c] = np.bincount(
                                bin_index, weights=product, minlength=n_bins)
                            wsums[a_index, b_index, c] = np.bincount(
                                bin_index, weights=weighted, minlength=n_bins)
                            counts[a_index, b_index, c] = bin_counts
                            wcounts[a_index, b_index, c] = bin_wcounts
                        else:
                            all_sums[a_index, b_index, c] = product.sum()
                            all_wsums[a_index, b_index, c] = weighted.sum()
                            all_counts[a_index, b_index, c] = len(left)
                            all_wcounts[a_index, b_index, c] = pair_weight.sum()

        # ---- binned, pairs up to the largest bin edge ----
        pairs = tree.query_pairs(r=float(bins[-1]), output_type="ndarray")
        left, right, separation = _pair_indices(pairs, xy, min_separation_px)
        if len(separation):
            bin_index = np.clip(np.searchsorted(bins, separation, side="right") - 1,
                                0, n_bins - 1)
            accumulate(left, right, bin_index, binned=True)
            n_pairs_binned = len(separation)

        # ---- un-binned, every pair of the exposure ----
        radius = np.inf if all_pairs_max_px is None else max(
            float(all_pairs_max_px), float(bins[-1]))
        if np.isinf(radius) and n_used * (n_used - 1) // 2 > max_all_pairs:
            raise RuntimeError(
                f"{Path(path).name}: {n_used} detections would give "
                f"{n_used * (n_used - 1) // 2} pairs. Set all_pairs_max_px to cap "
                "the separation, or raise max_all_pairs."
            )
        pairs_all = tree.query_pairs(r=radius, output_type="ndarray")
        left_all, right_all, separation_all = _pair_indices(pairs_all, xy, min_separation_px)
        if len(separation_all):
            accumulate(left_all, right_all, np.zeros(0, dtype=np.int64), binned=False)

    return (index, manifest, counts, sums, wcounts, wsums,
            all_counts, all_sums, all_wcounts, all_wsums, n_pairs_binned)


def pair_ellipticity_correlation(
    paths: Sequence[Path | str],
    methods: Sequence[str] = DEFAULT_CORRELATION_METHODS,
    bins_px: Sequence[float] = DEFAULT_BINS_PX,
    detection_weighted: bool = False,
    weight_key: str = "fpfs_w",
    min_separation_px: float = 2.0,
    unique_match: bool = True,
    all_pairs_max_px: Optional[float] = None,
    max_all_pairs: int = 50_000_000,
    progress_every: int = 200,
    verbose: bool = True,
    workers: int = 1,
    mag_cut: Optional[Mapping] = None,
) -> dict:
    """Bin pair products ``e_a(i) e_b(j)`` by separation, exposure by exposure.

    For every pair of shape methods ``(a, b)`` (including ``a == b``) and every
    component in :data:`PAIR_COMPONENTS`, these are accumulated per separation
    bin and per exposure:

    * ``counts``   -- number of pairs
    * ``sums``     -- ``sum e_a(i) e_b(j)``
    * ``wcounts``  -- ``sum w_i w_j``
    * ``wsums``    -- ``sum w_i w_j e_a(i) e_b(j)``

    The same four quantities are accumulated over *all* pairs of the exposure
    (``all_*``), i.e. the un-binned average cross-correlation requested for the
    blended sample.  ``all_pairs_max_px=None`` (default) uses every pair of the
    exposure; a finite value caps the separation instead, which is much cheaper
    for very crowded exposures.  ``max_all_pairs`` refuses to build the
    unbounded pair list when it would exceed that many pairs.

    ``mag_cut`` (see :func:`load_exposure_arrays`) restricts every statistic to
    detections whose matched truth object passes the magnitude cut.

    ``detection_weighted=True`` uses ``w_i w_j`` as the pair weight, so
    ``xi = sum(w_i w_j e_i e_j) / sum(w_i w_j)``; the raw pair counts are still
    reported in ``counts``/``all_counts`` and the weight sums in
    ``wcounts``/``all_wcounts``.

    ``truth`` uses ``truth_e1_weighted``/``truth_e2_weighted``, i.e. the
    intrinsic bulge+disk ellipticity; because distinct galaxies' intrinsic
    shapes are uncorrelated, ``ml`` vs ``truth`` isolates the
    blending-induced *measurement* correlation.

    ``workers > 1`` spreads the exposure loop over that many processes (one
    exposure per task, so ~1.4e6 pairs for the un-binned all-pairs average).
    Each task returns the accumulators of its own exposure, which are written
    back at that exposure's index, so the pooled arrays are bit-identical to
    the serial ones.

    Returns a dict of arrays with shapes ``(n_exposure, n_method, n_method, 3,
    n_bins)`` for the binned quantities and ``(n_exposure, n_method, n_method,
    3)`` for the all-pairs ones, alongside ``bins_px``, ``methods`` and the
    per-exposure manifest.
    """
    bins = np.asarray(sorted(set(float(b) for b in bins_px)), dtype=np.float64)
    if bins.size < 2:
        raise ValueError("Need at least two separation bin edges")
    if min_separation_px < 0:
        raise ValueError("min_separation_px must be nonnegative")
    n_bins = bins.size - 1
    n_methods = len(methods)
    n_components = len(PAIR_COMPONENTS)
    n_exp = len(paths)

    shape_binned = (n_exp, n_methods, n_methods, n_components, n_bins)
    shape_all = (n_exp, n_methods, n_methods, n_components)
    counts = np.zeros(shape_binned)
    sums = np.zeros(shape_binned)
    wcounts = np.zeros(shape_binned)
    wsums = np.zeros(shape_binned)
    all_counts = np.zeros(shape_all)
    all_sums = np.zeros(shape_all)
    all_wcounts = np.zeros(shape_all)
    all_wsums = np.zeros(shape_all)

    def call_args(index: int, path: Path | str) -> tuple:
        """Positional arguments of :func:`_exposure_accumulation`."""
        return (index, str(path), tuple(methods), bins, float(min_separation_px),
                bool(unique_match), all_pairs_max_px, int(max_all_pairs), weight_key,
                mag_cut)

    results = []
    n_workers = min(max(1, int(workers or 1)), max(1, n_exp))
    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_exposure_accumulation, *call_args(i, p))
                       for i, p in enumerate(paths)]
            for done, future in enumerate(as_completed(futures), 1):
                results.append(future.result())
                if verbose and done % max(1, progress_every) == 0:
                    print(f"[test_func]   correlations: {done}/{n_exp} exposures "
                          f"done on {n_workers} workers")
    else:
        for i, path in enumerate(paths):
            result = _exposure_accumulation(*call_args(i, path))
            results.append(result)
            if verbose and (i + 1) % max(1, progress_every) == 0:
                print(f"[test_func]   correlations: {i + 1}/{n_exp} exposures "
                      f"({result[-1]} pairs within {bins[-1]:g} px in "
                      f"{Path(path).name})")

    manifest: List[Optional[dict]] = [None] * n_exp
    for (index, row, c, s, wc, ws, ac, asum, awc, aws, _pairs) in results:
        counts[index], sums[index] = c, s
        wcounts[index], wsums[index] = wc, ws
        all_counts[index], all_sums[index] = ac, asum
        all_wcounts[index], all_wsums[index] = awc, aws
        manifest[index] = row

    return dict(bins_px=bins, methods=list(methods), components=list(PAIR_COMPONENTS),
                min_separation_px=float(min_separation_px),
                all_pairs_max_px=all_pairs_max_px,
                detection_weighted=bool(detection_weighted),
                counts=counts, sums=sums, wcounts=wcounts, wsums=wsums,
                all_counts=all_counts, all_sums=all_sums,
                all_wcounts=all_wcounts, all_wsums=all_wsums,
                manifest=pd.DataFrame(manifest))


def summarize_pair_correlation(
    accumulation: dict,
    method_a: str = "ml",
    method_b: str = "ml",
    component: str = "e1e1",
    weighted: bool = False,
    min_valid_exposures: int = 2,
) -> pd.DataFrame:
    """``xi(r)`` for one method pair with an exposure-scatter error bar.

    Per exposure the bin average is ``sum / count`` (NaN when a bin is empty);
    the reported ``xi`` is the mean over the exposures that have that bin and
    ``xi_err`` is ``std / sqrt(n_exposures)``.  The pooled value
    ``xi_pooled = sum / count`` over all exposures is reported as well.  The
    last row (``is_all_pairs``) is the un-binned average over every pair of the
    exposure; ``n_pairs`` is always the raw pair count and ``pair_weight_sum``
    the summed pair weight (equal to ``n_pairs`` when unweighted).
    """
    if component not in accumulation["components"]:
        raise ValueError(f"Unknown component {component!r}; "
                         f"known: {accumulation['components']}")
    if method_a not in accumulation["methods"] or method_b not in accumulation["methods"]:
        raise ValueError(f"Unknown method pair ({method_a!r}, {method_b!r})")
    c = accumulation["components"].index(component)
    ia = accumulation["methods"].index(method_a)
    ib = accumulation["methods"].index(method_b)
    counts = accumulation["counts"][:, ia, ib, c]
    sums = accumulation["sums"][:, ia, ib, c]
    weight_counts = accumulation["wcounts"][:, ia, ib, c]
    weight_sums = accumulation["wsums"][:, ia, ib, c]
    all_counts = accumulation["all_counts"][:, ia, ib, c]
    all_sums = accumulation["all_sums"][:, ia, ib, c]
    all_weight_counts = accumulation["all_wcounts"][:, ia, ib, c]
    all_weight_sums = accumulation["all_wsums"][:, ia, ib, c]
    bins = accumulation["bins_px"]

    def per_exposure_mean(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        return np.divide(numerator, denominator,
                         out=np.full(numerator.shape, np.nan, dtype=np.float64),
                         where=denominator > 0)

    if weighted:
        exposure_values = per_exposure_mean(weight_sums, weight_counts)
        pooled = per_exposure_mean(weight_sums.sum(axis=0), weight_counts.sum(axis=0))
    else:
        exposure_values = per_exposure_mean(sums, counts)
        pooled = per_exposure_mean(sums.sum(axis=0), counts.sum(axis=0))

    valid = np.isfinite(exposure_values).sum(axis=0)
    with np.errstate(invalid="ignore"):
        xi = np.where(valid >= min_valid_exposures,
                      np.nanmean(exposure_values, axis=0), np.nan)
    # Only bins present in at least two exposures get an exposure-scatter error.
    xi_err = np.full(exposure_values.shape[1], np.nan)
    enough = valid >= max(2, min_valid_exposures)
    if np.any(enough):
        xi_err[enough] = (np.nanstd(exposure_values[:, enough], axis=0, ddof=1)
                          / np.sqrt(valid[enough]))

    rows = []
    for b in range(len(bins) - 1):
        rows.append(dict(
            method_a=method_a, method_b=method_b, component=component,
            weighted=weighted, is_all_pairs=False,
            separation_min_px=float(bins[b]), separation_max_px=float(bins[b + 1]),
            separation_mid_px=float(0.5 * (bins[b] + bins[b + 1])),
            n_pairs=float(counts[:, b].sum()),
            pair_weight_sum=float(weight_counts[:, b].sum()),
            n_exposures=int(valid[b]), xi=float(xi[b]), xi_err=float(xi_err[b]),
            xi_pooled=float(pooled[b]),
        ))

    if weighted:
        exposure_all = per_exposure_mean(all_weight_sums, all_weight_counts)
        pooled_all = per_exposure_mean(all_weight_sums.sum(), all_weight_counts.sum())
    else:
        exposure_all = per_exposure_mean(all_sums, all_counts)
        pooled_all = per_exposure_mean(all_sums.sum(), all_counts.sum())
    n_valid_all = int(np.isfinite(exposure_all).sum())
    rows.append(dict(
        method_a=method_a, method_b=method_b, component=component,
        weighted=weighted, is_all_pairs=True,
        separation_min_px=float(accumulation["min_separation_px"]),
        separation_max_px=float(accumulation["all_pairs_max_px"]) if accumulation[
            "all_pairs_max_px"] is not None else float("inf"),
        separation_mid_px=float("nan"),
        n_pairs=float(all_counts.sum()),
        pair_weight_sum=float(all_weight_counts.sum()),
        n_exposures=n_valid_all,
        xi=float(np.nanmean(exposure_all)) if n_valid_all else float("nan"),
        xi_err=(float(np.nanstd(exposure_all, ddof=1) / np.sqrt(n_valid_all))
                if n_valid_all >= 2 else float("nan")),
        xi_pooled=float(pooled_all),
    ))
    return pd.DataFrame(rows)


def correlate_sweep(
    sweep: Mapping[str, Sequence[Path | str]],
    method_pairs: Sequence[Tuple[str, str]] = (("ml", "ml"), ("fpfs", "fpfs"),
                                              ("truth", "truth"), ("ml", "fpfs"),
                                              ("ml", "truth"), ("fpfs", "truth")),
    methods: Sequence[str] = DEFAULT_CORRELATION_METHODS,
    components: Sequence[str] = ("e1e1", "e2e2", "e1e2"),
    bins_px: Sequence[float] = DEFAULT_BINS_PX,
    min_separation_px: float = 2.0,
    unique_match: bool = True,
    weighted: bool = False,
    all_pairs_max_px: Optional[float] = None,
    workers: int = 1,
    mag_cut: Optional[Mapping] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Correlate every model in ``sweep`` and return one long table.

    Returns ``(table, accumulations)``; ``table`` has a ``label`` column so the
    per-model ``xi(r)`` curves can be compared directly, and ``accumulations``
    keeps the raw per-exposure sums for custom plots.
    """
    accumulations: Dict[str, dict] = {}
    frames = []
    for label, paths in sweep.items():
        accumulation = pair_ellipticity_correlation(
            paths, methods=methods, bins_px=bins_px,
            min_separation_px=min_separation_px, unique_match=unique_match,
            all_pairs_max_px=all_pairs_max_px, workers=workers,
            mag_cut=mag_cut, verbose=verbose,
        )
        accumulations[label] = accumulation
        for method_a, method_b in method_pairs:
            if method_a not in methods or method_b not in methods:
                continue
            for component in components:
                frame = summarize_pair_correlation(
                    accumulation, method_a=method_a, method_b=method_b,
                    component=component, weighted=weighted,
                )
                frame.insert(0, "label", label)
                frames.append(frame)
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return table, accumulations


# ===========================================================================
# 6. Sweep discovery and regression checks
# ===========================================================================

_SIGMA_RE = re.compile(r"_s(\d+)$")


def discover_sweep_jobs(
    sweep_root: Path | str,
    prefix: str = SWEEP_JOB_PREFIX,
    pattern: str = "catalog_*.npz",
) -> pd.DataFrame:
    """Find ``<sweep_root>/<prefix>*_s<sigma>/`` measurement directories.

    Returns a DataFrame with ``label``, ``sigma``, ``job``, ``directory`` and
    ``n_catalogs``, sorted by sigma.
    """
    sweep_root = Path(sweep_root)
    rows = []
    for directory in sorted(sweep_root.glob(f"{prefix}*")):
        if not directory.is_dir():
            continue
        match = _SIGMA_RE.search(directory.name)
        if match is None:
            continue
        files = sorted(directory.glob(pattern))
        if not files:
            continue
        rows.append(dict(label=directory.name, sigma=float(match.group(1)),
                         job=directory.name, directory=str(directory),
                         n_catalogs=len(files)))
    if not rows:
        raise FileNotFoundError(
            f"No measurement directories matching {prefix}*_s<sigma> under {sweep_root}"
        )
    return pd.DataFrame(rows).sort_values("sigma").reset_index(drop=True)


def compare_catalog_dirs(
    reference_dir: Path | str,
    test_dir: Path | str,
    index_range: Optional[Tuple[int, int]] = None,
    ml_keys: Sequence[str] = ("ml_e1", "ml_e2", "ml_R11", "ml_R22"),
    strict_keys: Sequence[str] = ("det_id", "y", "x", "fpfs_e1", "fpfs_e2",
                                  "fpfs_R11", "fpfs_R22", "fpfs_w", "fpfs_m00",
                                  "fpfs_dw_dg1", "fpfs_dw_dg2", "match_idx"),
) -> pd.DataFrame:
    """Column-by-column comparison of two measurement directories.

    Used to prove that a re-run reproduces an existing catalog set: the
    detection/FPFS columns must match exactly (identical code path and seeds)
    while the ML columns are expected to agree only to floating-point tolerance
    because the batch composition differs.

    Returns a DataFrame with ``key``, ``max_abs_diff``, ``n_finite``,
    ``n_compared`` and ``exact``.
    """
    reference_dir, test_dir = Path(reference_dir), Path(test_dir)
    reference_paths = catalog_paths(reference_dir, index_range)
    rows = []
    for key in list(strict_keys) + list(ml_keys):
        max_diff, n_finite, n_compared = 0.0, 0, 0
        for reference_path in reference_paths:
            test_path = test_dir / reference_path.name
            if not test_path.is_file():
                raise FileNotFoundError(f"Missing counterpart: {test_path}")
            with np.load(reference_path) as ref, np.load(test_path) as test:
                if key not in ref.files or key not in test.files:
                    raise KeyError(f"{key!r} missing in {reference_path.name} or {test_path.name}")
                a = np.asarray(ref[key], dtype=np.float64)
                b = np.asarray(test[key], dtype=np.float64)
            if a.shape != b.shape:
                raise ValueError(f"{key}: shape mismatch {a.shape} vs {b.shape}")
            n_compared += a.size
            finite = np.isfinite(a) & np.isfinite(b)
            n_finite += int(finite.sum())
            if finite.any():
                max_diff = max(max_diff, float(np.max(np.abs(a[finite] - b[finite]))))
        rows.append(dict(key=key, max_abs_diff=max_diff, n_finite=n_finite,
                         n_compared=n_compared, exact=bool(max_diff == 0.0)))
    frame = pd.DataFrame(rows)
    frame["group"] = ["ml" if key in ml_keys else "strict" for key in frame["key"]]
    return frame[["key", "group", "max_abs_diff", "n_finite", "n_compared", "exact"]]
