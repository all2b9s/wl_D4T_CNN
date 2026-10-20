"""Load a round yaml, fill defaults, resolve ${var} references and paths."""
import copy
import re
from pathlib import Path

import yaml

ML_WL_DIR = Path(__file__).resolve().parent
SG_DIR = ML_WL_DIR.parent                       # single_galaxies repo root (cwd of CNN-side steps)

STEPS = ["measure", "catalog", "edsm", "retrain", "rdsm", "s_measure", "s_catalog", "s_label", "dataset", "train", "evaluate"]
# step -> (directory name, lives under data_root instead of round_dir)
STEP_DIRS = {
    "measure": ("01_measure", True),
    "catalog": ("02_catalog", True),
    "edsm": ("03_edsm", False),
    "retrain": ("04_retrain", False),
    "rdsm": ("05_rdsm", False),
    "s_measure": ("06_s_measure", True),     # student sim only (student_sim set): teacher CNN on the other galaxies
    "s_catalog": ("07_s_catalog", True),
    "s_label": ("08_s_label", False),        # eDSM trained on `sim` applied to the student galaxies
    "dataset": ("09_dataset", True),
    "train": ("10_cnn", False),
    "evaluate": ("11_evaluate", False),
}

DEFAULTS = {
    "rounds": "/work/hdd/bfmo/wenyinli/measurement/ml_wl",
    "round_dir": "${rounds}/${name}",
    "data_root": "${round_dir}",
    "log_dir": str(SG_DIR / "output" / "ml_wl" / "${name}"),
    "paths": {"opt_e": "/projects/bdsp/wenyinli/codes/opt_e"},
    "sim": {"shear_task": "2_g1", "n_batches": 100, "batch_start": 0, "abs_shear": 0.02, "psf_fwhm": 0.85},
    "teacher_cnn": {"arch": "Forward8_fixW_CNN",
                    "kwargs": {"num_layers": 5, "base_channels": 32, "res_factor": 0.1}},
    "measure": {"noise": 0, "n_jobs": 1},
    "catalog": {"fpfs_dir": None, "fpfs_prefix": "fpfs_gauss_train"},   # fpfs m00 only gives the magnitude column; optional
    "edsm": {"dsm_sigma": 0.5, "workers": 5, "trials_per_worker": 25, "tpe_startup_trials": 10,
             "study_name": None},
    # NOTE: optuna_eDSM.py hard-codes 20 training epochs per trial; keep retrain.epochs equal to it,
    # otherwise the retrained models drift away from the numbers optuna reported (50 inflated |m|).
    "retrain": {"n_select": 10, "n_repeats": 3, "epochs": 20, "bias_min": 1e-4, "bias_max": 3e-3},
    "rdsm": {"max_abs_m": 0.002, "fallback": "fail"},          # fallback: fail | min_m
    "dataset": {"n_train": 10000, "n_val": 1000, "n_test": 1000, "label": "rdsm",
                "noise": {"mode": "adaptive", "level": 0.594}, "seed": 0, "workers": 16,
                "image_sim": None},          # override of `sim` (dir/shear_task/abs_shear/psf_fwhm) for the student images
    # student galaxies different from the teacher's `sim`: keys override `sim` (dir/shear_task/n_batches/...); optional
    # fpfs_dir/fpfs_prefix give the magnitude column. None = student images come from the same galaxies as `sim`.
    "student_sim": None,
    "blended": None,  # optional adapter for detection catalogs; see r1_AttnW_last_detw.yaml
    "student_cnn": {"arch": "Forward8_fixW_CNN",
                    "kwargs": {"num_layers": 5, "base_channels": 32, "res_factor": 0.1},
                    "epochs": 50, "batch_size": 500, "lr": 1e-3, "wd": 1e-4, "num_workers": 4,
                    "export_as": None},
    "slurm": {"partition": "ghx4", "account": "bfmo-dtai-gh", "cpus": 64, "mem": "120G",
              "env": "/u/wenyinli/.conda/envs/anacal_clone",
              "conda_sh": "/projects/bfcn/wenyinli/miniconda3/etc/profile.d/conda.sh",
              "time": {"measure": "2:00:00", "catalog": "1:00:00", "edsm": "6:00:00",
                       "retrain": "4:00:00", "rdsm": "1:00:00", "s_measure": "2:00:00",
                       "s_catalog": "1:00:00", "s_label": "1:00:00", "dataset": "2:00:00",
                       "train": "4:00:00", "evaluate": "1:00:00"}},
    "steps": None,                                              # None = all steps
}
REQUIRED = [("name",), ("sim", "dir"), ("teacher_cnn", "checkpoint")]

TEST_OVERRIDES = {
    "sim": {"n_batches": 1},
    "measure": {"n_jobs": 1},
    "edsm": {"workers": 1, "trials_per_worker": 2, "tpe_startup_trials": 2},
    "retrain": {"n_select": 1, "n_repeats": 1, "epochs": 2, "bias_min": 0.0, "bias_max": 1e9},
    "rdsm": {"max_abs_m": 1e9},
    "dataset": {"n_train": 800, "n_val": 100, "n_test": 100},
    "student_cnn": {"epochs": 2, "export_as": None},
    "slurm": {"time": {"measure": "0:40:00", "s_measure": "0:40:00", "edsm": "1:00:00", "retrain": "1:00:00",
                       "train": "1:00:00", "dataset": "1:00:00"}},
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else copy.deepcopy(v)
    return out


def _subst(obj, variables):
    if isinstance(obj, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: str(variables[m.group(1)]), obj)
    if isinstance(obj, dict):
        return {k: _subst(v, variables) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_subst(v, variables) for v in obj]
    return obj


NOISE_MODES = ("adaptive", "q_clean", "none")


def _derive_sim(sim):
    sim = dict(sim)
    mode, comp = sim["shear_task"].split("_")[0], sim["shear_task"].split("_")[1]
    sim["comp"], sim["mode"] = comp, int(mode)
    sim["split_dir"] = f"{sim['dir']}/{comp}_{mode}_val_{sim['abs_shear']:.3f}"
    return sim


def load(path, test=False):
    user = yaml.safe_load(Path(path).read_text())
    unknown = set(user) - set(DEFAULTS) - {"name"}
    if unknown:
        raise KeyError(f"unknown top-level config keys: {sorted(unknown)}")
    cfg = _merge(DEFAULTS, user)
    if test:
        cfg = _merge(cfg, TEST_OVERRIDES)
        if cfg["student_sim"] is not None:
            cfg["student_sim"]["n_batches"] = 1
        cfg["name"] = f"{cfg['name']}_test"
    required = [k for k in REQUIRED if not (cfg["blended"] is not None and k == ("sim", "dir"))]
    for key in required:
        node = cfg
        for part in key:
            if not isinstance(node, dict) or node.get(part) in (None, ""):
                raise KeyError(f"missing required config key: {'.'.join(key)}")
            node = node[part]
    # resolve ${rounds} ${name} ${round_dir} ${data_root} (chained, a few passes)
    for _ in range(5):
        variables = {k: cfg[k] for k in ("rounds", "name", "round_dir", "data_root")}
        new = _subst(cfg, variables)
        if new == cfg:
            break
        cfg = new
    cfg["test"] = bool(test)
    cfg["source_config"] = str(Path(path).resolve())
    if cfg["edsm"]["study_name"] is None:
        cfg["edsm"]["study_name"] = f"DSMsigma{cfg['edsm']['dsm_sigma']}_{cfg['name']}"
    if f"DSMsigma{cfg['edsm']['dsm_sigma']}" not in cfg["edsm"]["study_name"]:
        raise ValueError("edsm.study_name must contain 'DSMsigma<sigma>' (optuna_retrain parses sigma from it)")
    if cfg["blended"] is not None:
        b = cfg["blended"]
        if cfg["student_sim"] is not None:
            raise ValueError("blended rounds do not support student_sim")
        for key in ("old_catalog", "measurement", "index_range"):
            if key not in b:
                raise KeyError(f"missing blended.{key}")
        start, stop = b["index_range"]
        if test:
            stop = min(stop, start + 10)
            b["index_range"] = [start, stop]
        if stop - start < 5:
            raise ValueError("blended.index_range needs at least five exposures for 40/20/40")
        b.setdefault("seed", 42)
        if test:
            b.pop("sample_targets", None)
        allowed = ["measure", "catalog", "edsm", "retrain", "rdsm", "evaluate"]
        if cfg["steps"] is None:
            cfg["steps"] = allowed
        if set(cfg["steps"]) - set(allowed):
            raise ValueError("blended rounds support only " + ", ".join(allowed))
        return cfg
    cfg["sim"] = _derive_sim(cfg["sim"])
    cfg["sim"]["fname"] = "teacher_shape"
    if cfg["student_sim"] is not None:
        cfg["student_sim"] = _derive_sim(_merge(cfg["sim"], cfg["student_sim"]))
        if cfg["student_sim"]["split_dir"] == cfg["sim"]["split_dir"]:
            raise ValueError("student_sim points at the same directory as sim; drop student_sim "
                             "(same galaxies) or use dataset.image_sim for another rendering of them")
        cfg["student_sim"].setdefault("fpfs_dir", None)
        cfg["student_sim"].setdefault("fpfs_prefix", cfg["catalog"]["fpfs_prefix"])
    # student images: from student_sim if given, else from sim; image_sim then overrides (same galaxies only)
    base = cfg["student_sim"] if cfg["student_sim"] is not None else cfg["sim"]
    cfg["dataset"]["sim"] = _derive_sim(_merge(base, cfg["dataset"]["image_sim"] or {}))
    if cfg["dataset"]["noise"]["mode"] not in NOISE_MODES:
        raise ValueError(f"dataset.noise.mode must be one of {NOISE_MODES}")
    return cfg


def round_dir(cfg):
    return Path(cfg["round_dir"])


def step_dir(cfg, step):
    """Round-relative step directory (a symlink into data_root for the big-data steps)."""
    return round_dir(cfg) / STEP_DIRS[step][0]


def big_dir(cfg, step):
    return Path(cfg["data_root"]) / STEP_DIRS[step][0]


def frozen_config_path(cfg):
    return round_dir(cfg) / "config.resolved.json"
