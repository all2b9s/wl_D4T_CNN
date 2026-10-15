"""Step 4: retrain Pareto-front trials (opt_e/optuna_retrain.py)."""
from ml_wl.config import step_dir
from ml_wl.steps.catalog import artifact as catalog_npz
from ml_wl.steps.edsm import db_path


def study_dir(cfg):
    return step_dir(cfg, "retrain") / cfg["edsm"]["study_name"]


def jobs(cfg):
    r = cfg["retrain"]
    cmd = (f"python -u optuna_retrain.py --log-file {db_path(cfg)} --data {catalog_npz(cfg)} "
           f"--out-dir {step_dir(cfg, 'retrain')} --n-select {r['n_select']} --n-repeats {r['n_repeats']} "
           f"--epochs {r['epochs']} --bias-min {r['bias_min']} --bias-max {r['bias_max']} --device cuda")
    return [dict(lines=[cmd], cwd="opt")]
