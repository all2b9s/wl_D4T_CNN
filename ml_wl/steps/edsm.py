"""Step 3: eDSM Optuna search workers (opt_e/optuna_eDSM.py creates the study itself)."""
from ml_wl.config import step_dir
from ml_wl.steps.catalog import training_artifact as catalog_npz, evaluation_args


def db_path(cfg):
    return step_dir(cfg, "edsm") / f"{cfg['edsm']['study_name']}.db"


def jobs(cfg):
    e = cfg["edsm"]
    smoke = " --bias-threshold 1e9" if cfg["test"] and cfg.get("blended") is not None else ""
    stopping = ""
    if e.get("early_stop_patience", 0):
        stopping = (f" --early-stop-patience {e['early_stop_patience']} --min-epochs {e.get('min_epochs', 6)}"
                    f" --trial-timeout {e.get('trial_timeout', 300)} --study-timeout {e.get('study_timeout', 4800)}")
    out = []
    for w in range(1, e["workers"] + 1):
        out.append(dict(cwd="opt", lines=[
            "sleep $((RANDOM % 6))   # avoid SQLite access spikes at startup",
            f"python -u optuna_eDSM.py --data {catalog_npz(cfg)}{evaluation_args(cfg)} --study-name {e['study_name']} "
            f"--storage sqlite:///{db_path(cfg).resolve()} --n-trials {e['trials_per_worker']} "
            f"--DSM_sigma {e['dsm_sigma']} --device cuda --tpe-startup-trials {e['tpe_startup_trials']} "
            f"--seed {w + 888}{smoke}{stopping}"]))
    return out
