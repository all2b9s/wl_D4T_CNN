"""Step 3: eDSM Optuna search workers (opt_e/optuna_eDSM.py creates the study itself)."""
from ml_wl.config import step_dir
from ml_wl.steps.catalog import artifact as catalog_npz


def db_path(cfg):
    return step_dir(cfg, "edsm") / f"{cfg['edsm']['study_name']}.db"


def jobs(cfg):
    e = cfg["edsm"]
    out = []
    for w in range(1, e["workers"] + 1):
        out.append(dict(cwd="opt", lines=[
            "sleep $((RANDOM % 6))   # avoid SQLite access spikes at startup",
            f"python -u optuna_eDSM.py --data {catalog_npz(cfg)} --study-name {e['study_name']} "
            f"--storage sqlite:///{db_path(cfg).resolve()} --n-trials {e['trials_per_worker']} "
            f"--DSM_sigma {e['dsm_sigma']} --device cuda --tpe-startup-trials {e['tpe_startup_trials']} "
            f"--seed {w + 888}"]))
    return out
