"""Step 2: merge the teacher shapes into the eDSM input npz (opt_e/build_cnn_gauss_train_catalog.py)."""
from pathlib import Path

from ml_wl.config import step_dir
from ml_wl.steps._common import opt_dir


def artifact(cfg):
    return step_dir(cfg, "catalog") / "catalog.npz"


def jobs(cfg):
    sim, c = cfg["sim"], cfg["catalog"]
    a, b = sim["batch_start"], sim["batch_start"] + sim["n_batches"]
    cmd = (f"python -u build_cnn_gauss_train_catalog.py --meas-dir {step_dir(cfg, 'measure')} "
           f"--prefix {sim['fname']} --fpfs-dir {c['fpfs_dir']} --fpfs-prefix {c['fpfs_prefix']} "
           f"--truth-dir {sim['split_dir']} --out {artifact(cfg)} --batches {a} {b}")
    return [dict(lines=[cmd], cwd="opt")]
