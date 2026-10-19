"""Steps 2 / s_catalog: merge teacher shapes into the eDSM input npz (opt_e/build_cnn_gauss_train_catalog.py).

fpfs_dir (optional) only supplies the flux/magnitude column; without it the catalog has no magnitude.
"""
from ml_wl.config import step_dir


def artifact(cfg, step="catalog"):
    return step_dir(cfg, step) / "catalog.npz"


def training_artifact(cfg):
    return artifact(cfg)


def evaluation_args(cfg):
    return f" --eval-data {artifact(cfg)} --data-split train --eval-split val" if cfg.get("blended") is not None else ""


def make_job(cfg, meas_step, cat_step, sim, fpfs_dir, fpfs_prefix):
    a, b = sim["batch_start"], sim["batch_start"] + sim["n_batches"]
    fpfs = f"--fpfs-dir {fpfs_dir} --fpfs-prefix {fpfs_prefix}" if fpfs_dir else "--no-fpfs"
    cmd = (f"python -u build_cnn_gauss_train_catalog.py --meas-dir {step_dir(cfg, meas_step)} "
           f"--prefix {sim['fname']} {fpfs} --truth-dir {sim['split_dir']} "
           f"--out {artifact(cfg, cat_step)} --batches {a} {b}")
    return [dict(lines=[cmd], cwd="opt")]


def jobs(cfg):
    if cfg.get("blended") is not None:
        if cfg["blended"].get("prepare_catalog_in_measure", False):
            return []
        from ml_wl.steps.blended import job
        return job(cfg, "catalog")
    c = cfg["catalog"]
    return make_job(cfg, "measure", "catalog", cfg["sim"], c["fpfs_dir"], c["fpfs_prefix"])
