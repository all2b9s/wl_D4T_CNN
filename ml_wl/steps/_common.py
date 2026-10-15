import json
import sys
from pathlib import Path

ML_WL_DIR = Path(__file__).resolve().parents[1]


def sg_job(step, cfg, n=1, extra=""):
    """Shell lines for CNN-side jobs: `python -m ml_wl.steps.<step>` run from the single_galaxies root."""
    cfgp = Path(cfg["round_dir"]) / "config.resolved.json"
    return [f"python -u -m ml_wl.steps.{step} --config {cfgp} --job $JOB_INDEX{extra}"]


def opt_dir(cfg):
    return cfg["paths"]["opt_e"]


def load_job_args(step):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--job", type=int, default=0)
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    return cfg, args.job


def d(cfg, step):
    from ml_wl.config import step_dir
    return step_dir(cfg, step)
