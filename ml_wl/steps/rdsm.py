"""Step 5: choose a retrained checkpoint and write the RDSM label catalog.

Selection: among retrained checkpoints with finite shape noise and max|m| < rdsm.max_abs_m take the lowest
shape noise. If none qualify: fallback=fail aborts, fallback=min_m takes the smallest max|m|.
Standalone (stdlib only): `python rdsm.py select --study-dir D --max-abs-m X --fallback F --out selection.json`
prints the chosen checkpoint path.
"""
import argparse
import csv
import json
import math
from pathlib import Path


def select(study_dir, max_abs_m, fallback, out):
    study_dir = Path(study_dir)
    rows = list(csv.DictReader((study_dir / "retrain_summary.csv").open()))
    ok_rows = [r for r in rows if math.isfinite(float(r["remeasured_sn_mean"]))
               and math.isfinite(float(r["remeasured_bias_max_abs_m"]))
               and float(r["remeasured_bias_max_abs_m"]) >= 0]
    ok = [r for r in ok_rows if 0 <= float(r["remeasured_bias_max_abs_m"]) < max_abs_m]
    rule = f"max|m| < {max_abs_m}; minimize remeasured_sn_mean"
    if ok:
        best = min(ok, key=lambda r: (float(r["remeasured_sn_mean"]), float(r["remeasured_bias_max_abs_m"])))
    elif fallback == "min_m" and ok_rows:
        best = min(ok_rows, key=lambda r: float(r["remeasured_bias_max_abs_m"]))
        rule = f"FALLBACK (nothing had max|m| < {max_abs_m}): smallest max|m|"
    else:
        raise SystemExit(f"no retrained checkpoint with max|m| < {max_abs_m} (fallback={fallback}); "
                         f"best max|m| = {min((float(r['remeasured_bias_max_abs_m']) for r in ok_rows), default=float('nan')):.5f}")
    ckpt = (study_dir / Path(best["checkpoint"]).name).resolve()
    if not ckpt.is_file():
        raise SystemExit(f"missing checkpoint: {ckpt}")
    Path(out).write_text(json.dumps({"rule": rule, "eligible": len(ok), "n_retrained": len(rows),
                                     "selected": best, "checkpoint": str(ckpt)}, indent=2) + "\n")
    print(ckpt)


def jobs(cfg):
    from ml_wl.config import step_dir
    from ml_wl.steps.catalog import artifact as catalog_npz
    from ml_wl.steps.retrain import study_dir
    out = step_dir(cfg, "rdsm")
    r = cfg["rdsm"]
    here = Path(__file__).resolve()
    lines = [
        f"CKPT=$(python {here} select --study-dir {study_dir(cfg)} --max-abs-m {r['max_abs_m']} "
        f"--fallback {r['fallback']} --out {out}/selection.json)",
        'echo "selected checkpoint: $CKPT"',
    ]
    norm_arg = ""
    if cfg.get("blended") is not None:
        # Use the SG module by absolute package path in a separate process:
        # the two repositories both own a top-level `src` package.
        from ml_wl.config import SG_DIR, frozen_config_path
        lines.append(f"NORM=$(cd {SG_DIR} && python -m ml_wl.steps.blended checkpoint-norm "
                     f"--config {frozen_config_path(cfg)})")
        norm_arg = " --norm $NORM --compact"
    lines.append(
        f"python -u build_gauss_train_rdsm_catalog.py --checkpoint \"$CKPT\" --data {catalog_npz(cfg)} "
        f"--out {out}/rdsm_catalog.npz --device cuda{norm_arg}")
    return [dict(lines=lines, cwd="opt")]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["select", "ckpt", "norm"])
    p.add_argument("--study-dir")
    p.add_argument("--max-abs-m", type=float)
    p.add_argument("--fallback", default="fail", choices=["fail", "min_m"])
    p.add_argument("--out")
    p.add_argument("--selection", help="ckpt: selection.json written by `select`")
    p.add_argument("--catalog", help="norm: rdsm_catalog.npz holding rdsm_norm")
    a = p.parse_args()
    if a.cmd == "select":
        select(a.study_dir, a.max_abs_m, a.fallback, a.out)
    elif a.cmd == "ckpt":
        print(json.loads(Path(a.selection).read_text())["checkpoint"])
    else:
        import numpy as np
        with np.load(a.catalog) as z:
            print(float(z["rdsm_norm"]))
