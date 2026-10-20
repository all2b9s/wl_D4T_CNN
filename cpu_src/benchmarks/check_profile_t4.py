"""Check an isolated optimized build against saved real-stamp GPU T=4 outputs."""
import argparse
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
for name in ("base", "data", "output"):
    parser.add_argument("--" + name, type=Path, required=True)
args = parser.parse_args()
if "SLURM_JOB_ID" not in os.environ:
    raise RuntimeError("Use a compute node")
sys.path.insert(0, str(args.base))
import numpy as np
from cpu_src import Model


def compare(a, b):
    delta = a.astype(np.float64) - b.astype(np.float64)
    bias = delta.mean(axis=0)
    return {"mean_bias": bias.tolist(), "max_abs_mean_bias": float(np.abs(bias).max()),
            "max_abs": float(np.abs(delta).max()),
            "mean_bias_passed": bool(np.isfinite(delta).all() and (np.abs(bias) < 1e-4).all())}


q = np.load(args.data / "qstamps.npz", allow_pickle=False)["q"]
report = {}
for name in ("s4", "attnw"):
    reference = np.load(args.data / f"{name}_paired.npz", allow_pickle=False)
    count = len(reference["gpu_T4_dE_dx"])
    model = Model.load(args.data / f"{name}.npz")
    result = model.measure_qstamps(q[:count], 4, min(72, len(os.sched_getaffinity(0))))
    two = model.measure_qstamps(q[:count], 2, min(72, len(os.sched_getaffinity(0))))
    report[name] = {"count": count, "e": compare(result.e, reference["strict_fp32_e"][:count]),
                    "R": compare(result.R, reference["strict_fp32_R"][:count]),
                    "dE_dx_diagnostic": compare(result.dE_dx, reference["gpu_T4_dE_dx"]),
                    "T2_vs_T4_e": compare(two.e, result.e), "T2_vs_T4_R": compare(two.R, result.R)}
    np.savez(args.output / f"combined_T4_{name}.npz", e=result.e, R=result.R, dE_dx=result.dE_dx)
(args.output / "combined_T4_checks.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
if not all(row[key]["mean_bias_passed"] for row in report.values() for key in ("e", "R")):
    raise SystemExit(1)
