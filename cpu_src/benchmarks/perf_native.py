"""Single-core real-stamp workload for external perf stat / perf record."""
import argparse
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base", type=Path, required=True)
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--model", choices=("s4", "attnw"), default="s4")
parser.add_argument("--count", type=int, default=96)
parser.add_argument("--threads", type=int, default=1)
args = parser.parse_args()
if "SLURM_JOB_ID" not in os.environ:
    raise RuntimeError("Use a compute node")
cores = sorted(os.sched_getaffinity(0))
if args.threads < 1 or args.threads > len(cores) or args.count < 1:
    raise ValueError("Invalid thread or sample count")
os.sched_setaffinity(0, set(cores[:args.threads]))
sys.path.insert(0, str(args.base))
import numpy as np
from cpu_src import Model
q = np.load(args.data / "qstamps.npz", allow_pickle=False)["q"]
q = np.ascontiguousarray(np.tile(q, (int(np.ceil(args.count / len(q))), 1, 1, 1))[:args.count])
model = Model.load(args.data / (args.model + ".npz"))
model.measure_qstamps(q[:2 * args.threads], 2, args.threads)
start = time.perf_counter()
result = model.measure_qstamps(q, 2, args.threads)
print({"model": args.model, "count": len(q), "seconds": time.perf_counter() - start,
       "checksum": float(result.e.sum())}, flush=True)
