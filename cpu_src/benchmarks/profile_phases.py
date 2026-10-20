"""Per-phase time of a -DD4CNN_PROFILE build at 1 core and all cores (inside an allocation).

Phase sums are per worker thread; dividing by sources gives core-ms per source,
so a phase that grows from 1 to N cores indicates shared-resource contention.
Nested phases are not additive (total contains the others).
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="directory containing a profiled cpu_src build")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Use a compute node")
    sys.path.insert(0, str(args.base))
    import numpy as np
    from cpu_src import Model, _native
    if not hasattr(_native, "profile_stats"):
        raise RuntimeError("Extension was not built with -DD4CNN_PROFILE")
    cores = sorted(os.sched_getaffinity(0))
    q = np.load(args.data / "qstamps.npz", allow_pickle=False)["q"]
    report = {"cores": len(cores), "models": {}}
    for name in ("s4", "attnw"):
        model = Model.load(args.data / f"{name}.npz")
        rows = []
        for threads, count in ((1, 32), (len(cores), 4 * len(q))):
            inputs = np.ascontiguousarray(q[np.arange(count) % len(q)])
            os.sched_setaffinity(0, set(cores[:threads]))
            model.measure_qstamps(inputs[:2 * threads], 2, threads)
            _native.profile_reset()
            start = time.perf_counter()
            model.measure_qstamps(inputs, 2, threads)
            wall = time.perf_counter() - start
            stats = _native.profile_stats()
            row = {"threads": threads, "count": count, "gal_per_sec": count / wall,
                   "core_ms_per_source": {k: v / 1e6 / count for k, v in stats.items()}}
            rows.append(row)
            print(json.dumps({"model": name, **row}), flush=True)
        os.sched_setaffinity(0, set(cores))
        report["models"][name] = rows
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "phases.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
