"""m/c calibration for cpu_src.isolated_measure outputs (shear amplitude configurable).

CPU counterpart of ``src/cal_biases.py`` (no mag cut) for the xlens_shear005 sims, where
|g| = 0.05 instead of the 0.02 hard-coded in get_biases' default.
"""
import argparse
import os
import json
import time
from pathlib import Path
import numpy as np
from src.anacal.batch_calibration import get_biases
from src.anacal.cal_toolkit import format_number

def load_with_retry(path, attempts=5):
    for attempt in range(1, attempts + 1):
        try:
            return np.load(path, allow_pickle=False)
        except OSError as exc:
            print(f"Read failed ({attempt}/{attempts}): {path}: {exc}", flush=True)
            if attempt == attempts:
                raise
            time.sleep(min(5 * attempt, 30))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--noise", type=float, required=True)
    p.add_argument("--fname", required=True)
    p.add_argument("--range", nargs=2, type=int, default=[0, 200])
    p.add_argument("--img_bs", type=int, default=100)
    p.add_argument("--shear_value", type=float, default=0.05)
    p.add_argument("--data_dir", default="/work/hdd/bdsp/wenyinli/measurement_cpu/")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--bs_times", type=int, default=100)
    p.add_argument("--out_csv", default="./logs/calibration_result.csv")
    p.add_argument("--cache_dir", help="Resumable float64 input cache on a reliable filesystem")
    a = p.parse_args()
    folder = format_number(a.noise)
    tasks = [(1, "g1"), (0, "g1"), (1, "g2"), (0, "g2")]  # order: 1p, 1n, 2p, 2n (as cal_biases)
    nb = a.range[1] - a.range[0]
    n = nb * a.img_bs * 100
    cache = Path(a.cache_dir) if a.cache_dir else None
    completed = 0
    metadata = dict(data_dir=os.path.realpath(a.data_dir), noise=a.noise,
                    fname=a.fname, range=a.range, img_bs=a.img_bs)
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        state_path = cache / "progress.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state["metadata"] != metadata:
                raise ValueError("Cache belongs to different inputs; use a new cache_dir")
            completed = state["completed"]
            g = np.load(cache / "shapes.npy", mmap_mode="r+")
            R = np.load(cache / "responses.npy", mmap_mode="r+")
        else:
            g = np.lib.format.open_memmap(cache / "shapes.npy", mode="w+", dtype="float64", shape=(n, 4, 2))
            R = np.lib.format.open_memmap(cache / "responses.npy", mode="w+", dtype="float64", shape=(n, 4, 2, 2))
    else:
        g = np.zeros([n, 4, 2]); R = np.zeros([n, 4, 2, 2])

    def checkpoint(count):
        if cache:
            g.flush(); R.flush()
            tmp = cache / "progress.json.tmp"
            tmp.write_text(json.dumps(dict(metadata=metadata, completed=count)))
            tmp.replace(state_path)

    print(f"Loading {4 * nb} task/batches; resuming at {completed}", flush=True)
    checkpoint(completed)
    try:
        for i, (mode, comp) in enumerate(tasks):
            for b in range(a.range[0], a.range[1]):
                index = i * nb + b - a.range[0]
                if index < completed:
                    continue
                d = os.path.join(a.data_dir, folder, f"{comp}_{mode}")
                s = (b - a.range[0]) * a.img_bs * 100
                e = s + a.img_bs * 100
                g[s:e, i] = load_with_retry(f"{d}/{a.fname}_shapes_{b}.npy")
                R[s:e, i] = load_with_retry(f"{d}/{a.fname}_Rana_{b}.npy")[:, 0]
                completed = index + 1
                if completed % 100 == 0:
                    checkpoint(completed)
                    print(f"Loaded {completed}/{4 * nb} task/batches", flush=True)
    except Exception:
        checkpoint(completed)
        raise
    checkpoint(completed)
    print(f"Loaded N per shear = {n}; bootstrap workers = {a.workers}", flush=True)
    m, c, m_err, c_err = get_biases(g, R, shear_value=a.shear_value, bs_size=n, bs_times=a.bs_times,
                                    n_factor=1, is_twin=True, n_jobs=a.workers, chunk_bs=a.workers)
    line = f"{a.noise},0.85,{a.fname},{m.tolist()},{m_err.tolist()},{c.tolist()},{c_err.tolist()}"
    print(f"N per shear = {n}\nm = {m} +- {m_err}\nc = {c} +- {c_err}")
    print(line)
    with open(a.out_csv, "a") as f:
        f.write(line + "\n")

if __name__ == "__main__":
    main()
