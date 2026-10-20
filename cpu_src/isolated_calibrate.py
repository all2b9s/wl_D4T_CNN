"""m/c calibration for cpu_src.isolated_measure outputs (shear amplitude configurable).

CPU counterpart of ``src/cal_biases.py`` (no mag cut) for the xlens_shear005 sims, where
|g| = 0.05 instead of the 0.02 hard-coded in get_biases' default.
"""
import argparse
import os
import numpy as np
from src.anacal.batch_calibration import get_biases
from src.anacal.cal_toolkit import format_number

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--noise", type=float, required=True)
    p.add_argument("--fname", required=True)
    p.add_argument("--range", nargs=2, type=int, default=[0, 200])
    p.add_argument("--img_bs", type=int, default=100)
    p.add_argument("--shear_value", type=float, default=0.05)
    p.add_argument("--data_dir", default="/work/hdd/bdsp/wenyinli/measurement_cpu/")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--bs_times", type=int, default=100)
    p.add_argument("--out_csv", default="./logs/calibration_result.csv")
    a = p.parse_args()
    folder = format_number(a.noise)
    tasks = [(1, "g1"), (0, "g1"), (1, "g2"), (0, "g2")]  # order: 1p, 1n, 2p, 2n (as cal_biases)
    nb = a.range[1] - a.range[0]
    n = nb * a.img_bs * 100
    g = np.zeros([n, 4, 2]); R = np.zeros([n, 4, 2, 2])
    for i, (mode, comp) in enumerate(tasks):
        for b in range(a.range[0], a.range[1]):
            d = os.path.join(a.data_dir, folder, f"{comp}_{mode}")
            s = (b - a.range[0]) * a.img_bs * 100
            e = s + a.img_bs * 100
            g[s:e, i] = np.load(f"{d}/{a.fname}_shapes_{b}.npy")
            R[s:e, i] = np.load(f"{d}/{a.fname}_Rana_{b}.npy")[:, 0]
    m, c, m_err, c_err = get_biases(g, R, shear_value=a.shear_value, bs_size=n, bs_times=a.bs_times,
                                    n_factor=1, is_twin=True, n_jobs=a.workers, chunk_bs=20)
    line = f"{a.noise},0.85,{a.fname},{m.tolist()},{m_err.tolist()},{c.tolist()},{c_err.tolist()}"
    print(f"N per shear = {n}\nm = {m} +- {m_err}\nc = {c} +- {c_err}")
    print(line)
    with open(a.out_csv, "a") as f:
        f.write(line + "\n")

if __name__ == "__main__":
    main()
