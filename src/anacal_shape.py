import argparse

import anacal
import multiprocessing as mp
from src.anacal.fpfs import fpfs_measure


def parse_shear_tasks(task_list):
    """Parse shear tasks like ['0_g1', '1_g2'] -> [(0, 'g1'), (1, 'g2')]"""
    parsed = []
    for t in task_list:
        try:
            idx, shear = t.split('_')
            parsed.append((int(idx), shear))
        except ValueError:
            raise ValueError(f"Invalid shear task format '{t}', expected like '0_g1' or '1_g2'")
    return parsed


def parse_noise_level(value):
    if value == 'adaptive':
        return value
    return float(value)


def main():
    parser = argparse.ArgumentParser(description="Measure FPFS magnitude for a set of shear tasks.")
    parser.add_argument("--dir", type=str, required=True,
                         help="Directory holding the simulated cutouts.")
    parser.add_argument("--save-dir", type=str, required=True,
                         help="Output directory for the measured magnitude (m00) files.")
    parser.add_argument("--noise-level", type=parse_noise_level, required=True,
                         help="Gaussian noise sigma (ADU), or 'adaptive'.")
    parser.add_argument("--shear-tasks", type=str, nargs='+', required=True,
                         help="Shear tasks like '0_g1' '1_g2'.")
    parser.add_argument("--fname", type=str, default="fpfs_mag",
                         help="Output filename prefix.")
    parser.add_argument("--index-range", type=int, nargs=2, default=[0, 4000],
                         metavar=("START", "STOP"),
                         help="[start, stop) batch index range to process.")
    args = parser.parse_args()

    shear_tasks = parse_shear_tasks(args.shear_tasks)

    for shear_task in shear_tasks:
        print(f"Measuring FPFS magnitude for shear task: {shear_task}")
        fpfs_measure(
            dir=args.dir,
            img_nB_range=args.index_range,
            imgBs=100,
            noise_level=args.noise_level,
            n_workers=128,
            shear_task=shear_task,
            fname=args.fname,
            save_dir=args.save_dir,
        )


if __name__ == "__main__":
    main()
