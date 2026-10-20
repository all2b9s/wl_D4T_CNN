"""CPU (C++) shape measurement of isolated-galaxy xlens cutouts.

CPU counterpart of ``src/shape_measurement.py``: same data loading, noise seeds
and q-image preparation, but e and R are computed by the exported ``cpu_src``
model instead of the PyTorch/GPU ``Calibrator``. Output files are compatible
with ``src/cal_biases.py`` (``<fname>_shapes_<b>.npy`` [N,2] and
``<fname>_Rana_<b>.npy`` [N,1,2,2], R[i,j] = de_i/dg_j).

Run from ``single_galaxies/``:
    python -u -m cpu_src.isolated_measure --model cpu_src/runs/X.npz --noise 0.594 \
        --fname NAME --shear_tasks 0_g1 --range 0 100
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from cpu_src import Model
from src.anacal.batch_calibration import prepare_q_images
from src.anacal.cal_toolkit import build_dataset, format_number

def parse_shear_tasks(task_list):
    parsed = []
    for t in task_list:
        idx, comp = t.split("_")
        parsed.append((int(idx), comp))
    return parsed


def prepare_batch(shear_comp, shear_mode, start, args):
    t0 = time.perf_counter()
    cutouts, psfs, cats, noise_levels = build_dataset(
        shear_comp=shear_comp, shear_mode=shear_mode, abs_shear_value=args.shear_value,
        noise_level=args.noise, index_range=[start * args.img_bs, (start + 1) * args.img_bs],
        directory=args.dir, workers=args.io_workers, use_threads=False)
    # Same MT19937 stream as src/shape_measurement.py
    rs = np.random.RandomState(20020620 + start)
    noises = rs.normal(0, 1, cutouts.shape) * noise_levels.reshape(-1, 1, 1)
    q = prepare_q_images(cutouts, psfs, noises, cats, sigma_arcsec=args.psf_fwhm / 2.355,
                         workers=args.io_workers, flim=10)
    return q.astype(np.float32), time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Exported cpu_src .npz model")
    p.add_argument("--noise", type=float, required=True)
    p.add_argument("--fname", required=True, help="Filename prefix")
    p.add_argument("--shear_tasks", nargs="+", default=["0_g1", "1_g1", "0_g2", "1_g2"])
    p.add_argument("--range", nargs=2, type=int, default=[0, 200], metavar=("START", "STOP"),
                   help="[start, stop) batch index; one batch = 100 img dirs = 10k galaxies")
    p.add_argument("--img_bs", type=int, default=100, help="img_ dirs per batch (100 galaxies each); keep 100 for production")
    p.add_argument("--dir", default="/work/hdd/bdsp/wenyinli/datasets/xlens_shear005")
    p.add_argument("--shear_value", type=float, default=0.05, help="|g| of the simulation")
    p.add_argument("--save_base", default="/work/hdd/bdsp/wenyinli/measurement_cpu/")
    p.add_argument("--psf_fwhm", type=float, default=0.85)
    p.add_argument("--nthreads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 32)))
    p.add_argument("--io_workers", type=int, default=16, help="processes for loading / q-image prep")
    p.add_argument("--precision", choices=["float32", "float64"], default="float32")
    args = p.parse_args()

    t_start = time.time()
    model = Model.load(args.model, precision=args.precision, isa="auto")
    folder = "nada" if args.noise == "adaptive" else format_number(args.noise)
    jobs = [(m, c, s) for m, c in parse_shear_tasks(args.shear_tasks)
            for s in range(args.range[0], args.range[1])]
    print(f"model={args.model} noise={args.noise} dir={args.dir} jobs={len(jobs)} nthreads={args.nthreads}", flush=True)

    # Prefetch the next batch's CPU-side preparation while the model measures the current one.
    with ThreadPoolExecutor(max_workers=1) as pool:
        submit = lambda j: pool.submit(prepare_batch, j[1], j[0], j[2], args)
        fut = submit(jobs[0]) if jobs else None
        for k, (mode, comp, start) in enumerate(jobs):
            out = os.path.join(args.save_base, folder, f"{comp}_{mode}")
            done = os.path.isfile(os.path.join(out, f"{args.fname}_Rana_{start}.npy"))
            q, t_prep = fut.result()
            fut = submit(jobs[k + 1]) if k + 1 < len(jobs) else None
            if done:
                print(f"skip {comp}_{mode} batch {start} (exists)", flush=True)
                continue
            t0 = time.perf_counter()
            res = model.measure_qstamps(q, n_tangent=2, nthreads=args.nthreads)
            t_cpu = time.perf_counter() - t0
            os.makedirs(out, exist_ok=True)
            np.save(os.path.join(out, f"{args.fname}_shapes_{start}.npy"), res.e)
            np.save(os.path.join(out, f"{args.fname}_Rana_{start}.npy"), res.R[:, None])
            print(f"{comp}_{mode} batch {start}: prep {t_prep:.1f}s  infer {t_cpu:.1f}s "
                  f"({len(q) / t_cpu:.0f} gal/s)", flush=True)
    print(f"Execution time: {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
