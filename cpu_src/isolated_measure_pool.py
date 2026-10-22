"""CPU shape measurement of isolated-galaxy xlens cutouts with one unified worker pool.

Replaces the prefetch-thread design of ``isolated_measure.py``.  Every worker is
pinned to one core and handles one ``img_`` directory (100 galaxies) end to end:
load -> q-image -> 1-thread inference.  There is no separate prep pool, so the
cores are neither idle between stages nor oversubscribed.

Results are bit-identical to ``isolated_measure.py``: the per-batch measurement
noise is still drawn in the parent from the same ``RandomState(20020620 + batch)``
stream (shared with the workers through shared memory).  Output files and the
skip-if-done behaviour are unchanged, so a timed-out job can simply be resubmitted.

Run from ``single_galaxies/``:
    python -u -m cpu_src.isolated_measure_pool --model cpu_src/runs/X.npz --noise 0.594 \
        --fname NAME --shear_tasks 0_g1 --range 0 100
"""
import argparse
import os
import signal
import sys
import threading
import time
import multiprocessing as mp
from multiprocessing import resource_tracker, shared_memory

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from cpu_src import Model
from cpu_src.isolated_measure import parse_shear_tasks
from src.anacal.batch_calibration import _process_image
from src.anacal.cal_toolkit import format_number, load_img

STAMP = 64            # cutout size
GAL_PER_DIR = 100     # galaxies per img_ dir (checked at run time)
_cfg = {}
_state = {}
_counter = mp.Value("i", 0)


def _init_worker(cfg, cores):
    with _counter.get_lock():
        w = _counter.value
        _counter.value += 1
    os.sched_setaffinity(0, {cores[w % len(cores)]})
    _cfg.update(cfg)
    _state["model"] = Model.load(cfg["model"], precision=cfg["precision"], isa="auto")
    _state["shm"] = {}


def _attach(name, shape):
    shms = _state["shm"]
    if name not in shms:
        for old in shms.values():     # a new batch started: drop finished ones
            old.close()
        shms.clear()
        shm = shared_memory.SharedMemory(name=name)
        # The parent owns the segment; stop this worker's tracker from unlinking it.
        resource_tracker.unregister(shm._name, "shared_memory")
        shms[name] = shm
    return np.ndarray(shape, dtype=np.float64, buffer=shms[name].buf)


def _task(args):
    key, comp, mode, start, d, shm_name, nrows = args
    c = _cfg
    t0 = time.perf_counter()
    cut, psf, cat, _ = load_img(
        shear_comp=comp, shear_mode=mode, abs_shear_value=c["shear_value"],
        index=start * c["img_bs"] + d, noise_level=c["noise"], directory=c["dir"])
    n = len(cut)
    noises = _attach(shm_name, (nrows, STAMP, STAMP))[d * n:(d + 1) * n]
    centers = np.ones((n, 2), dtype=np.float64) * (cut.shape[1] // 2)
    centers[:, 0] += (cat["image_x"] - (cat["image_x"] + 0.5) // 1).to_numpy(np.float64)
    centers[:, 1] += (cat["image_y"] - (cat["image_y"] + 0.5) // 1).to_numpy(np.float64)
    t1 = time.perf_counter()
    q = np.array([_process_image(i, images=cut, psfs=psf, centers=centers, noises=noises,
                                 pixel_scale=0.2, sigma_arcsec=c["psf_fwhm"] / 2.355, flim=10)
                  for i in range(n)]).astype(np.float32)
    t2 = time.perf_counter()
    res = _state["model"].measure_qstamps(q, n_tangent=2, nthreads=1)
    t3 = time.perf_counter()
    return key, d, res.e, res.R, (t1 - t0, t2 - t1, t3 - t2)


def _save(path, arr):
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Exported cpu_src .npz model")
    p.add_argument("--noise", type=float, required=True)
    p.add_argument("--fname", required=True, help="Filename prefix")
    p.add_argument("--shear_tasks", nargs="+", default=["0_g1", "1_g1", "0_g2", "1_g2"])
    p.add_argument("--range", nargs=2, type=int, default=[0, 200], metavar=("START", "STOP"),
                   help="[start, stop) batch index; one batch = 100 img dirs = 10k galaxies")
    p.add_argument("--img_bs", type=int, default=100, help="img_ dirs per batch; keep 100 for production")
    p.add_argument("--dir", default="/work/hdd/bdsp/wenyinli/datasets/xlens_shear005")
    p.add_argument("--shear_value", type=float, default=0.05, help="|g| of the simulation")
    p.add_argument("--save_base", default="/work/hdd/bdsp/wenyinli/measurement_cpu/")
    p.add_argument("--psf_fwhm", type=float, default=0.85)
    p.add_argument("--workers", type=int, default=0,
                   help="worker processes, one pinned core each (default: allowed cores - 2)")
    p.add_argument("--parent_cores", type=int, default=2,
                   help="cores kept for the parent (noise draw, collection, saving)")
    p.add_argument("--max_batches_inflight", type=int, default=4,
                   help="batches whose noise is held in shared memory (328 MB each)")
    p.add_argument("--precision", choices=["float32", "float64"], default="float32")
    # Accepted so callers of isolated_measure.py keep working; unused here.
    p.add_argument("--nthreads", type=int, default=None)
    p.add_argument("--io_workers", type=int, default=None)
    args = p.parse_args()
    if args.noise == "adaptive":
        raise SystemExit("adaptive noise is not supported by the pool version")

    allowed = sorted(os.sched_getaffinity(0))
    nparent = min(args.parent_cores, max(len(allowed) - 1, 0))
    cores = allowed[nparent:]
    nw = args.workers or len(cores)
    nw = min(nw, len(cores))
    os.sched_setaffinity(0, set(allowed[:max(nparent, 1)]))

    folder = format_number(args.noise)
    jobs = []
    for mode, comp in parse_shear_tasks(args.shear_tasks):
        out = os.path.join(args.save_base, folder, f"{comp}_{mode}")
        for s in range(args.range[0], args.range[1]):
            if os.path.isfile(os.path.join(out, f"{args.fname}_Rana_{s}.npy")):
                print(f"skip {comp}_{mode} batch {s} (exists)", flush=True)
            else:
                jobs.append((mode, comp, s, out))
    print(f"model={args.model} noise={args.noise} dir={args.dir} jobs={len(jobs)} workers={nw}", flush=True)
    if not jobs:
        return 0

    cfg = dict(model=args.model, precision=args.precision, shear_value=args.shear_value,
               noise=args.noise, dir=args.dir, img_bs=args.img_bs, psf_fwhm=args.psf_fwhm)
    shapes = {}                       # job index -> noise shared memory + collected results
    lock = threading.Lock()
    inflight = threading.Semaphore(args.max_batches_inflight)
    errors = []
    all_done = threading.Event()
    n_done = [0]
    t_start = time.time()
    busy = np.zeros(3)

    def finish(ji):
        mode, comp, s, out = jobs[ji]
        st = shapes.pop(ji)
        e = np.concatenate([st["res"][d][0] for d in range(args.img_bs)])
        R = np.concatenate([st["res"][d][1] for d in range(args.img_bs)])
        os.makedirs(out, exist_ok=True)
        _save(os.path.join(out, f"{args.fname}_shapes_{s}.npy"), e)
        _save(os.path.join(out, f"{args.fname}_Rana_{s}.npy"), R[:, None])
        st["shm"].close()
        st["shm"].unlink()
        inflight.release()
        n_done[0] += 1
        el = time.time() - t_start
        print(f"{comp}_{mode} batch {s} done ({n_done[0]}/{len(jobs)}) elapsed {el:.0f}s "
              f"{n_done[0] * len(e) / el:.0f} gal/s avg", flush=True)
        if n_done[0] == len(jobs):
            all_done.set()

    def on_result(r):
        ji, d, e, R, tm = r
        with lock:
            busy[:] += tm
            st = shapes[ji]
            st["res"][d] = (e, R)
            if len(st["res"]) == args.img_bs:
                finish(ji)

    def on_error(exc):
        errors.append(exc)
        print(f"ERROR in worker: {exc!r}", flush=True)
        all_done.set()

    pool = mp.Pool(nw, initializer=_init_worker, initargs=(cfg, cores))

    def feeder():
        for ji, (mode, comp, s, out) in enumerate(jobs):
            while not inflight.acquire(timeout=1):
                if errors:
                    return
            # Same MT19937 stream and float32 noise level as isolated_measure.py.
            rs = np.random.RandomState(20020620 + s)
            nrows = args.img_bs * GAL_PER_DIR
            nl = np.full(nrows, args.noise, dtype=np.float32).reshape(-1, 1, 1)
            noise = rs.normal(0, 1, (nrows, STAMP, STAMP)) * nl
            shm = shared_memory.SharedMemory(create=True, size=noise.nbytes)
            np.ndarray(noise.shape, dtype=np.float64, buffer=shm.buf)[:] = noise
            del noise
            with lock:
                shapes[ji] = {"shm": shm, "res": {}}
            for d in range(args.img_bs):
                pool.apply_async(_task, ((ji, comp, mode, s, d, shm.name, nrows),),
                                 callback=on_result, error_callback=on_error)

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    th = threading.Thread(target=feeder, daemon=True)
    th.start()
    try:
        all_done.wait()
    finally:
        pool.terminate()
        pool.join()
        with lock:
            for st in shapes.values():
                st["shm"].close()
                try:
                    st["shm"].unlink()
                except FileNotFoundError:
                    pass
    if errors:
        print("FAILED", flush=True)
        return 1
    tot = busy.sum()
    print(f"busy core-s: load {busy[0]:.0f} q {busy[1]:.0f} infer {busy[2]:.0f}; "
          f"worker utilization {tot / (nw * (time.time() - t_start)) * 100:.0f}%", flush=True)
    print(f"Execution time: {time.time() - t_start:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
