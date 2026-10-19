"""Step 1: teacher CNN shapes + 2x2 R_ana on the (noiseless) simulation.

Same calls as src/shape_measurement.py::loop_shape_measurement, but the architecture comes
from the yaml and the data/output dirs are explicit. Output per npy batch k:
    <step_dir>/teacher_shape_shapes_{k}.npy   (N, 2)
    <step_dir>/teacher_shape_Rana_{k}.npy     (N, 1, 2, 2)
"""
import numpy as np

from ml_wl.steps._common import d, load_job_args, sg_job


def batch_ranges(cfg, sim=None):
    sim = sim or cfg["sim"]
    s, n, j = sim["batch_start"], sim["n_batches"], cfg["measure"]["n_jobs"]
    per = -(-n // j)
    out = []
    for i in range(j):
        a, b = s + i * per, min(s + (i + 1) * per, s + n)
        if a < b:
            out.append((a, b))
    return out


def jobs(cfg):
    if cfg.get("blended") is not None:
        from ml_wl.steps.blended import job
        return job(cfg, "measure")
    return [dict(lines=sg_job("measure", cfg), cwd="sg") for _ in batch_ranges(cfg)]


def main():
    cfg, job = load_job_args("measure")
    run(cfg, job, "measure", cfg["sim"])


def run(cfg, job, step, sim):
    """Measure npy batches of one job on `sim` (teacher sim, or the student sim for step s_measure)."""
    import os
    import torch
    from src.anacal.batch_calibration import Calibrator, prepare_q_images
    from src.anacal.cal_toolkit import build_dataset
    from ml_wl.models import build_model

    lo, hi = batch_ranges(cfg, sim)[job]
    out = d(cfg, step)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t = cfg["teacher_cnn"]
    model = build_model(t["arch"], t["kwargs"]).to(device)
    model.load_state_dict(torch.load(t["checkpoint"], map_location="cpu"))
    model.eval()
    cal = Calibrator(model, device, dtype="float32")
    noise_level = float(cfg["measure"]["noise"])
    print(f"{step} batches [{lo},{hi}) noise={noise_level} model={t['checkpoint']}", flush=True)
    for k in range(lo, hi):
        cut, psf, cat, nl = build_dataset(
            shear_comp=sim["comp"], shear_mode=sim["mode"], abs_shear_value=sim["abs_shear"],
            noise_level=noise_level, index_range=[k * 100, (k + 1) * 100], directory=sim["dir"],
            workers=32, use_threads=False)
        np.random.seed(20020620 + k)
        noises = np.random.normal(0, 1, cut.shape) * nl.reshape(-1, 1, 1)
        q = prepare_q_images(cut, psf, noises, cat, sigma_arcsec=sim["psf_fwhm"] / 2.355,
                             workers=32, flim=10).astype(np.float32)
        shapes, R = cal.shape_measure(q, batch_size=800)
        np.save(os.path.join(out, f"{sim['fname']}_shapes_{k}.npy"), shapes)
        np.save(os.path.join(out, f"{sim['fname']}_Rana_{k}.npy"), R)
        print(f"batch {k} done", flush=True)


if __name__ == "__main__":
    main()
