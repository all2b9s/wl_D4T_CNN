import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import galsim
import torch
from tqdm import tqdm
from src.anacal.cal_toolkit import load_img, build_dataset
from src.anacal.batch_calibration import prepare_q_images, Calibrator, get_biases
from src.architecture.single_gal_CNN import R180Inv_CNN_GeLU, D4T_CNN_GeLU,SmoothCNN_GeLU
from src.architecture.forward8_CNN import Forward8_fixW_CNN, SmoothCNN_GeLU_mocked,Forward8_fixW_nores_CNN, Forward8_simp_CNN
import argparse, os, random,re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# =====================================================================
# Configuration (adjust for your own environment)
# =====================================================================
# Default pretrained model checkpoint (override with --model_path). Pick one that
# exists in ./models/, e.g. './models/forward8_CNN_nada_50ep.pth' or
# './models/F8_fpfs_l5c32r01_50ep.pth'.
# Anchored to the repository root (parent of src/) so it is CWD-independent.
_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = str(_REPO_ROOT / "models" / "forward8_CNN_nada_50ep.pth")

# Directory containing the simulated datasets (produced by build_dataset).
# Replace this with the path to your own simulated cutouts.
dir = '/work/nvme/bfmo/wenyinli/datasets/xlens_shift'

# Output directory where shape-measurement results are saved.
save_base = '/work/hdd/bfmo/wenyinli/measurement/'

device = ("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path):
    model = Forward8_fixW_CNN(num_layers = 5, base_channels = 32, res_factor=0.1).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

shear_tasks = [
            (0, "g1"),
            (1, "g1"),
            (0, "g2"),
            (1, "g2"),
        ]

def format_number(x: float) -> str:
    s = f"{x:.3f}".rstrip('0').rstrip('.')  # keep up to 3 decimal places, trim trailing zeros
    s = s.replace('.', '')                  # remove the decimal point
    return f"n{s}"

def _prepare_cpu_batch(shear_comp, shear_mode, start, imgBs, noise_level, dtype, psf_fwhm, shear_value=0.02):
    """CPU part of one iteration: load cutouts, draw noise, build q-images."""
    t0 = time.perf_counter()
    range_idx = [start*imgBs, (start+1)*imgBs]
    all_cutouts, all_psfs, all_cats, noise_levels = build_dataset(
            shear_comp=shear_comp, shear_mode=shear_mode, abs_shear_value=shear_value,
            noise_level=noise_level, index_range=range_idx, directory=dir, workers=32, use_threads=False
        )

    # Same MT19937 stream as np.random.seed(s); np.random.normal(...), but
    # without touching global state (this runs in a prefetch thread).
    rs = np.random.RandomState(20020620+start)
    noises = rs.normal(0, 1, all_cutouts.shape)*noise_levels.reshape(-1,1,1)
    q_imgs = prepare_q_images(
            all_cutouts, all_psfs, noises, all_cats,
            sigma_arcsec=psf_fwhm/2.355,
            workers=32, flim=10
        )
    #q_imgs = q_imgs[:,:,1:,1:]

    if dtype == 'float32':
        q_imgs = q_imgs.astype(np.float32)
    elif dtype == 'float64':
        q_imgs = q_imgs.astype(np.float64)
    return q_imgs, time.perf_counter() - t0


def loop_shape_measurement(
    calibrator: Calibrator,
    shear_tasks =[
            (0, "g1"),
            (1, "g1"),
            (0, "g2"),
            (1, "g2"),
        ],
    imgBs = 100,            # batch size for npy file
    img_nB_range = [0,100],           # number of npy batch
    noise_level = 0,
    fname = 'p85',
    dtype = 'float32',
    psf_fwhm = 0.85,
    shear_value = 0.02,
    batch_size = 800,       # GPU batch size for calibrator.shape_measure
    ):
    if noise_level == 'adaptive':
        folder_name = 'nada'
    else:
        folder_name = format_number(noise_level)
    print(f"Starting shape measurement and calibration for noise level: {noise_level}")
    print(f"Using filename prefix: {fname}")
    print(f"Selected shear tasks: {shear_tasks}")
    print(f"Dataset directory: {dir}")

    jobs = [(shear_mode, shear_comp, start)
            for shear_mode, shear_comp in shear_tasks
            for start in range(img_nB_range[0], img_nB_range[1])]

    # Prefetch the next iteration's CPU work while the GPU measures the current one.
    with ThreadPoolExecutor(max_workers=1) as prefetch:
        submit = lambda job: prefetch.submit(
            _prepare_cpu_batch, job[1], job[0], job[2], imgBs, noise_level, dtype, psf_fwhm, shear_value)
        future = submit(jobs[0]) if jobs else None
        for k, (shear_mode, shear_comp, start) in enumerate(jobs):
            print(f'Processing {shear_comp}_{shear_mode}, batch {start}')
            t0 = time.perf_counter()
            q_imgs, t_prep = future.result()
            t_wait = time.perf_counter() - t0
            future = submit(jobs[k+1]) if k + 1 < len(jobs) else None

            t0 = time.perf_counter()
            shapes, R_ana = calibrator.shape_measure(q_imgs, batch_size=batch_size)
            t_gpu = time.perf_counter() - t0

            t0 = time.perf_counter()
            # Output directory — replace with your own path or pass --save_base
            save_name = os.path.join(save_base, folder_name, f'{shear_comp}_{shear_mode}')
            os.makedirs(save_name, exist_ok=True)
            np.save(os.path.join(save_name, f'{fname}_shapes_{start}.npy'), shapes)
            np.save(os.path.join(save_name, f'{fname}_Rana_{start}.npy'), R_ana)
            t_save = time.perf_counter() - t0
            print(f'  cpu_prep {t_prep:.2f}s (waited {t_wait:.2f}s)  gpu {t_gpu:.2f}s  save {t_save:.2f}s')
            del q_imgs, shapes, R_ana



def parse_shear_tasks(task_list):
    """Parse shear tasks like ['0_g1', '1_g2'] → [(0, 'g1'), (1, 'g2')]"""
    parsed = []
    for t in task_list:
        try:
            idx, shear = t.split('_')
            parsed.append((int(idx), shear))
        except ValueError:
            raise ValueError(f"Invalid shear task format '{t}', expected like '0_g1' or '1_g2'")
    return parsed

if __name__ == "__main__":
    start_time = time.time()

    p = argparse.ArgumentParser()
    p.add_argument("--noise", type=float, required=True, help="Noise level for calibration")
    p.add_argument("--fname", type=str, default="test", help="Filename prefix")
    p.add_argument(
        "--shear_tasks",
        nargs="+",
        type=str,
        default=["0_g1", "1_g1", "0_g2", "1_g2"],
        help="List of shear tasks like: 0_g1 1_g1 0_g2 1_g2"
    )
    p.add_argument("--range", nargs=2, type=int, default=[2000, 4000], metavar=("START", "STOP"),
                   help="[start, stop) npy batch index range (100 galaxies per batch)")
    p.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                   help="Path to the Forward8_fixW_CNN checkpoint (.pth)")
    p.add_argument("--dir", type=str, default=dir,
                   help="Dataset directory holding {g1,g2}_{mode}_val_*/img_*/")
    p.add_argument("--save_base", type=str, default=save_base,
                   help="Output root; results go to <save_base>/<noise_tag>/<comp>_<mode>/")
    p.add_argument("--batch_size", type=int, default=800, help="GPU batch size for shape_measure")
    p.add_argument("--shear_value", type=float, default=0.02, help="Absolute simulated shear amplitude")
    args = p.parse_args()
    dir, save_base = args.dir, args.save_base

    noise_level = args.noise
    fname = args.fname
    shear_tasks = parse_shear_tasks(args.shear_tasks)
    dtype = 'float32'

    print(f"Path of the model dict: {args.model_path}")
    print(f"Batch range: {args.range}")
    cal_CNN = Calibrator(load_model(args.model_path), device, dtype=dtype)
    # --- Run main processes ---
    loop_shape_measurement(
        shear_tasks=shear_tasks,
        calibrator=cal_CNN,
        imgBs=100,          # batch size per npy
        img_nB_range=args.range,         # [start, stop) npy batches
        noise_level=noise_level,
        psf_fwhm=0.85,
        fname=fname,
        dtype=dtype,
        batch_size=args.batch_size,
        shear_value=args.shear_value,
    )

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
