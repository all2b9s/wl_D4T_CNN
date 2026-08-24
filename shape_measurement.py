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

# =====================================================================
# Configuration (adjust for your own environment)
# =====================================================================
# Path to the pretrained model checkpoint. Pick one that exists in ./models/,
# e.g. './models/forward8_CNN_nada_50ep.pth' or './models/F8_fpfs_l5c32r01_50ep.pth'.
model_path = './models/forward8_CNN_nada_50ep.pth'

# Directory containing the simulated datasets (produced by build_dataset).
# Replace this with the path to your own simulated cutouts.
dir = ''

# Output directory where shape-measurement results are saved.
save_base = ''

device = ("cuda" if torch.cuda.is_available() else "cpu")
# model = Forward8_fixW_CNN(num_layers = 5, base_channels = 32, res_factor=0.1).to(device)
model = Forward8_fixW_CNN(num_layers = 5, base_channels = 32, res_factor=0.1).to(device)

model.load_state_dict(torch.load(model_path, map_location="cpu"))
_ = model.eval()

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

def loop_shape_measurement(
    shear_tasks =[
            (0, "g1"),
            (1, "g1"),
            (0, "g2"),
            (1, "g2"),
        ],
    calibrator = None,
    imgBs = 100,            # batch size for npy file
    img_nB_range = [0,100],           # number of npy batch
    noise_level = 0,
    fname = 'p85',
    dtype = 'float32',
    psf_fwhm = 0.85,    
    ):
    if noise_level == 'adaptive':
        folder_name = 'nada'
    else:
        folder_name = format_number(noise_level)
    print(f"Starting shape measurement and calibration for noise level: {noise_level}")
    print(f"Using filename prefix: {fname}")
    print(f"Selected shear tasks: {shear_tasks}")
    print(f"Dataset directory: {dir}")

    for shear_mode, shear_comp in shear_tasks:
        for start in range(img_nB_range[0], img_nB_range[1]):
            print(f'Processing {shear_comp}_{shear_mode}, batch {start}')
            range_idx = [start*imgBs, (start+1)*imgBs]
            all_cutouts, all_psfs, all_cats, noise_levels = build_dataset(
                    shear_comp=shear_comp, shear_mode=shear_mode, abs_shear_value=0.02,
                    noise_level=noise_level, index_range=range_idx, directory=dir, workers=32, use_threads=False
                )
            
            np.random.seed(20020620+start)
            noises = np.random.normal(0, 1, all_cutouts.shape)*noise_levels.reshape(-1,1,1)
            q_imgs = prepare_q_images(
                    all_cutouts, all_psfs, noises, all_cats,
                    sigma_arcsec=psf_fwhm/2.355,
                    workers=32, flim=10
                )
            q_imgs = q_imgs[:,:,1:,1:]

            if dtype == 'float32':
                q_imgs = q_imgs.astype(np.float32)
            elif dtype == 'float64':
                q_imgs = q_imgs.astype(np.float64)
            
            shapes, R_ana = calibrator.shape_measure(q_imgs,batch_size=800)
            # Output directory — replace with your own path or pass --save_base
            save_name = os.path.join(save_base, folder_name, f'{shear_comp}_{shear_mode}')
            if not os.path.exists(save_name):
                os.makedirs(save_name)
            np.save(os.path.join(save_name, f'{fname}_shapes_{start}.npy'), shapes)
            np.save(os.path.join(save_name, f'{fname}_Rana_{start}.npy'), R_ana)
            del all_cutouts, all_psfs, all_cats, noises, q_imgs, shapes, R_ana



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
    args = p.parse_args()

    noise_level = args.noise
    fname = args.fname
    shear_tasks = parse_shear_tasks(args.shear_tasks)
    dtype = 'float32'

    print(f"Path of the model dict: {model_path}")
    cal_CNN = Calibrator(model, device, dtype=dtype)
    # --- Run main processes ---
    loop_shape_measurement(
        shear_tasks=shear_tasks,
        calibrator=cal_CNN,
        imgBs=100,          # batch size per npy
        img_nB_range=[2000,4000],         # number of npy batches
        noise_level=noise_level,
        psf_fwhm=0.85,
        fname=fname,
        dtype=dtype,
    )

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
