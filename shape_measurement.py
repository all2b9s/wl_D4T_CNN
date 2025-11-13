import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import galsim
import torch
from tqdm import tqdm
from src.anacal.cal_toolkit import load_img, build_dataset
from src.anacal.batch_calibration import prepare_q_images, Calibrator, get_biases
from src.architecture.single_gal_CNN import R180Inv_CNN_GeLU, D4T_CNN_GeLU,SmoothCNN_GeLU
from src.architecture.forward8_CNN import Forward8_fixW_CNN, SmoothCNN_GeLU_mocked,Forward8_fixW_nores_CNN, Forward8_attnW_CNN
import argparse, os, random
import time


device = ("cuda" if torch.cuda.is_available() else "cpu")
#model_path = './models/F8_TBigUni_l5c32r0_5ep.pth'
model_path = './models/forward8_CNN_nada_50ep.pth'
#model = D4T_CNN_GeLU(num_layers = 8).to(device)
model = Forward8_fixW_CNN(num_layers = 5, base_channels = 32, res_factor=0.1).to(device)
#model = Forward8_attnW_CNN(num_layers = 5, base_channels = 64, res_factor=1).to(device)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
#model.load_state_dict(torch.load('./models/avg_CNN_n0.1_50ep.pth', map_location="cpu"))
_ = model.eval()
cal_CNN = Calibrator(model, device)
dir = '/work/nvme/bfmo/wenyinli/datasets/xlens_shift'
#dir = '/work/hdd/bdsp/wenyinli/datasets/xlens_fixed'

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
    fname = 'p85'    
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
            
            np.random.seed(20020520+start)
            noises = np.random.normal(0, 1, all_cutouts.shape)*noise_levels.reshape(-1,1,1)
            q_imgs = prepare_q_images(
                    all_cutouts, all_psfs, noises, all_cats,
                    sigma_arcsec=0.85/2.355,
                    workers=32, flim=10
                )
            q_imgs = q_imgs[:,:,1:,1:]
            
            shapes, R_ana = calibrator.shape_measure(q_imgs,batch_size=700)
            save_name = f'/projects/bfmo/wenyinli/datasets/xlens_sims/{folder_name}/{shear_comp}_{shear_mode}/'
            if not os.path.exists(save_name):
                os.makedirs(save_name)
            np.save(save_name + f'{fname}_shapes_{start}.npy', shapes)
            np.save(save_name + f'{fname}_Rana_{start}.npy', R_ana)
            del all_cutouts, all_psfs, all_cats, noises, q_imgs, shapes, R_ana

def get_calibration_biases(
        grid_per_img = 100,     # imgs per npy file
        imgBs = 100,            # batch size for npy file
        img_nB = 100,           # number of npy batch
        folder_name = 'n0',
        fname = 'p85',
        workers = 32,
        ):

    Nimg = imgBs*img_nB*grid_per_img
    print(Nimg)

    g_all = np.zeros([grid_per_img*imgBs*img_nB, 4, 2])
    R_all = np.zeros([grid_per_img*imgBs*img_nB, 4, 2, 2])
    shear_tasks = [
                (1, "g1"),
                (0, "g1"),
                (1, "g2"),
                (0, "g2"),
            ]
    dir_gr = f'/projects/bdsp/wenyinli/datasets/xlens_sims/{folder_name}/'
    for i, (shear_mode, shear_comp) in enumerate(shear_tasks):
        for index in range(img_nB):
            begin = index*imgBs*grid_per_img
            end = (index+1)*imgBs*grid_per_img
            g_all[begin:end,i] = np.load(f'{dir_gr}{shear_comp}_{shear_mode}/{fname}_shapes_{index}.npy')
            R_all[begin:end,i] = np.load(f'{dir_gr}{shear_comp}_{shear_mode}/{fname}_Rana_{index}.npy')[:,0]

    #dir = '/work/hdd/bdsp/wenyinli/datasets/xlens_gal/'
    all_cutouts, all_psfs, all_cats, noise_level = build_dataset(
            shear_comp='g1', shear_mode=2, abs_shear_value=0.02,
            noise_level=0, 
            index_range=[0,imgBs*img_nB], 
            directory=dir, 
            workers=workers, 
            return_imgs=False,
            use_threads=False
        )

    cal_result = get_biases(g_all, R_all, 
                                    bs_size=grid_per_img*imgBs*img_nB, 
                                    bs_times=100,
                                    n_factor=1,
                                    n_jobs=16,
                                    )
    #print(cal_result)
    print(f'm={cal_result[0]} +- {cal_result[2]}, c={cal_result[1]} +- {cal_result[3]}')

    save_name = f'/work/nvme/bfmo/wenyinli/datasets/xlens_sims/{folder_name}/'
    if not os.path.exists(save_name):
        os.makedirs(save_name)
    np.save(save_name+f'{fname}_g_all.npy', g_all)
    np.save(save_name+f'{fname}_R_all.npy', R_all)
    all_cats.to_csv(save_name+f'{fname}_cats_all.csv')



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


    print(f"Path of the model dict: {model_path}")

    # --- Run main processes ---
    loop_shape_measurement(
        shear_tasks=shear_tasks,
        calibrator=cal_CNN,
        imgBs=100,          # batch size per npy
        img_nB_range=[1400,2000],         # number of npy batches
        noise_level=noise_level,
        fname=fname,
    )

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
