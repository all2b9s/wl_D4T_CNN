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


dir = '/work/nvme/bfmo/wenyinli/datasets/xlens_shift'
#dir = '/work/hdd/bdsp/wenyinli/datasets/xlens_fixed'

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

def format_number(x: float) -> str:
    s = f"{x:.3f}".rstrip('0').rstrip('.')  # keep up to 3 decimal places, trim trailing zeros
    s = s.replace('.', '')                  # remove the decimal point
    return f"n{s}"

def get_calibration_biases(
        grid_per_img = 100,     # imgs per npy file
        imgBs = 100,            # batch size for npy file
        img_nB_range = [0,100],           # number of npy batch
        folder_name = 'n0',
        fname = 'p85',
        is_twin = True,
        workers = 32,
        ):

    img_nB = img_nB_range[1]-img_nB_range[0]
    Nimg = imgBs*(img_nB_range[1]-img_nB_range[0])*grid_per_img
    print(Nimg)

    g_all = np.zeros([grid_per_img*imgBs*img_nB, 4, 2])
    R_all = np.zeros([grid_per_img*imgBs*img_nB, 4, 2, 2])
    shear_tasks = [
                (1, "g1"),
                (0, "g1"),
                (1, "g2"),
                (0, "g2"),
            ]
    dir_gr = f'/projects/bfmo/wenyinli/datasets/xlens_sims/{folder_name}/'
    
    for i, (shear_mode, shear_comp) in enumerate(shear_tasks):
        if not os.path.isfile(f'{dir_gr}{shear_comp}_{shear_mode}/{fname}_shapes_{img_nB_range[1]-1}.npy'):
            print(f"Required file '{dir_gr}{shear_comp}_{shear_mode}/{fname}_shapes_{img_nB_range[1]-1}.npy' not found.")
            return 0

    for i, (shear_mode, shear_comp) in enumerate(shear_tasks):
        for index in range(img_nB_range[0], img_nB_range[1]):
            begin = index*imgBs*grid_per_img
            end = (index+1)*imgBs*grid_per_img
            g_all[begin:end,i] = np.load(f'{dir_gr}{shear_comp}_{shear_mode}/{fname}_shapes_{index}.npy')
            R_all[begin:end,i] = np.load(f'{dir_gr}{shear_comp}_{shear_mode}/{fname}_Rana_{index}.npy')[:,0]

    #dir = '/work/hdd/bdsp/wenyinli/datasets/xlens_gal/'
    all_cutouts, all_psfs, all_cats, noise_level = build_dataset(
            shear_comp='g1', shear_mode=0, abs_shear_value=0.02,
            noise_level=0, 
            index_range=[0,imgBs*(img_nB_range[1]-img_nB_range[0])], 
            directory=dir, 
            workers=workers, 
            return_imgs=False,
            use_threads=False
        )

    cal_result = get_biases(g_all, R_all, 
                                    bs_size=grid_per_img*imgBs*(img_nB_range[1]-img_nB_range[0]), 
                                    bs_times=100,
                                    n_factor=1,
                                    is_twin=True,
                                    n_jobs=16,
                                    )
    #print(cal_result)
    print(f'm={cal_result[0]} +- {cal_result[2]}, c={cal_result[1]} +- {cal_result[3]}')

    save_name = f'/work/hdd/bfmo/wenyinli/datasets/xlens_cali_results/{folder_name}/'
    if not os.path.exists(save_name):
        os.makedirs(save_name)
    np.save(save_name+f'{fname}_g_all.npy', g_all)
    np.save(save_name+f'{fname}_R_all.npy', R_all)
    all_cats.to_csv(save_name+f'{fname}_cats_all.csv')


if __name__ == "__main__":
    start_time = time.time()

    p = argparse.ArgumentParser()
    p.add_argument("--noise", type=float, required=True, help="Noise level for calibration")
    p.add_argument("--fname", type=str, default="test", help="Filename prefix")
    args = p.parse_args()

    noise_level = args.noise
    fname = args.fname

    print(f"Starting shape measurement and calibration for noise level: {noise_level}")
    print(f"Using filename prefix: {fname}")


    get_calibration_biases(
        grid_per_img=100,
        imgBs=100,
        img_nB_range= [0,2000],
        folder_name=format_number(noise_level),
        is_twin=True,
        fname=fname,
    )

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")