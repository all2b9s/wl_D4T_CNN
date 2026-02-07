import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import galsim
import torch
from tqdm import tqdm
from src.anacal.cal_toolkit import selection_response
from src.anacal.batch_calibration import get_biases
import argparse, os, random, re
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

def parse_info(fname, folder, cal_result):
    # ==== parse noise ====
    # folder = "n0594"
    def parse_noise_level(folder):
        s = folder[1:]
        if len(s) == 4:
            return float(s) / 1000.0
        elif len(s) == 3:
            return float(s) / 100.0
        elif len(s) == 2:
            return float(s) / 10.0
        elif len(s) == 1:
            return float(s)
        else:
            raise ValueError(f"Unexpected noise level format: {folder}")

    noise_level = parse_noise_level(folder)

    
    # ==== parse psf_fwhm ====
    # fname = "p85_f8_2e7_m253"
    # 取 p85 -> 8.5 / 10 = 0.85
    psf_match = re.search(r"p(\d+)", fname)
    psf_fwhm = float(psf_match.group(1)) / 100.0 if psf_match else None

    # ==== extract values ====
    m, c, m_err, c_err = cal_result

    # ==== output string ====
    out = f"{noise_level},{psf_fwhm},{fname},{m.tolist()},{m_err.tolist()},{c.tolist()},{c_err.tolist()}"
    return out

def get_calibration_biases(
        grid_per_img = 100,     # imgs per npy file
        imgBs = 100,            # batch size for npy file
        img_nB_range = [0,100],           # number of npy batch
        folder_name = 'n0',
        fname = 'p85',
        is_twin = True,
        workers = 16,
        mag_cut = None,
        flux_name = 'fpfs_flux',
        ):
    zero_point = 30.0
    img_nB = img_nB_range[1]-img_nB_range[0]
    Nimg = imgBs*(img_nB_range[1]-img_nB_range[0])*grid_per_img
    print(Nimg)

    g_all = np.zeros([grid_per_img*imgBs*img_nB, 5, 2])
    R_all = np.zeros([grid_per_img*imgBs*img_nB, 4, 2, 2])
    if mag_cut is not None:
        flux_all = np.zeros([grid_per_img*imgBs*img_nB,5,1])
        Rflux_all = np.zeros([grid_per_img*imgBs*img_nB,1,2])
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
            if mag_cut is not None:
                flux_all[begin:end,i] = np.load(f'{dir_gr}{shear_comp}_{shear_mode}/{flux_name}_m00_{index}.npy')


    if mag_cut is not None:
        for index in range(img_nB_range[0], img_nB_range[1]):
            begin = index*imgBs*grid_per_img
            end = (index+1)*imgBs*grid_per_img
            flux_all[begin:end,4] = np.load(f'{dir_gr}g1_2/{flux_name}_m00_{index}.npy')
            g_all[begin:end,4] = np.load(f'{dir_gr}g1_2/{fname}_shapes_{index}.npy') 
            Rflux_all[begin:end,0] = np.load(f'{dir_gr}g1_2/{flux_name}_Rm00_{index}.npy') # [ngal, 1, 2]
        R_sel = selection_response(
            flux=flux_all[:,4,0],
            R_flux=Rflux_all[:,0],
            shapes=g_all[:,4,:],
            mag_cut=mag_cut,
        )
        print(f"Selection response for flux cut at mag {mag_cut}: {R_sel}")
        R_all[:,:,1,1] += R_sel[1]
        R_all[:,:,0,0] += R_sel[0]
        mask = zero_point - 2.5 * np.log10(flux_all[:,:4,0]) < mag_cut
        print(f"Number of galaxies after mag cut at {mag_cut}: {np.sum(mask)/4}")



    cal_result = get_biases(g_all[:,:4], R_all, mask=mask,
                                    bs_size=grid_per_img*imgBs*(img_nB_range[1]-img_nB_range[0]), 
                                    bs_times=100,
                                    n_factor=1,
                                    is_twin=is_twin,
                                    n_jobs=workers,
                                    chunk_bs=20,
                                    )
    #print(cal_result)
    
    print(parse_info(fname, folder_name, cal_result))

    save_name = f'/work/hdd/bfmo/wenyinli/datasets/xlens_cali_results/{folder_name}/'
    if not os.path.exists(save_name):
        os.makedirs(save_name)
    np.save(save_name+f'{fname}_g_all.npy', g_all)
    np.save(save_name+f'{fname}_R_all.npy', R_all)
    #all_cats.to_csv(save_name+f'{fname}_cats_all.csv')
    with open("./logs/calibration_result.csv","a") as f:
        if mag_cut is not None:
            f.write(parse_info(fname+'_mag'+str(mag_cut), folder_name, cal_result) + "\n")
        else:
            f.write(parse_info(fname, folder_name, cal_result) + "\n")


if __name__ == "__main__":
    start_time = time.time()

    p = argparse.ArgumentParser()
    p.add_argument("--noise", type=float, required=True, help="Noise level for calibration")
    p.add_argument("--fname", type=str, default="test", help="Filename prefix")
    p.add_argument("--mag_cut", type=float, default=None, help="Magnitude cut for selection")
    args = p.parse_args()

    noise_level = args.noise
    fname = args.fname
    mag_cut = args.mag_cut
    print(f"Starting shape measurement and calibration for noise level: {noise_level}")
    print(f"Using filename prefix: {fname}")
    if mag_cut is not None:
        print(f"Applying magnitude cut at: {mag_cut}")

    get_calibration_biases(
        grid_per_img=100,
        imgBs=100,
        img_nB_range= [0,4000],
        folder_name=format_number(noise_level),
        is_twin=True,
        fname=fname,
        workers=64,
        mag_cut=mag_cut,
        flux_name='fpfs_mag',
    )

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")