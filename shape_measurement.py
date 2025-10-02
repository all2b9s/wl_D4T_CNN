import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import galsim
import torch
from tqdm import tqdm
from src.anacal.cal_toolkit import load_img, build_dataset
from src.anacal.batch_calibration import prepare_q_images, Calibrator
from src.architecture.single_gal_CNN import R180Inv_CNN_GeLU, D4T_CNN_GeLU
import os


device = ("cuda" if torch.cuda.is_available() else "cpu")
model = D4T_CNN_GeLU().to(device)
model.load_state_dict(torch.load("./models/D4T_CNN_nl_10ep.pth", map_location="cpu"))
_ = model.eval()
cal_CNN = Calibrator(model, device)
dir = '/work/hdd/bdsp/wenyinli/datasets/xlens_gal/'

shear_tasks = [
            #(0, "g1"),
            #(1, "g1"),
            #(0, "g2"),
            (1, "g2"),
        ]

for shear_mode, shear_comp in shear_tasks:
    for start in range(0, 100):
        print(f'Processing {shear_comp}_{shear_mode}, batch {start}')
        range_idx = [start*100, (start+1)*100]
        all_cutouts, all_psfs, all_cats = build_dataset(
                shear_comp=shear_comp, shear_mode=shear_mode, abs_shear_value=0.02,
                noise_level=0, index_range=range_idx, directory=dir, workers=32, use_threads=False
            )


        noises = np.random.normal(0, 0, all_cutouts.shape)
        q_imgs = prepare_q_images(
                all_cutouts, all_psfs, noises, all_cats,
                sigma_arcsec=0.7/2.355,
                workers=32, flim=10
            )
        
        shapes, R_ana = cal_CNN.shape_measure(q_imgs,batch_size=50)
        save_name = f'/projects/bdsp/wenyinli/datasets/xlens_sims/no_noise/{shear_comp}_{shear_mode}/'
        if not os.path.exists(save_name):
            os.makedirs(save_name)
        np.save(save_name + f'p70_shapes_{start}.npy', shapes)
        np.save(save_name + f'p70_Rana_{start}.npy', R_ana)
        del all_cutouts, all_psfs, all_cats, noises, q_imgs, shapes, R_ana

