import anacal
import numpy as np
import matplotlib.pylab as plt
import numbers
from numpy.lib import recfunctions as rfn
from astropy.visualization import simple_norm
import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import time
from numpy.random import SeedSequence, default_rng


def robust_load_npy(path, retries=5, delay=0.05):
    for i in range(retries):
        try:
            with open(path, "rb") as f:
                return np.load(f, allow_pickle=False)
        except Exception as e:
            msg = str(e)
            # only retry on short-read-like errors
            if "Failed to read all data for array" in msg or "file seems not fully written" in msg:
                if i < retries - 1:
                    time.sleep(delay * (2**i))
                    continue
            raise

def load_img(shear_comp,
            shear_mode,
            index,
            abs_shear_value = 0.02,
            noise_level = 0.594,
            noise_mode = 'fixed',
            ori_seed = 1115,
            directory='./temp/',
            ):
    dir = f'{directory}/{shear_comp}_{shear_mode}_val_{abs_shear_value:.3f}/img_{int(index)}/'
    try:
        cutouts = robust_load_npy(dir + 'cutouts.npy')
        psfs = robust_load_npy(dir + 'psf_image.npy')
        cat = pd.read_csv(dir + 'gt_cat.csv')
    except Exception as e:
        raise RuntimeError(f"Failed to load data from {dir}: {e}")
    psfs = np.tile(psfs, (len(cutouts), 1, 1))
    if noise_level <= 0:
        return cutouts, psfs, cat, np.zeros(len(cutouts))
    if (noise_mode == 'adaptive') or (noise_mode == 'uniform'):
        noise_list = np.zeros(len(cutouts))
    if noise_mode == 'fixed':
        noise_list = np.full(len(cutouts), noise_level)
    base_seq = SeedSequence([ori_seed, index])
    rng = np.random.default_rng(base_seq)
    for (idx, cutout) in enumerate(cutouts):
        if noise_mode == 'adaptive':
            center = (cutout.shape[-2]//2, cutout.shape[-1]//2)
            cutout_crop = cutout[center[0]-2:(center[0]+3), center[1]-2:(center[1]+3)]
            noise_list[idx] = min((cutout_crop.mean())*(rng.uniform(0,0.3))**2,noise_level)
        if noise_mode == 'uniform':
            center = (cutout.shape[-2]//2, cutout.shape[-1]//2)
            cutout_crop = cutout[center[0]-2:(center[0]+3), center[1]-2:(center[1]+3)]
            noise_list[idx] = min((cutout_crop.mean())/(rng.uniform(5,40)),noise_level)
            
        noise = rng.standard_normal(cutout.shape)*noise_list[idx]
        cutouts[idx] += noise
    return cutouts, psfs, cat, noise_list


def load_one(i, *, shear_comp, shear_mode, abs_shear_value, noise_level,noise_mode, directory):
    return load_img(
        shear_comp=shear_comp,
        shear_mode=shear_mode,
        abs_shear_value=abs_shear_value,
        index=i,
        noise_level=noise_level,
        noise_mode=noise_mode,
        directory=directory,
    )

def build_dataset(
    shear_comp,
    shear_mode,
    abs_shear_value=0.02,
    noise_level=0.594,
    noise_mode='fixed',
    index_range=[0,100],
    directory='dir',
    workers=8,
    return_imgs = True,
    use_threads=True,
):
    # 1) Probe shapes once — capture noise levels too
    cutouts0, psfs0, cat0, noise0 = load_img(
        shear_comp=shear_comp,
        shear_mode=shear_mode,
        abs_shear_value=abs_shear_value,
        index=index_range[0],
        noise_level=noise_level,
        noise_mode=noise_mode,
        directory=directory,
    )
    B, H, W = cutouts0.shape
    N_BATCH = index_range[1] - index_range[0]
    N = N_BATCH * B

    # 2) Preallocate
    if return_imgs:
        all_cutouts = np.empty((N, H, W), dtype=cutouts0.dtype)
        all_psfs    = np.empty((N, H, W), dtype=psfs0.dtype)
    all_noise = np.empty((N,), dtype=np.float32)
    cat_chunks  = [None] * N_BATCH

    # 3) Fill first batch (already loaded above) — avoids reloading it
    if return_imgs:
        all_cutouts[0:B] = cutouts0
        all_psfs[0:B]    = psfs0
    all_noise[0:B] = noise0
    cat_chunks[0] = cat0

    # 4) Parallel read REMAINING batches (skip index_range[0])
    Exec = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    remaining = N_BATCH - 1
    if remaining > 0:
        max_w = min(workers, remaining)

        loader = partial(load_one, 
                         shear_comp=shear_comp, 
                         shear_mode=shear_mode, 
                         abs_shear_value=abs_shear_value, 
                         noise_level=noise_level,
                         noise_mode=noise_mode, 
                         directory=directory)

        map_kwargs = {}
        if not use_threads:
            map_kwargs["chunksize"] = 4  # tune 2–16 for processes

        with Exec(max_workers=max_w) as ex:
            it = ex.map(loader, range(index_range[0] + 1, index_range[1]), **map_kwargs)
            for i_rem, (cutouts, psfs, cat, noise_level) in enumerate(
                tqdm(it, total=remaining, desc="Reading batches", unit="batch"),
                start=0,
            ):
                i = i_rem + 1  # skip first slot
                s = i * B
                e = s + B
                if return_imgs:
                    all_cutouts[s:e] = cutouts
                    all_psfs[s:e]    = psfs
                all_noise[s:e] = noise_level
                cat_chunks[i]    = cat

    all_cats = pd.concat(cat_chunks, ignore_index=True)
    if return_imgs:
        return all_cutouts, all_psfs, all_cats, all_noise
    else:
        return [], [], all_cats, all_noise
    

def selection_response(
    flux,
    R_flux,
    shapes,
    zero_point = 30,
    mag_cut = 25.0,
):
    dg = 0.02
    def flux_to_mag(flux):
        return zero_point-2.5 * np.log10(flux)
    mask_1p = flux_to_mag(flux+dg*R_flux[:,0])< mag_cut
    mask_1m = flux_to_mag(flux-dg*R_flux[:,0])< mag_cut
    mask_2p = flux_to_mag(flux+dg*R_flux[:,1])< mag_cut
    mask_2m = flux_to_mag(flux-dg*R_flux[:,1])< mag_cut

    e_1p = shapes[mask_1p,0].mean()
    e_1m = shapes[mask_1m,0].mean()
    e_2p = shapes[mask_2p,1].mean()
    e_2m = shapes[mask_2m,1].mean()
    R_s_1 = (e_1p - e_1m) / (2*dg)
    R_s_2 = (e_2p - e_2m) / (2*dg)
    R_s = np.array([R_s_1, R_s_2])
    return R_s

def format_number(x: float) -> str:
    s = f"{x:.3f}".rstrip('0').rstrip('.')  # keep up to 3 decimal places, trim trailing zeros
    s = s.replace('.', '')                  # remove the decimal point
    return f"n{s}"