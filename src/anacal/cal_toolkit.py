import anacal
import numpy as np
import matplotlib.pylab as plt

from numpy.lib import recfunctions as rfn
from astropy.visualization import simple_norm
import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial

def load_img(shear_comp,
            shear_mode,
            index,
            abs_shear_value = 0.02,
            noise_level = 0.1,
            ori_seed = 777,
            directory='./temp/',
            ):
    dir = f'{directory}/{shear_comp}_{shear_mode}_val_{abs_shear_value:.3f}/img_{int(index)}/'
    cutouts = np.load(dir + 'cutouts.npy')
    psfs = np.load(dir + 'psf_image.npy')
    cat = pd.read_csv(dir + 'gt_cat.csv')
    psfs = np.tile(psfs, (len(cutouts), 1, 1))
    seed = np.random.default_rng(ori_seed+index).integers(0,100000)
    for (idx, cutout) in enumerate(cutouts):
        rng = np.random.default_rng(seed+idx)
        noise = rng.standard_normal(cutout.shape)*noise_level
        cutouts[idx] += noise

    return cutouts, psfs, cat






def load_one(i, *, shear_comp, shear_mode, abs_shear_value, noise_level, directory):
    return load_img(
        shear_comp=shear_comp,
        shear_mode=shear_mode,
        abs_shear_value=abs_shear_value,
        index=i,
        noise_level=noise_level,
        directory=directory,
    )

def build_dataset(
    shear_comp,
    shear_mode,
    abs_shear_value=0.02,
    noise_level=0.1,
    index_range=[0,100],
    directory=dir,
    workers=8,
    use_threads=True,
):
    # 1) Probe shapes once
    cutouts0, psfs0, cat0 = load_img(
        shear_comp=shear_comp,
        shear_mode=shear_mode,
        abs_shear_value=abs_shear_value,
        index=index_range[0],
        noise_level=noise_level,
        directory=directory,
    )
    B, H, W = cutouts0.shape
    N_BATCH = index_range[1] - index_range[0]
    N = N_BATCH * B

    # 2) Preallocate
    all_cutouts = np.empty((N, H, W), dtype=cutouts0.dtype)
    all_psfs    = np.empty((N, H, W), dtype=psfs0.dtype)
    cat_chunks  = [None] * N_BATCH

    # 3) Place first batch
    all_cutouts[0:B] = cutouts0
    all_psfs[0:B]    = psfs0
    cat_chunks[0]    = cat0

    # 4) Parallel read remaining batches with a progress bar
    Exec = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    max_w = min(workers, max(1, N_BATCH - 1))

    loader = partial(
        load_one,
        shear_comp=shear_comp,
        shear_mode=shear_mode,
        abs_shear_value=abs_shear_value,
        noise_level=noise_level,
        directory=directory,
    )

    map_kwargs = {}
    if not use_threads:
        map_kwargs["chunksize"] = 4  # tune 2–16 for processes

    with Exec(max_workers=max_w) as ex:
        it = ex.map(loader, range(index_range[0]+1, index_range[1]), **map_kwargs)
        for i, (cutouts, psfs, cat) in enumerate(
            tqdm(it, total=N_BATCH - 1, desc="Reading batches", unit="batch"),
            start=1,
        ):
            s = i * B
            e = s + B
            all_cutouts[s:e] = cutouts
            all_psfs[s:e]    = psfs
            cat_chunks[i]    = cat

    all_cats = pd.concat(cat_chunks, ignore_index=True)
    return all_cutouts, all_psfs, all_cats