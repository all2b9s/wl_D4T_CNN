import os
import math
# Must be set BEFORE importing numpy / scipy / anacal etc.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fitsio
from src.anacal.batch_calibration import prepare_q_images
from src.anacal.cal_toolkit import build_dataset
import anacal
import multiprocessing as mp

sigma_shapelets=0.52
sigma_shapelets1=0.45

fpfs_config = anacal.fpfs.FpfsConfig(
    sigma_shapelets=sigma_shapelets,  # The first measurement scale (also for detection)
    sigma_shapelets1=sigma_shapelets1,  # The second measurement scale
)


def format_number(x: float) -> str:
    s = f"{x:.3f}".rstrip('0').rstrip('.')  # keep up to 3 decimal places, trim trailing zeros
    s = s.replace('.', '')                  # remove the decimal point
    return f"n{s}"


# ---- Globals for workers ----
_FPFS_GLOBALS = {
    "cutouts": None,
    "psfs": None,
    "noises": None,
    "fpfs_config": None,
    "mag_zero": None,
    "pixel_scale": None,
    "noise_variance": None,
    "detection": None,
}

def _init_fpfs_worker(cutouts, psfs, noises,
                      fpfs_config, mag_zero, pixel_scale,
                      noise_variance, detection):
    """Initializer: store large arrays & constants in worker-local globals."""
    _FPFS_GLOBALS["cutouts"]        = cutouts
    _FPFS_GLOBALS["psfs"]           = psfs
    _FPFS_GLOBALS["noises"]         = noises
    _FPFS_GLOBALS["detection"]      = detection
    _FPFS_GLOBALS["fpfs_config"]    = fpfs_config
    _FPFS_GLOBALS["mag_zero"]       = mag_zero
    _FPFS_GLOBALS["pixel_scale"]    = pixel_scale
    _FPFS_GLOBALS["noise_variance"] = noise_variance


def _fpfs_worker(idx):
    """Process a single galaxy index for measurements."""
    g = _FPFS_GLOBALS

    try:
        cutout = g["cutouts"][idx].astype(np.float64, copy=False)
        psf    = g["psfs"][idx].astype(np.float64, copy=False)
        noise  = g["noises"][idx].astype(np.float64, copy=False)

        out = anacal.fpfs.process_image(
            fpfs_config=g["fpfs_config"],
            mag_zero=g["mag_zero"],
            gal_array=cutout,
            psf_array=psf,
            pixel_scale=g["pixel_scale"],
            noise_variance=g["noise_variance"],
            noise_array=noise,
            detection=g["detection"],
            do_compute_detect_weight=False,
        )

        # ---- Key: guard against empty results ----
        if out is None or (hasattr(out, "shape") and out.shape[0] == 0) or (len(out) == 0):
            return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, idx)

        e1 = out["fpfs1_e1"][0]
        e2 = out["fpfs1_e2"][0]
        R11 = out["fpfs1_de1_dg1"][0]
        R22 = out["fpfs1_de2_dg2"][0]
        m00 = out["fpfs1_m00"][0]
        dm00_dg1 = out["fpfs1_dm00_dg1"][0]
        dm00_dg2 = out["fpfs1_dm00_dg2"][0]

        return (e1, e2, R11, R22, m00, dm00_dg1, dm00_dg2, idx)

    except Exception as e:
        # Keep logging minimal to avoid flooding output with 64 processes;
        # you can also write errors to a log file.
        msg = str(e)
        if ("max() iterable argument is empty" in msg) or ("merge_arrays" in msg):
            return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, idx)

        print(f"[FPFS failed] idx={idx} err={e}")
        # print(traceback.format_exc())  # enable for detailed debugging
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, idx)




def fpfs_measure(
    dir,
    imgBs = 100,            # batch size for npy file
    img_nB_range = [0,100],           # number of npy batch
    noise_level = 0,
    n_workers = 32,
    mag_zero = 30.0,
    center = [32,32],
    pixel_scale = 0.2,
    shear_task = (2, "g1"),
    fname = 'fpfs',
    if_return = False,
):
    print(f"Measuring shape and magnitude with FPFS for {str(dir)}, under noise_level={noise_level}")
    nBs = img_nB_range[1] - img_nB_range[0]
    shear_mode, shear_comp = shear_task
    gal_per_file = 100      # how your build_dataset is arranged per index_range step

    # Initialize detection array once
    dtype = np.dtype(
    [
        ("y", np.int32),
        ("x", np.int32),
    ]
    )
    detection = np.empty(1, dtype=dtype)
    detection["y"] = center[0]
    detection["x"] = center[1]

    if noise_level > 0:
        noise_variance = noise_level**2.0
    else:
        noise_variance = 1e-4
    

    if noise_level == 'adaptive':
        folder_name = 'nada'
    else:
        folder_name = format_number(noise_level)
    
    if if_return:
        shapes_all = []
        R_ana_all = []
        flux_all = []
        Rflux_all = []


    # Process all batches
    for start in range(img_nB_range[0], img_nB_range[1]):
        print(f'Processing batch {start}')
        range_idx = [start*imgBs, (start+1)*imgBs]

        all_cutouts, all_psfs, all_cats, noise_levels = build_dataset(
            shear_comp=shear_comp, shear_mode=shear_mode, abs_shear_value=0.02,
            noise_level=noise_level, index_range=range_idx,
            directory=dir, workers=n_workers, use_threads=False
        )

        # noise per-galaxy
        np.random.seed(20020620 + start)
        noises = np.random.normal(0, 1, all_cutouts.shape) * noise_levels.reshape(-1, 1, 1)

        n_gal = all_cutouts.shape[0]   # should be 10000 here

        # Parallel processing with multiprocessing pool
        with mp.Pool(
            processes=n_workers,
            initializer=_init_fpfs_worker,
            initargs=(
                all_cutouts[:,:,:],
                all_psfs,
                noises,
                fpfs_config,
                mag_zero,
                pixel_scale,
                noise_variance,
                detection,
            ),
        ) as pool:
            # map over indices 0..n_gal-1
            indices = range(n_gal)
            # adjust chunksize for performance if need
            results = pool.map(_fpfs_worker, indices, chunksize=50)
        
        # results is a list of (e1, e2, R11, R22, m00, dm00_dg1, dm00_dg2, idx) tuples
        results_arr = np.array(results)           # (e1, e2, R11, R22, m00, dm00_dg1, dm00_dg2, idx) -- (n_gal, 8)
        shapes = results_arr[:, 0:2]
        R_ana = results_arr[:, 2:4]
        flux = results_arr[:, 4:5] *4*np.pi*(sigma_shapelets1**2)/2
        Rflux = results_arr[:, 5:7] *4*np.pi*(sigma_shapelets1**2)/2

        if if_return:
            shapes_all.append(shapes)
            R_ana_all.append(R_ana)
            flux_all.append(flux)
            Rflux_all.append(Rflux)

        else:
            save_name = f'/work/hdd/bfmo/wenyinli/measurement/xlens_sims/{folder_name}/{shear_comp}_{shear_mode}/'
            if not os.path.exists(save_name):
                os.makedirs(save_name)
            np.save(save_name + f'{fname}_e_fpfs_{start}.npy', shapes)
            np.save(save_name + f'{fname}_R_fpfs_{start}.npy', R_ana)
            np.save(save_name + f'{fname}_m00_{start}.npy', flux)
            np.save(save_name + f'{fname}_Rm00_{start}.npy', Rflux)

        del all_cutouts, all_psfs, all_cats, noises, results, results_arr

    if if_return:
        shapes_all = np.concatenate(shapes_all, axis=0)
        R_ana_all = np.concatenate(R_ana_all, axis=0)
        flux_all = np.concatenate(flux_all, axis=0)
        Rflux_all = np.concatenate(Rflux_all, axis=0)
        return shapes_all, R_ana_all, flux_all, Rflux_all



