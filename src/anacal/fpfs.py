import os
# Must be set BEFORE importing numpy / scipy / anacal etc.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
from src.anacal.cal_toolkit import build_dataset, format_number
import anacal
import multiprocessing as mp

sigma_shapelets1 = 0.45
sigma_shapelets2 = 0.55

# Detection is no longer done by FPFS itself (new anacal API): FpfsConfig
# only takes the two measurement kernels, sigma_shapelets1 (required) and
# sigma_shapelets2 (optional second scale). See src/anacal/detection_ml_pipeline.py.
fpfs_config = anacal.fpfs.FpfsConfig(
    sigma_shapelets1=sigma_shapelets1,  # primary measurement scale
    sigma_shapelets2=sigma_shapelets2,  # second measurement scale
)

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

        # Detection is now required (new anacal API); fpfs_measure always
        # passes a fixed-center catalogue, no internal FPFS detection weight.
        out = anacal.fpfs.process_image(
            fpfs_config=g["fpfs_config"],
            mag_zero=g["mag_zero"],
            gal_array=cutout,
            psf_array=psf,
            pixel_scale=g["pixel_scale"],
            noise_variance=g["noise_variance"],
            noise_array=noise,
            detection=g["detection"],
        )

        # ---- Key: guard against empty results ----
        if out is None or (hasattr(out, "shape") and out.shape[0] == 0) or (len(out) == 0):
            #print(f"[FPFS empty result] idx={idx} out={out}")
            return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, idx)

        e1 = out["fpfs1_e1"][0]
        e2 = out["fpfs1_e2"][0]
        R11 = out["fpfs1_de1_dg1"][0]
        R22 = out["fpfs1_de2_dg2"][0]
        # Flux columns use the photoz token order (flux_{kernel}), same
        # convention as flux_gauss0 in detection_ml_pipeline.py.
        flux = out["flux_fpfs1"][0]
        dflux_dg1 = out["dflux_fpfs1_dg1"][0]
        dflux_dg2 = out["dflux_fpfs1_dg2"][0]

        return (e1, e2, R11, R22, flux, dflux_dg1, dflux_dg2, idx)

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
    save_dir = None,        # output directory (override the hard-coded default below)
):
    """
    Batch FPFS shape & flux measurement (the baseline estimator compared in the paper).

    Args:
        dir          : directory holding the simulated cutouts (see build_dataset).
        imgBs        : number of galaxies per .npy batch.
        img_nB_range : [start, stop) batch indices to process.
        noise_level  : Gaussian noise sigma in ADU ('adaptive' -> per-galaxy noise).
        n_workers    : multiprocessing pool size.
        mag_zero     : magnitude zero point.
        center       : (y, x) detection position within each cutout.
        pixel_scale  : arcsec per pixel.
        shear_task   : (shear_mode, shear_comp) tuple, e.g. (2, 'g1').
        fname        : output filename prefix.
        save_dir     : output directory. Defaults to a hard-coded cluster path
                       (see below) — override it for your own environment.

    Saves per batch: {fname}_e_fpfs.npy (shapes), {fname}_R_fpfs.npy (responses),
    {fname}_m00.npy (flux) and {fname}_Rm00.npy (flux responses).
    """
    print(f"Measuring shape and magnitude with FPFS for {str(dir)}, under noise_level={noise_level}")
    nBs = img_nB_range[1] - img_nB_range[0]
    shear_mode, shear_comp = shear_task
    gal_per_file = 100      # how your build_dataset is arranged per index_range step

    # Detection is required by the new anacal API (no internal FPFS
    # detection): pass a single fixed-center catalogue.
    dtype = np.dtype([("y", np.int32), ("x", np.int32)])
    detection = np.empty(1, dtype=dtype)
    detection["y"] = center[0]
    detection["x"] = center[1]

    # Determine noise variance for FPFS config
    if noise_level > 0:
        noise_variance = noise_level**2.0
    else:
        noise_variance = 1e-4
    

    if noise_level == 'adaptive':
        folder_name = 'nada'
    else:
        folder_name = format_number(noise_level)

    if save_dir is None:
        save_dir = '/work/hdd/bfmo/wenyinli/measurement/xlens_sims/'

    ####################
    if if_return:
        shapes_all = []
        R_ana_all = []
        flux_all = []
        Rflux_all = []

    ######################################################
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

        # results: (e1, e2, R11, R22, flux, dflux_dg1, dflux_dg2, idx)
        results_arr = np.array(results)           # (n_gal, 8)
        shapes = results_arr[:, 0:2]
        R_ana = results_arr[:, 2:4]
        # flux / dflux_dg{1,2} come straight from anacal (flux_fpfs1), so no
        # m00 -> flux conversion is needed.
        flux = results_arr[:, 4:5]
        Rflux = results_arr[:, 5:7]

        if if_return:
            shapes_all.append(shapes)
            R_ana_all.append(R_ana)
            flux_all.append(flux)
            Rflux_all.append(Rflux)

        else:
            save_name = os.path.join(save_dir, folder_name, f'{shear_comp}_{shear_mode}')
            if not os.path.exists(save_name):
                os.makedirs(save_name)
            np.save(os.path.join(save_name, f'{fname}_e_fpfs_{start}.npy'), shapes)
            np.save(os.path.join(save_name, f'{fname}_R_fpfs_{start}.npy'), R_ana)
            np.save(os.path.join(save_name, f'{fname}_m00_{start}.npy'), flux)
            np.save(os.path.join(save_name, f'{fname}_Rm00_{start}.npy'), Rflux)

        del all_cutouts, all_psfs, all_cats, noises, results, results_arr

    if if_return:
        shapes_all = np.concatenate(shapes_all, axis=0)
        R_ana_all = np.concatenate(R_ana_all, axis=0)
        flux_all = np.concatenate(flux_all, axis=0)
        Rflux_all = np.concatenate(Rflux_all, axis=0)
        return shapes_all, R_ana_all, flux_all, Rflux_all


