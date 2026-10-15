#!/usr/bin/env python3
"""
Conservative xlens simulation wrapper.

This version intentionally follows the original xlens example style:
- no cached CatalogShearTask / MultibandSimTask
- no multiprocessing initializer
- no maxtasksperchild
- no atomic FITS writing
"""

import os
import gc
import argparse
import multiprocessing as mp

import fitsio
from tqdm import tqdm


from lsst.skymap.discreteSkyMap import DiscreteSkyMap, DiscreteSkyMapConfig
from xlens.simulator.catalog import CatalogShearTask, CatalogShearTaskConfig
from xlens.simulator.sim import MultibandSimConfig, MultibandSimTask


#OUTPUT_ROOT = "/taiga/illinois/las/astro/xinliuxl/DC1_sim_blended/noiseless_sim"
OUTPUT_ROOT = "/work/hdd/bfcn/wenyinli/blended_shapenoise_fullIMG"

def make_skymap():
    cfg = DiscreteSkyMapConfig()
    cfg.raList = [0.0]
    cfg.decList = [0.0]
    cfg.radiusList = [0.1]
    cfg.rotation = 0.0
    cfg.projection = "TAN"
    cfg.patchInnerDimensions = [1500, 1500]
    cfg.patchBorder = 0
    cfg.pixelScale = 0.2
    cfg.tractOverlap = 0.0
    return DiscreteSkyMap(cfg)


def make_catalog_task(args):
    cfg = CatalogShearTaskConfig()
    cfg.z_bounds = [0.0, 0.63, 0.98, 1.48, 10.0]
    cfg.mode = args.mode
    cfg.rotId = args.rot
    cfg.kappa_value = args.kappa
    cfg.test_value = args.shear
    cfg.test_target = args.target
    cfg.layout = args.layout
    cfg.extend_ratio = 1.08 if args.layout == "random" else 0.92
    cfg.sep_arcsec = 14
    return CatalogShearTask(config=cfg)


def make_sim_task(args):
    cfg = MultibandSimConfig()
    cfg.survey_name = "lsst"
    # xlens (upstream) has no noise_variance config field, so the noise
    # level cannot be set here: whenever draw_image_noise is True, xlens
    # draws noise at its fixed per-band default variance
    # (noise_variance_defaults[band][survey], e.g. ~0.354 for lsst/i).
    # --noise-level is therefore a pure on/off switch: 0 -> noiseless,
    # >0 -> draw noise at xlens's default variance.
    cfg.draw_image_noise = args.noise_level > 0
    return MultibandSimTask(config=cfg)


def run_seed(args_seed):
    args, seed, index = args_seed

    print(f"[PID {os.getpid()}] START index={index} seed={seed}", flush=True)

    skymap = make_skymap()
    cat_task = make_catalog_task(args)
    sim_task = make_sim_task(args)

    noise_tag = ""
    if args.noise_level > 0:
        s = f"{args.noise_level:.3f}".rstrip("0").rstrip(".")
        noise_tag = "_n" + s.replace(".", "")

    outdir = os.path.join(
        OUTPUT_ROOT,
        f"sim_mode{args.mode}{noise_tag}",
    )
    os.makedirs(outdir, exist_ok=True)

    print(
    f"[PID {os.getpid()}] before catalog seed={seed}",
    flush=True,
    )

    
    truth = cat_task.run(
        tract_info=skymap[0],
        seed=seed,
    ).truthCatalog


    
    print(
        f"[PID {os.getpid()}] after catalog index={index} seed={seed}, N={len(truth)}",
        flush=True,
    )
    # Output files are keyed by the running index (same convention the
    # measurement side uses); the RNG seed is ori_seed + index.
    fitsio.write(
        os.path.join(outdir, f"truth-{index:05d}.fits"),
        truth,
        clobber=True,
    )
    print(
        f"[PID {os.getpid()}] after truth write index={index} seed={seed}",
        flush=True,
    )

    for band in args.bands:
        print(
            f"[PID {os.getpid()}] before sim seed={seed} band={band}",
            flush=True,
            )

        exp = sim_task.run(
            tract_info=skymap[0],
            patch_id=0,
            band=band,
            seed=seed,
            truthCatalog=truth,
        ).simExposure

        print(
        f"[PID {os.getpid()}] after sim index={index} seed={seed} band={band}",
        flush=True,
        )

        exp.writeFits(
            os.path.join(
                outdir,
                f"exp-{band}-{index:05d}.fits",
            )
        )

        print(
        f"[PID {os.getpid()}] after write index={index} seed={seed} band={band}",
        )

    del truth
    gc.collect()

    print(f"[PID {os.getpid()}] DONE index={index} seed={seed}", flush=True)
    return seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, default=0)
    parser.add_argument("--shear", type=float, default=0.02)
    parser.add_argument("--kappa", type=float, default=0.0)
    parser.add_argument("--target", default="g1")
    parser.add_argument("--layout", default="random")
    parser.add_argument("--rot", type=int, default=0)
    parser.add_argument("--band", default="i")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--ori_seed",
        type=int,
        default=0,
        help="base seed: exposure at running index i is simulated with seed = "
        "ori_seed + i. Output files stay keyed to i (e.g. truth-00000.fits), "
        "so the measurement side is unaffected. Default 0 reproduces the "
        "original seed == index behavior. Default: %(default)s",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.0,
        help="pure on/off switch for noise: 0 -> noiseless; >0 -> draw "
        "noise at xlens's fixed per-band default variance (the numeric "
        "value is not used). Default: %(default)s",
    )

    args = parser.parse_args()
    args.bands = [x.strip() for x in args.band.split(",") if x.strip()]

    # Running index i in [start, end) is the output filename key; the RNG
    # seed passed to xlens is ori_seed + i.
    jobs = [(args, args.ori_seed + i, i) for i in range(args.start, args.end)]

    if args.workers == 1:
        for job in tqdm(jobs):
            run_seed(job)
    else:
        with mp.Pool(args.workers) as pool:
            for _ in tqdm(
                pool.imap_unordered(
                    run_seed,
                    jobs,
                    chunksize=1,
                ),
                total=len(jobs),
            ):
                pass


if __name__ == "__main__":
    main()
