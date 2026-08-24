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
OUTPUT_ROOT = "/work/hdd/bfcn/wenyinli/noiseless_sim"

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


def make_sim_task():
    cfg = MultibandSimConfig()
    cfg.survey_name = "lsst"
    cfg.draw_image_noise = False
    return MultibandSimTask(config=cfg)


def run_seed(args_seed):
    args, seed = args_seed

    print(f"[PID {os.getpid()}] START {seed}", flush=True)

    skymap = make_skymap()
    cat_task = make_catalog_task(args)
    sim_task = make_sim_task()

    outdir = os.path.join(
        OUTPUT_ROOT,
        f"sim_mode{args.mode}",
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
        f"[PID {os.getpid()}] after catalog seed={seed}, N={len(truth)}",
        flush=True,
    )
    fitsio.write(
        os.path.join(outdir, f"truth-{seed:05d}.fits"),
        truth,
        clobber=True,
    )
    print(
    f"[PID {os.getpid()}] after truth write seed={seed}",
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
        f"[PID {os.getpid()}] after sim seed={seed} band={band}",
        flush=True,
        )

        exp.writeFits(
            os.path.join(
                outdir,
                f"exp-{band}-{seed:05d}.fits",
            )
        )

        print(
            f"[PID {os.getpid()}] after write seed={seed} band={band}",
            flush=True,
        )

    del truth
    gc.collect()

    print(f"[PID {os.getpid()}] DONE {seed}", flush=True)
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

    args = parser.parse_args()
    args.bands = [x.strip() for x in args.band.split(",") if x.strip()]

    seeds = list(range(args.start, args.end))

    if args.workers == 1:
        for seed in tqdm(seeds):
            run_seed((args, seed))
    else:
        with mp.Pool(args.workers) as pool:
            for _ in tqdm(
                pool.imap_unordered(
                    run_seed,
                    [(args, s) for s in seeds],
                    chunksize=1,
                ),
                total=len(seeds),
            ):
                pass


if __name__ == "__main__":
    main()
