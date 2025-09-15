import anacal
import numpy as np
import matplotlib.pylab as plt

import lsst.geom as geom
from lsst.afw.geom import makeSkyWcs

from xlens.simulator.multiband import (
    MultibandSimShearTaskConfig,
    MultibandSimShearTask,
)

from numpy.lib import recfunctions as rfn
from astropy.visualization import simple_norm
import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def cutout_xlens_image(full_image, truth_catalog, seed):
    cutouts = np.zeros((len(truth_catalog),64,64))
    for idx, gal in truth_catalog.iterrows():
        x, y = int(gal['image_x']), int(gal['image_y'])
        cutout = full_image[y-32:y+32, x-32:x+32]
        rng = np.random.default_rng(seed//2+idx)
        noise = rng.standard_normal(cutout.shape)*0.354
        cutout+=noise
        cutouts[idx] = cutout
    return cutouts
    


def xlens_gal_sim(
        output_dir = "./temp/",
        shear_mode = 2,
        shear_comp = 'g1',
        shear_value = 0.02,
        kappa_value = 0.0,
        seed = 20020103,
        band = 'i',
        dim = 500,
        sep = 11.0, # arcsec
        pixel_scale = 0.2,
    ):

    bbox = geom.Box2I(
        minimum=geom.Point2I(x=0, y=0),
        maximum=geom.Point2I(x=dim - 1, y=dim - 1),
    )
    cd_matrix = np.array([[-1.0, -0.0], [0.0, 1.0]]) * (pixel_scale / 3600.0)

    crval = geom.SpherePoint(
        np.pi / 2.0,  # radians
        0.0,  # radians
        geom.radians,
    )

    stack_crpix = geom.Point2D(
        (dim - 1) / 2.0,
        (dim - 1) / 2.0,
    )

    wcs_stack = makeSkyWcs(
        crpix=stack_crpix,
        crval=crval,
        cdMatrix=cd_matrix,
    )

    config = MultibandSimShearTaskConfig()
    config.survey_name = (
        "lsst"  # The scale parameter needs to be consistent with scale
    )
    config.layout="grid"
    config.sep= sep  # arcsec
    config.mode = shear_mode # 0: - ; 1: + ; 2: no shear
    config.test_target = shear_comp  # 'g1' or 'g2'
    config.test_value = shear_value  # e.g., 0.02
    config.kappa_value = kappa_value  # e.g., 0 
    config.kappa_value = kappa_value

    sim_task = MultibandSimShearTask(config=config)
    outcome = sim_task.run(band=band, seed=seed, boundaryBox=bbox, wcs=wcs_stack)

    # Galaxy image
    gal_array = np.asarray(
        outcome.outputExposure.getMaskedImage().getImage().array,
        np.float64,
    )

    # PSF image
    lsst_psf = outcome.outputExposure.getPsf()
    xc = int(dim // 2)  # we need PSF model without subpixel offset
    yc = int(dim // 2)
    psf_array = np.asarray(
        anacal.utils.resize_array(
            lsst_psf.computeImage(geom.Point2D(xc, yc)).getArray(), (64, 64)
        ),
        dtype=np.float64,
    )

    # Truth Catalog
    gal_cat = pd.DataFrame(outcome.outputTruthCatalog)

    # Cutout Images
    cutouts = cutout_xlens_image(gal_array, gal_cat, seed)

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "full_image.npy"), gal_array)
    np.save(os.path.join(output_dir, "psf_image.npy"), psf_array)
    np.save(os.path.join(output_dir, "cutouts.npy"), cutouts)
    gal_cat.to_csv(os.path.join(output_dir, "gt_cat.csv"), index=False)

class xlen_simulator():
    def __init__(self, output_dir, 
                 abs_shear = 0.02, 
                 ori_seed = 666,
                 image_size = 800,
                 pixel_scale = 0.2,
                 separation = 20.0,
                 num_workers=8):
        self.seed = ori_seed
        self.output_dir = output_dir
        self.abs_shear = abs_shear
        self.image_size = image_size
        self.pixel_scale = pixel_scale
        self.separation = separation
        self.num_workers = num_workers

    def __call__(self, num_sims: int):
        shear_tasks = [
            (2, "g1"),
            (2, "g2"),
            (0, "g1"),
            (1, "g1"),
            (0, "g2"),
            (1, "g2"),
        ]
        seeds = np.random.default_rng(self.seed).integers(0, 1e8, size=num_sims)
        results = []
        for shear_mode, shear_comp in shear_tasks:
            args = [(i, seeds[i], shear_mode, shear_comp) for i in range(num_sims)]
            with Pool(processes=self.num_workers) as pool:
                # Parallelize over sims for this single shear task
                res = pool.starmap(self._worker, args, chunksize=8)
                results.extend(res)

        return results

    def _worker(self, index, seed, shear_mode, shear_comp):
        output_dir = f'{self.output_dir}/{shear_comp}_{shear_mode}_val_{self.abs_shear:.3f}/img_{int(index)}/'
        xlens_gal_sim(
            output_dir = output_dir,
            shear_mode = shear_mode,
            shear_comp = shear_comp,
            shear_value = self.abs_shear,
            seed = seed,
            dim = self.image_size,
            pixel_scale = self.pixel_scale,
            sep = self.separation,
        )


simulator = xlen_simulator('/work/hdd/bdsp/wenyinli/datasets/xlens_gal/', num_workers=64)
simulator(10000)
