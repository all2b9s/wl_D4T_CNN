import anacal
import numpy as np
import matplotlib.pylab as plt
import lsst.geom as geom
from lsst.afw.geom import makeSkyWcs
import fitsio
import os
# LSST task to define DC2-like skymap 
from lsst.skymap.discreteSkyMap import (
    DiscreteSkyMapConfig, DiscreteSkyMap,
)
from lsst.pipe.tasks.coaddBase import makeSkyInfo

from xlens.simulator.catalog import (
    CatalogShearTask,
    CatalogShearTaskConfig,
)
from xlens.simulator.sim import (
    MultibandSimConfig, MultibandSimTask
)

# Detection Task: Detect, shape measurement
from xlens.process_pipe.anacal_detect import (
    AnacalDetectPipeConfig, 
    AnacalDetectPipe,
)

# Force color measurement Task: 
# flux measurement on the other bands
from xlens.process_pipe.anacal_force import (
    AnacalForcePipe,
    AnacalForcePipeConfig,
)

# Match Task: match to input catalog
from xlens.process_pipe.match import (
    matchPipe,
    matchPipeConfig,
)

from astropy.visualization import ZScaleInterval

from numpy.lib import recfunctions as rfn
from astropy.visualization import simple_norm
import sys
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from dataset_toolkit import get_weighted_e

cat_ref,h = fitsio.read(
    os.path.join(os.environ.get('CATSIM_DIR'), "OneDegSq.fits"),header=True
)
print(os.path.join(os.environ.get('CATSIM_DIR'), "OneDegSq.fits"))
# put this at module level, not inside __call__
def _worker_wrapper(args):
    obj, func = args[0], args[1]   # unpack object and function
    return func(*args[2:])         # call method with remaining args

def cutout_xlens_image(full_image, truth_catalog, seed, noise_level=0.354):
    cutouts = np.zeros((len(truth_catalog),64,64))
    for idx, gal in truth_catalog.iterrows():
        x, y = int(0.5+gal['image_x']), int(0.5+gal['image_y'])
        cutout = full_image[y-32:y+32, x-32:x+32]
        rng = np.random.default_rng(seed//2+idx)
        if noise_level>0:
            noise = rng.standard_normal(cutout.shape)*noise_level
            cutout+=noise
        cutouts[idx] = cutout
    return cutouts

def xlens_gal_sim(
        output_dir = "./temp/",
        shear_mode = 2,
        shear_comp = 'g1',
        shear_value = 0.02,
        kappa_value = 0.0,
        rotId = 0,
        psf_e = 0.0,
        seed = 20020103,
        band = 'i',
        dim = 800,
        sep = 11.0, # arcsec
        pixel_scale = 0.2,
        has_shift = True,
    ):
    tract_id = 0
    patch_id = 0
    skymap_config = DiscreteSkyMapConfig()
    skymap_config.projection = "TAN"

    # Define tract center explicitly 
    skymap_config.raList = [0]         # degrees
    skymap_config.decList = [0]        # degrees
    skymap_config.radiusList = [0.2*dim/5/3600]     # radius in degrees

    skymap_config.rotation = 0.0         # tract rotation in degrees

    # Patch and tract configuration
    skymap_config.patchInnerDimensions = [int(dim),int(dim)]  # inner size of patch in pixels
    skymap_config.patchBorder = 50                    # border size in pixels
    skymap_config.pixelScale = pixel_scale             # arcsec/pixel
    skymap_config.tractOverlap = 0.0                   # no overlap
    
    # Create the skymap
    skymap = DiscreteSkyMap(skymap_config)

    # configuration
    task_config = CatalogShearTaskConfig()
    task_config.kappa_value = kappa_value  # e.g., 0.0
    task_config.layout="grid"
    task_config.z_bounds = [-0.01, 20.0]
    task_config.test_target = shear_comp  # 'g1' or 'g2'
    task_config.test_value = shear_value  # e.g., 0.02
    task_config.mode = shear_mode # g1: (-0.02, 0.0), 
    #config.sep= sep  # arcsec
    # we can change to mode = 5 for shear g1: (0.02, 0)
    task_config.extend_ratio = 0.9
    if has_shift:
        task_config.force_pixel_center = False
        task_config.apply_lensing_position_shifts = True
    else:
        task_config.force_pixel_center = True
        task_config.apply_lensing_position_shifts = False
    task_config.select_observable = ['i_ab']
    task_config.select_lower_limit = [0]
    task_config.select_upper_limit = [25.3]
    task_config.sep_arcsec = sep # arcsec

    cattask = CatalogShearTask(config=task_config)
    truthCatalog = cattask.run(
        tract_info=skymap[tract_id],
        seed=seed,
    ).truthCatalog
    for item in truthCatalog:
        item[2] += rotId * np.pi / 2 # fix all galaxies' angle to 180 degree
    
    config = MultibandSimConfig()
    config.survey_name = (
        "lsst"  # The scale parameter needs to be consistent with scale
    )
    config.draw_image_noise = False
    config.truncate_stamp_size = 65
    #config.rotId = rotId
    config.psf_e1 = psf_e
    config.psf_e2 = -psf_e

    sim_task = MultibandSimTask(config=config)
    outcome = sim_task.run(
                tract_info=skymap[tract_id],
                patch_id=patch_id,
                band=band,
                seed=seed,
                truthCatalog=truthCatalog,
            )

    # Galaxy image
    gal_array = np.asarray(
        outcome.simExposure.image.array,
        np.float64,
    )

    # PSF image
    lsst_psf = outcome.simExposure.getPsf()
    xc = int(dim // 2)  # we need PSF model without subpixel offset
    yc = int(dim // 2)
    psf_array = np.asarray(
        anacal.utils.resize_array(
            lsst_psf.computeImage(geom.Point2D(xc, yc)).getArray(), (64, 64)
        ),
        dtype=np.float64,
    )

    # Truth Catalog
    cat_dtype = [
    ("indices", "i8"),
    ("redshift", "f8"),
    ("angles", "f8"),
    ("gamma1", "f8"), ("gamma2", "f8"), ("kappa", "f8"),
    ("dx", "f8"), ("dy", "f8"),
    ("ra", "f8"), ("dec", "f8"),       # post-lensed ra, dec
    ("image_x", "f8"), ("image_y", "f8"),
    ("prelensed_image_x", "f8"), ("prelensed_image_y", "f8"),
    ("has_finite_shear", "bool"),('hlr', '<f8')
]
    gal_cat = np.array(truthCatalog, dtype=cat_dtype)
    gal_cat = pd.DataFrame(gal_cat)


    gal_ref = cat_ref[gal_cat['indices']]
    e = get_weighted_e(gal_ref['a_b'], gal_ref['b_b'], gal_ref['pa_bulge'], gal_ref['fluxnorm_bulge']*1e21,
                          gal_ref['a_d'], gal_ref['b_d'], gal_ref['pa_disk'], gal_ref['fluxnorm_disk']*1e21,
                            gal_cat['angles'])
    gal_cat['e1'] = e[0]
    gal_cat['e2'] = e[1]
    # Cutout Images
    cutouts = cutout_xlens_image(gal_array, gal_cat, seed, noise_level=0)

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    full_img_path = os.path.join(output_dir, "full_image.npy")

    # Check if file exists before deleting
    '''if os.path.exists(full_img_path):
        os.remove(full_img_path)
        print(f"Deleted: {full_img_path}")
    else:
        print(f"File not found: {full_img_path}")'''
    np.save(os.path.join(output_dir, "psf_image.npy"), psf_array)
    #np.save(full_img_path, gal_array)
    np.save(os.path.join(output_dir, "cutouts.npy"), cutouts)
    gal_cat.to_csv(os.path.join(output_dir, "gt_cat.csv"), index=False)

class xlen_simulator():
    def __init__(self, output_dir, 
                 abs_shear = 0.02, 
                 ori_seed = 666,
                 image_size = 1000,
                 pixel_scale = 0.2,
                 separation = 18.0,
                 num_workers=8,
                 has_shift = True,
                 init_id=0,
                 psf_e = 0.0,
                 rotId=0,
                 mode = 'calibration'):
        self.seed = ori_seed
        self.output_dir = output_dir
        self.abs_shear = abs_shear
        self.image_size = image_size
        self.pixel_scale = pixel_scale
        self.separation = separation
        self.num_workers = num_workers
        self.has_shift = has_shift
        self.init_id = init_id
        self.rotId = rotId
        self.psf_e = psf_e
        self.mode = mode

    def __call__(self, num_sims: int):
        print(f"Starting xlens simulations: mode={self.mode}, num_sims={num_sims}, has_shift={self.has_shift}, rotId={self.rotId}")
        if self.mode == 'calibration':
            shear_tasks = [
                #(2, "g1"),
                #(2, "g2"),
                (0, "g1"),
                (1, "g1"),
                (0, "g2"),
                (1, "g2"),
            ]
        elif self.mode == 'training':
            shear_tasks = [
                (2, "g1"),
            ]
        seeds = np.random.default_rng(self.seed).integers(0, 1e8, size=num_sims)
        results = []

        for shear_mode, shear_comp in shear_tasks:
            args = [(self, self._worker, i+self.init_id, seeds[i], shear_mode, shear_comp) for i in range(num_sims)]
            with Pool(processes=self.num_workers) as pool:
                res = list(tqdm(
                    pool.imap_unordered(_worker_wrapper, args,chunksize=4),
                    total=num_sims,
                    desc=f"Mode {shear_mode}, Comp {shear_comp}"
                ))
                results.extend(res)
        '''for shear_mode, shear_comp in shear_tasks:
            for index in tqdm(range(num_sims), desc=f"Mode {shear_mode}, Comp {shear_comp}"):
                seed = seeds[index]
                output_dir = f'{self.output_dir}/{shear_comp}_{shear_mode}_val_{self.abs_shear:.3f}/img_{int(index)}/'
                temp = xlens_gal_sim(
                    output_dir = output_dir,
                    shear_mode = shear_mode,
                    shear_comp = shear_comp,
                    shear_value = self.abs_shear,
                    seed = seed,
                    dim = self.image_size,
                    pixel_scale = self.pixel_scale,
                    sep = self.separation,
                )
                results.append(temp)'''

        return results

    def _worker(self, index, seed, shear_mode, shear_comp):
        output_dir = f'{self.output_dir}/{shear_comp}_{shear_mode}_val_{self.abs_shear:.3f}/img_{int(index)}/'
        xlens_gal_sim(
            output_dir = output_dir,
            shear_mode = shear_mode,
            shear_comp = shear_comp,
            shear_value = self.abs_shear,
            rotId= self.rotId,
            seed = seed,
            dim = self.image_size,
            pixel_scale = self.pixel_scale,
            sep = self.separation,
            psf_e=self.psf_e,
            has_shift=self.has_shift,
        )

psf_e = 0
print(f'psf_e: {psf_e}')
#simulator = xlen_simulator('/work/hdd/bdsp/wenyinli/datasets/xlens_fixed/', 
#                            num_workers=128, mode='calibration', 
#                            rotId=0,
#                            psf_e=psf_e,
#                            has_shift=False, ori_seed=666)
#simulator(100000)
simulator = xlen_simulator('/work/hdd/bdsp/wenyinli/datasets/xlens_fixed/', 
                            num_workers=128, mode='calibration',
                            rotId=1, init_id=100000,
                            psf_e=psf_e,
                            has_shift=False, ori_seed=666)
simulator(100000)
#simulator = xlen_simulator('/work/nvme/bfmo/wenyinli/datasets/xlens_train/', 
#                           num_workers=64, ori_seed=20240411,
#                           mode='training')

