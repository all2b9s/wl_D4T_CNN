import anacal
import numpy as np
import matplotlib.pylab as plt
import lsst.geom as geom
import fitsio
import os
# LSST task to define DC2-like skymap 
from lsst.skymap.discreteSkyMap import (
    DiscreteSkyMapConfig, DiscreteSkyMap,
)

from xlens.simulator.catalog import (
    CatalogShearTask,
    CatalogShearTaskConfig,
)
from xlens.simulator.sim import (
    MultibandSimConfig, MultibandSimTask
)

from astropy.visualization import simple_norm
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
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
        around_corner = False,
        galaxy_profile = 1,
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
    # rotate intrinsic angles (xlens stores angles in radians)
    truthCatalog["angles"] += rotId * np.pi / 2
    if around_corner:
        # place each galaxy half a pixel off the pixel centre
        wcs_tmp = skymap[tract_id].getWcs()
        for col_ra, col_dec in (("ra", "dec"), ("prelensed_ra", "prelensed_dec")):
            ix, iy = wcs_tmp.skyToPixelArray(
                truthCatalog[col_ra], truthCatalog[col_dec], degrees=True,
            )
            truthCatalog[col_ra], truthCatalog[col_dec] = wcs_tmp.pixelToSkyArray(
                ix - 0.5, iy - 0.5, degrees=True,
            )

    config = MultibandSimConfig()
    config.survey_name = (
        "lsst"  # The scale parameter needs to be consistent with scale
    )
    config.draw_image_noise = False
    config.truncate_stamp_size = 65
    config.force_galaxy_profile = galaxy_profile # profile (1: gaussian, 2: exponential); 0 keeps the catalog's native
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
    ("has_finite_shear", "bool"), ('hlr', '<f8')
]
    # The current xlens truth catalog only carries positions as (ra, dec) and
    # (prelensed_ra, prelensed_dec) in degrees. Recover the pixel positions
    # with the same WCS the simulator uses to place galaxies, then copy the
    # shared fields by name (the field lists no longer align 1-to-1).
    wcs = skymap[tract_id].getWcs()
    image_x, image_y = wcs.skyToPixelArray(
        truthCatalog["ra"], truthCatalog["dec"], degrees=True,
    )
    prelensed_image_x, prelensed_image_y = wcs.skyToPixelArray(
        truthCatalog["prelensed_ra"], truthCatalog["prelensed_dec"], degrees=True,
    )

    gal_cat = np.zeros(len(truthCatalog), dtype=cat_dtype)
    for col in cat_dtype:
        name = col[0]
        if name in truthCatalog.dtype.names:
            gal_cat[name] = truthCatalog[name]
    gal_cat["image_x"] = image_x
    gal_cat["image_y"] = image_y
    gal_cat["prelensed_image_x"] = prelensed_image_x
    gal_cat["prelensed_image_y"] = prelensed_image_y
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
    np.save(os.path.join(output_dir, "psf_image.npy"), psf_array)
    np.save(os.path.join(output_dir, "cutouts.npy"), cutouts)
    gal_cat.to_csv(os.path.join(output_dir, "gt_cat.csv"), index=False)
    #np.save('./plots/patch_img.npy',gal_array)
    plt.imshow(gal_array, norm=simple_norm(gal_array, 'sqrt', percent=99.5), cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join('./plots/', "full_image.png"), bbox_inches='tight', pad_inches=0)
    plt.close()
    return gal_array,psf_array,gal_cat

class xlen_simulator():
    def __init__(self, output_dir, 
                 abs_shear = 0.02, 
                 ori_seed = 666,
                 image_size = 1000,
                 pixel_scale = 0.2,
                 num_workers=8,
                 has_shift = True,
                 around_corner = False,
                 init_id=0,
                 psf_e = 0.0,
                 rotId=0,
                 galaxy_profile=1,
                 mode = 'calibration'):
        self.seed = ori_seed
        self.output_dir = output_dir
        self.abs_shear = abs_shear
        self.image_size = image_size
        self.pixel_scale = pixel_scale
        self.num_workers = num_workers
        self.around_corner = around_corner
        self.has_shift = has_shift
        self.init_id = init_id
        self.rotId = rotId
        self.psf_e = psf_e
        self.galaxy_profile = galaxy_profile
        self.mode = mode

    def __call__(self, num_sims: int):
        print(f"Starting xlens simulations: mode={self.mode}, num_sims={num_sims}, has_shift={self.has_shift}, rotId={self.rotId}, galaxy_profile={self.galaxy_profile}")
        print(f"Saving at {self.output_dir}")
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
        print("Simulation finished")
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
            sep = 90*self.pixel_scale,
            psf_e=self.psf_e,
            has_shift=self.has_shift,
            around_corner = self.around_corner,
            galaxy_profile = self.galaxy_profile,
        )

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Single-galaxy xlens simulation wrapper"
    )
    parser.add_argument(
        "--output_dir",
        default="/work/hdd/bdsp/wenyinli/datasets/xlens_gauss_distribution/",
    )
    parser.add_argument(
        "--mode",
        default="calibration",
        choices=["calibration", "training"],
    )
    parser.add_argument("--abs_shear", type=float, default=0.02)
    parser.add_argument("--ori_seed", type=int, default=666)
    parser.add_argument("--image_size", type=int, default=1000)
    parser.add_argument("--pixel_scale", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=128)
    parser.add_argument("--has_shift", type=int, default=1)
    parser.add_argument("--around_corner", type=int, default=0)
    parser.add_argument("--init_id", type=int, default=0)
    parser.add_argument("--psf_e", type=float, default=0.0)
    parser.add_argument("--rotId", type=int, default=0)
    parser.add_argument(
        "--galaxy_profile",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="galaxy light profile (1: gaussian, 2: exponential); 0 keeps the catalog's native",
    )
    parser.add_argument("--num_sims", type=int, default=200_000)

    args = parser.parse_args()

    simulator = xlen_simulator(
        args.output_dir,
        abs_shear=args.abs_shear,
        ori_seed=args.ori_seed,
        image_size=args.image_size,
        pixel_scale=args.pixel_scale,
        num_workers=args.num_workers,
        has_shift=bool(args.has_shift),
        around_corner=bool(args.around_corner),
        init_id=args.init_id,
        psf_e=args.psf_e,
        rotId=args.rotId,
        galaxy_profile=args.galaxy_profile,
        mode=args.mode,
    )
    simulator(args.num_sims)


if __name__ == "__main__":
    main()


