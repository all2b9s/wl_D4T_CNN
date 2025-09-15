import os
import math
import galsim
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import argparse


# ---------------- simulation function (unchanged) ----------------
import matplotlib.pyplot as plt
def sim_simple_gal(e1=0.0, e2=0.0,
                   g1=0, g2=0,
                   gal_flux=1000,
                   gal_half_light_radius=0.5,
                   sersic_n=1,
                   psf_model = 'Gaussian',
                   psf_para=0.7,
                   pixel_scale=0.2,
                   image_size=100,
                   noise_std=0,
                   seed=42,
                   shift = [0,0],
                   return_psf = False,
                   show=False):
    gal = galsim.Sersic(flux=gal_flux,
                        half_light_radius=gal_half_light_radius,
                        n=sersic_n).shear(e1=e1, e2=e2)
    gal = gal.shear(g1=g1, g2=g2)

    if psf_model == 'Gaussian':
        psf_fwhm = psf_para
        psf = galsim.Gaussian(fwhm=psf_fwhm)
    elif psf_model == 'Moffat':
        psf_beta, psf_fwhm = psf_para
        psf = galsim.Moffat(fwhm=psf_fwhm, beta=psf_beta)

    final = galsim.Convolve([gal, psf]).shift(shift[0], shift[1])
    image = final.drawImage(scale=pixel_scale, nx=image_size, ny=image_size, method="no_pixel")
    if noise_std > 0:
        rng = galsim.BaseDeviate(int(seed))
        noise = galsim.GaussianNoise(rng, sigma=noise_std)
        image.addNoise(noise)
    
    if show:
        plt.imshow(image.array, origin='lower', cmap='gray')
        plt.title(f'Simulated Galaxy (e1={e1:.2f}, e2={e2:.2f})')
        plt.colorbar(label='ADU')
        plt.tight_layout(); plt.show()
    if return_psf:
        psf_image = psf.drawImage(scale=pixel_scale, nx=image_size, ny=image_size, method="no_pixel")
        return np.stack((image.array, psf_image.array))
    else:
        return image.array

# ---------------- parameter sampling ----------------
    
class single_gal_simulator():
    def __init__(self, seed=42, 
                 shear = [0,0],
                 hlr_range = [0.6, 1.2],
                 flux_range = [1e3, 5e4],
                 e_max = 0.6,
                 shift_std = 0.0,
                 n_range = [1,5],
                 psf_model = 'Moffat',
                 psf_para_range = [[2,3],[0.6,0.8]], # [[beta_min,max],[fwhm_min,max]]
                 noise_std_range = [0,0], # ADU
                 ):
        self.seed_base = seed

        self.shear = shear
        self.hlr_range = hlr_range
        self.flux_range = flux_range
        self.e_max = e_max
        self.shift_std = shift_std  
        self.n_range = n_range
        self.psf_model = psf_model
        self.psf_para_range = np.array(psf_para_range, dtype=float)  # (P, 2)
        self.noise_std_range = noise_std_range

    def _sample_one(self, rng: np.random.Generator):
        # symmetric box for (e1, e2), then enforce ellipse if you like
        while True:
            e1 = rng.uniform(-self.e_max, self.e_max)
            e2 = rng.uniform(-self.e_max, self.e_max)
            if e1*e1 + e2*e2 < 0.9*0.9:
                break

        g1, g2 = self.shear[0], self.shear[1]

        shift_1 = rng.normal(0, self.shift_std)
        shift_2 = rng.normal(0, self.shift_std)

        # log-uniform
        gal_flux = float(np.exp(rng.uniform(np.log(self.flux_range[0]), np.log(self.flux_range[1]))))
        gal_hlr  = float(np.exp(rng.uniform(np.log(self.hlr_range[0]),  np.log(self.hlr_range[1]))))

        sersic_n = float(rng.uniform(self.n_range[0], self.n_range[1]))

        # sample each PSF parameter within its [min,max]
        psf_para = np.zeros(self.psf_para_range.shape[0], dtype=float)
        for i in range(self.psf_para_range.shape[0]):
            lo, hi = self.psf_para_range[i]
            psf_para[i] = rng.uniform(lo, hi)

        noise_std  = float(rng.uniform(self.noise_std_range[0], self.noise_std_range[1]))

        return e1, e2, g1, g2, gal_flux, gal_hlr, sersic_n, [shift_1, shift_2], psf_para, noise_std

    def _worker(self, i, image_size, pixel_scale):
        rng = np.random.default_rng(self.seed_base + i)
        e1, e2, g1, g2, flux, hlr, n, shift, psf_para, noise = self._sample_one(rng)

        sim_outputs = sim_simple_gal(
            e1=e1, e2=e2, g1=g1, g2=g2,
            gal_flux=flux,
            gal_half_light_radius=hlr,
            sersic_n=n,
            psf_model=self.psf_model,
            psf_para=psf_para,
            pixel_scale=pixel_scale,
            shift=shift,
            image_size=image_size,
            noise_std=noise,
            seed=self.seed_base + i,
            show=False,
            return_psf=True,
        ).astype(np.float32, copy=False)

        # make meta CSV-friendly
        shift_t = (float(shift[0]), float(shift[1]))
        psf_t   = tuple(map(float, np.atleast_1d(psf_para)))

        meta = [i, e1, e2, g1, g2, flux, hlr, n, shift_t, psf_t, noise, pixel_scale, self.seed_base + i]
        return sim_outputs, meta

# ---------------- dataset builder ----------------
import os  # Add this import at the top

def build_galsim_dataset(
    out_images_npy="images.npy",
    out_psf_npy="psfs.npy",
    out_labels_csv="gt_info.csv",
    shear = [0,0],
    sim_args = {},
    N=100_000,
    image_size=50,
    pixel_scale=0.2,
    seed=20250811,
    batch_size=5000,
    num_workers=None,
):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_images_npy), exist_ok=True)
    os.makedirs(os.path.dirname(out_psf_npy), exist_ok=True)
    os.makedirs(os.path.dirname(out_labels_csv), exist_ok=True)

    # Pre-allocate memmap for all images
    images = np.lib.format.open_memmap(
        out_images_npy, mode="w+", dtype=np.float32, shape=(N, image_size, image_size)
    )
    psfs = np.lib.format.open_memmap(
        out_psf_npy, mode="w+", dtype=np.float32, shape=(N, image_size, image_size)
    )

    # Precompute split
    rng_split = np.random.default_rng(seed + 999)
    idx = np.arange(N, dtype=np.int64)
    rng_split.shuffle(idx)
    n_train = int(0.7 * N)
    n_val   = int(0.1 * N)
    split_arr = np.empty(N, dtype="<U5")
    split_arr[idx[:n_train]] = "train"
    split_arr[idx[n_train:n_train+n_val]] = "val"
    split_arr[idx[n_train+n_val:]] = "test"

    rows = []
    header = [
        "id", "e1", "e2", "g1", "g2", "flux", "hlr_arcsec",
        "sersic_n", "off-center_pix", "psf_parameters", "noise_std_ADU",
        "pixel_scale_arcsec_per_pix", "seed", "split"
    ]

    # Multiprocessing pool
    simulator = single_gal_simulator(seed=seed, shear=shear, **sim_args)
    with Pool(processes=num_workers) as pool:
        for start in tqdm(range(0, N, batch_size), desc="Generating", unit="img"):
            stop = min(N, start + batch_size)
            args = [(i, image_size, pixel_scale) for i in range(start, stop)]
            results = pool.starmap(simulator._worker, args, chunksize=64)

            for img, meta in results:
                i = meta[0]
                images[i] = img[0]
                psfs[i] = img[1]
                rows.append(meta + [split_arr[i]])

    # Close memmap
    del images
    del psfs

    # Save CSV
    df = pd.DataFrame(rows, columns=header)
    df.to_csv(out_labels_csv, index=False)

    print(f"Done.\nImages: {out_images_npy}\nLabels: {out_labels_csv}")
    print(f"Split counts: train={np.sum(split_arr=='train')}, val={np.sum(split_arr=='val')}, test={np.sum(split_arr=='test')}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--storage", type=str, required=True, help="Prefix for output files, e.g., /path/to/Moffat_gal")
    p.add_argument("--shear", type=float, nargs=2, default=[0.0, 0.0])
    p.add_argument("--seed", type=int, default=20020103)

    p.add_argument("--hlr_range", type=float, nargs=2, default=[0.6, 1.2])
    p.add_argument("--flux_range", type=float, nargs=2, default=[1e3, 5e4])
    p.add_argument("--e_max", type=float, default=0.7)
    p.add_argument("--shift_std", type=float, default=0.0)
    p.add_argument("--n_range", type=float, nargs=2, default=[1, 5])  # allow float n
    p.add_argument("--psf_model", type=str, default="Moffat")
    # For Moffat: beta_min beta_max fwhm_min fwhm_max
    p.add_argument("--psf_para_range", type=float, nargs=4, default=[2, 3, 0.6, 0.8])
    p.add_argument("--noise_std_range", type=float, nargs=2, default=[0, 0])

    # dataset/global knobs
    p.add_argument("--N", type=int, default=100_000)
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--pixel_scale", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=1000)
    p.add_argument("--num_workers", type=int, default=32)

    args = p.parse_args()

    # shape (2,2) [[beta_min,max],[fwhm_min,max]]
    psf_range = np.array(args.psf_para_range, dtype=float).reshape(2, 2)

    sim_args = dict(
        hlr_range=args.hlr_range,
        flux_range=args.flux_range,
        e_max=args.e_max,
        shift_std=args.shift_std,
        n_range=args.n_range,
        psf_model=args.psf_model,
        psf_para_range=psf_range,
        noise_std_range=args.noise_std_range,
    )

    build_dataset(
        out_images_npy=f"{args.storage}_images.npy",
        out_psf_npy=f"{args.storage}_psfs.npy",
        out_labels_csv=f"{args.storage}_info.csv",
        shear=args.shear,
        sim_args=sim_args,
        N=args.N,
        image_size=args.image_size,
        pixel_scale=args.pixel_scale,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )