import os
import math
import galsim
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ---------------- simulation function (unchanged) ----------------
import matplotlib.pyplot as plt
def sim_simple_gal(e1=0.0, e2=0.0,
                   g1=0, g2=0,
                   gal_flux=1000,
                   gal_half_light_radius=0.5,
                   sersic_n=1,
                   psf_fwhm=0.7,
                   pixel_scale=0.2,
                   image_size=50,
                   noise_std=0,
                   seed=42,
                   show=False):
    gal = galsim.Sersic(flux=gal_flux,
                        half_light_radius=gal_half_light_radius,
                        n=sersic_n).shear(e1=e1, e2=e2)
    gal = gal.shear(e1=g1, e2=g2)
    psf = galsim.Gaussian(fwhm=psf_fwhm)
    final = galsim.Convolve([gal, psf])
    image = final.drawImage(scale=pixel_scale, nx=image_size, ny=image_size)
    rng = galsim.BaseDeviate(int(seed))
    noise = galsim.GaussianNoise(rng, sigma=noise_std)
    image.addNoise(noise)
    if show:
        plt.imshow(image.array, origin='lower', cmap='gray')
        plt.title(f'Simulated Galaxy (e1={e1:.2f}, e2={e2:.2f})')
        plt.colorbar(label='ADU')
        plt.tight_layout(); plt.show()
    return image.array

# ---------------- parameter sampling ----------------
def _sample_one(rng: np.random.Generator):
    while True:
        e1 = rng.uniform(-0.6, 0.6)
        e2 = rng.uniform(-0.6, 0.6)
        if e1*e1 + e2*e2 < 0.8*0.8:
            break
    g1 = 0
    g2 = 0
    gal_flux = float(np.exp(rng.uniform(np.log(1000.0), np.log(50000.0))))
    gal_hlr = float(np.exp(rng.uniform(np.log(0.6), np.log(1.2))))
    sersic_choices = np.array([1.0, 1.5, 2.0, 3.0, 4.0])
    probs = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
    sersic_n = float(rng.choice(sersic_choices, p=probs))
    psf_fwhm = 0.7
    base = rng.uniform(0.8, 1.2)
    noise_std = 0
    return e1, e2, g1, g2, gal_flux, gal_hlr, sersic_n, psf_fwhm, noise_std

# ---------------- worker ----------------
def _worker(i, seed_base, image_size, pixel_scale):
    rng = np.random.default_rng(seed_base + i)
    e1, e2, g1, g2, flux, hlr, n, fwhm, noise = _sample_one(rng)
    img = sim_simple_gal(
        e1=e1, e2=e2, g1=g1, g2=g2,
        gal_flux=flux,
        gal_half_light_radius=hlr,
        sersic_n=n,
        psf_fwhm=fwhm,
        pixel_scale=pixel_scale,
        image_size=image_size,
        noise_std=noise,
        seed=seed_base + i,
        show=False
    ).astype(np.float32, copy=False)

    return img, [i, e1, e2, g1, g2, flux, hlr, n, fwhm, noise, pixel_scale, seed_base + i]

# ---------------- dataset builder ----------------
def build_dataset(
    out_images_npy="images.npy",
    out_labels_csv="gt_info.csv",
    N=100_000,
    image_size=50,
    pixel_scale=0.2,
    seed=20250811,
    batch_size=5000,
    num_workers=None,
):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    # Pre-allocate memmap for all images
    images = np.lib.format.open_memmap(
        out_images_npy, mode="w+", dtype=np.float32, shape=(N, image_size, image_size)
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
        "sersic_n", "psf_fwhm_arcsec", "noise_std_ADU",
        "pixel_scale_arcsec_per_pix", "seed", "split"
    ]

    # Multiprocessing pool
    with Pool(processes=num_workers) as pool:
        for start in tqdm(range(0, N, batch_size), desc="Generating", unit="img"):
            stop = min(N, start + batch_size)
            args = [(i, seed, image_size, pixel_scale) for i in range(start, stop)]
            results = pool.starmap(_worker, args, chunksize=64)

            for img, meta in results:
                i = meta[0]
                images[i] = img
                rows.append(meta + [split_arr[i]])

    # Close memmap
    del images

    # Save CSV
    df = pd.DataFrame(rows, columns=header)
    df.to_csv(out_labels_csv, index=False)

    print(f"Done.\nImages: {out_images_npy}\nLabels: {out_labels_csv}")
    print(f"Split counts: train={np.sum(split_arr=='train')}, val={np.sum(split_arr=='val')}, test={np.sum(split_arr=='test')}")

if __name__ == "__main__":
    build_dataset(
        out_images_npy="/projects/bdsp/wenyinli/datasets/simple_gal_images.npy",
        out_labels_csv="/projects/bdsp/wenyinli/datasets/simple_gal_info.npy",
        N=100_000,
        image_size=50,
        pixel_scale=0.2,
        seed=20020103,
        batch_size=1000,
        num_workers=32,  # auto: CPU-1
    )
