import argparse
from datasets.single_gal_sims import build_dataset
import numpy as np 
    
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
p.add_argument("--psf_para_range", type=float, nargs="+", required=True)
p.add_argument("--noise_std_range", type=float, nargs=2, default=[0, 0])

# dataset/global knobs
p.add_argument("--N", type=int, default=100_000)
p.add_argument("--image_size", type=int, default=64)
p.add_argument("--pixel_scale", type=float, default=0.2)
p.add_argument("--batch_size", type=int, default=1000)
p.add_argument("--num_workers", type=int, default=32)

args = p.parse_args()

if args.psf_model.lower() == "gaussian":
    if len(args.psf_para_range) != 2:
        raise ValueError("Gaussian requires [fwhm_min fwhm_max]")
    psf_range = np.array([args.psf_para_range], dtype=float)  # shape (1,2)
elif args.psf_model.lower() == "moffat":
    if len(args.psf_para_range) != 4:
        raise ValueError("Moffat requires [beta_min beta_max fwhm_min fwhm_max]")
    psf_range = np.array([args.psf_para_range[:2], args.psf_para_range[2:]], dtype=float)  # shape (2,2)
else:
    raise ValueError(f"Unknown psf_model {args.psf_model}")

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