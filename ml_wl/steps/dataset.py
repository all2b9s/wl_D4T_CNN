"""Step 6: build the (images .npy, labels .csv) pair read by src.training.train_model.

Takes train/val/test objects from the RDSM catalog, splitting by image directory (no directory is shared
between splits). Images are either the clean cutouts (noise.mode=none) or, like
src.shape_measurement feeds the CNN, adaptive-noise images resmoothed with a renoise map
(noise.mode=adaptive; same recipe and seeds as scripts/make_renoised_10k_images_64.py), full 64x64.

Row i of images.npy == row i of catalog.csv (id = i). Label columns e1,e2 come from dataset.label:
rdsm | truth (e_true) | teacher (measured teacher shape).
"""
import numpy as np

from ml_wl.config import step_dir
from ml_wl.steps._common import d, load_job_args, sg_job

SIGMA_ARCSEC_FWHM = 0.85
FLIM = 10.0


def jobs(cfg):
    return [dict(lines=sg_job("dataset", cfg), cwd="sg")]


def _labels(npz, kind):
    if kind == "rdsm":
        return np.stack([npz["rdsm_e1"], npz["rdsm_e2"]], 1)
    if kind == "truth":
        return npz["e_true"]
    if kind == "teacher":
        return npz["shapes"]
    raise ValueError(f"dataset.label must be rdsm|truth|teacher, got {kind!r}")


def _select(img, usable, counts, seed):
    """Whole image dirs, in random order, test -> val -> train; the last dir of a split may be cut short."""
    dirs = np.random.default_rng(seed).permutation(np.unique(img))
    pos, out = 0, {}
    for split in ("test", "val", "train"):
        need, rows = counts[split], []
        while pos < len(dirs) and (need == "all" or len(rows) < need):
            r = np.flatnonzero((img == dirs[pos]) & usable)
            pos += 1
            if need != "all":
                r = r[: need - len(rows)]
            rows.extend(r.tolist())
        if need != "all" and len(rows) < need:
            raise ValueError(f"not enough usable objects for split {split}: {len(rows)} < {need}")
        out[split] = np.asarray(rows, dtype=np.int64)
    return out


def _render(args):
    """One image dir -> (n,64,64) float32 model-input images and per-object noise sigma."""
    sim, noise, img_id, cr = args
    from src.anacal.batch_calibration import prepare_q_images
    from src.anacal.cal_toolkit import load_img
    if noise["mode"] == "none":
        raw = np.load(f"{sim['split_dir']}/img_{img_id}/cutouts.npy")[cr]
        return raw.astype(np.float32), np.zeros(len(cr), np.float32)
    cut, psf, cat, nl = load_img(sim["comp"], sim["mode"], img_id, abs_shear_value=sim["abs_shear"],
                                 noise_level=noise["level"], noise_mode=noise["mode"], directory=sim["dir"])
    cat = cat.iloc[cr].reset_index(drop=True)
    cut, psf, nl = cut[cr], psf[cr], nl[cr]
    renoise = np.random.default_rng(20020620 + img_id).standard_normal(cut.shape) * nl.reshape(-1, 1, 1)
    q = prepare_q_images(cut, psf, renoise, cat, sigma_arcsec=SIGMA_ARCSEC_FWHM / 2.355, flim=FLIM, workers=1)
    return q[:, 0].astype(np.float32), nl.astype(np.float32)


def main():
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    import pandas as pd

    cfg, _ = load_job_args("dataset")
    ds, sim = cfg["dataset"], cfg["sim"]
    out = d(cfg, "dataset")
    with np.load(step_dir(cfg, "rdsm") / "rdsm_catalog.npz") as z:
        npz = {k: z[k] for k in z.files}
    labels = _labels(npz, ds["label"])
    usable = np.isfinite(labels).all(1)
    if ds["label"] == "rdsm":
        usable &= npz["rdsm_valid"].astype(bool)
    counts = {"train": ds["n_train"], "val": ds["n_val"], "test": ds["n_test"]}
    sel = _select(npz["img"], usable, counts, ds["seed"])

    rows = np.concatenate([sel[s] for s in ("train", "val", "test")])
    split = np.concatenate([[s] * len(sel[s]) for s in ("train", "val", "test")])
    img_ids = npz["img"][rows]
    # group by image dir, keeping the final row order
    order = {int(i): [] for i in dict.fromkeys(img_ids.tolist())}
    for pos, i in enumerate(img_ids.tolist()):
        order[i].append(pos)
    tasks = [(sim, ds["noise"], i, npz["catalog_row"][rows[p]].astype(int)) for i, p in order.items()]
    images = np.lib.format.open_memmap(out / "images.npy", mode="w+", dtype=np.float32, shape=(len(rows), 64, 64))
    sigma = np.zeros(len(rows), np.float32)
    print(f"{len(rows)} objects from {len(tasks)} image dirs; split sizes "
          f"{ {s: len(v) for s, v in sel.items()} }; noise={ds['noise']}", flush=True)
    with ProcessPoolExecutor(max_workers=ds["workers"], mp_context=mp.get_context("fork")) as ex:
        for k, ((i, pos), (im, nl)) in enumerate(zip(order.items(), ex.map(_render, tasks))):
            images[pos] = im
            sigma[pos] = nl
            if k % 20 == 0:
                print(f"{k}/{len(tasks)} dirs", flush=True)
    images.flush()

    df = pd.DataFrame({"id": np.arange(len(rows)), "split": split,
                       "e1": labels[rows, 0], "e2": labels[rows, 1], "img": img_ids,
                       "catalog_row": npz["catalog_row"][rows], "indices": npz["indices"][rows],
                       "hlr": npz["hlr"][rows], "noise_sigma": sigma,
                       "rdsm_e1": npz["rdsm_e1"][rows], "rdsm_e2": npz["rdsm_e2"][rows],
                       "true_e1": npz["e_true"][rows, 0], "true_e2": npz["e_true"][rows, 1],
                       "teacher_e1": npz["shapes"][rows, 0], "teacher_e2": npz["shapes"][rows, 1]})
    if "mag_fpfs" in npz:
        df["mag_fpfs"] = npz["mag_fpfs"][rows]
    df.to_csv(out / "catalog.csv", index=False)
    print("wrote", out / "images.npy", images.shape, out / "catalog.csv", flush=True)


if __name__ == "__main__":
    main()
