"""Step 6: build the (images .npy, labels .csv) pair read by src.training.train_model.

Takes train/val/test objects from the RDSM catalog, splitting by image directory (no directory is shared
between splits). The sim cutouts are raw (PSF-convolved, no deconvolution, no renoise), so by default they go
through the same preparation the teacher saw (src.anacal prepare_q_images: PSF-deconvolving resmoothing to a
Gaussian of FWHM sim.psf_fwhm). dataset.noise.mode:
  adaptive : adaptive noise added, renoise map used, then prepare_q_images   (matches noisy measurement;
             same recipe and seeds as scripts/make_renoised_10k_images_64.py)
  q_clean  : prepare_q_images on the noiseless cutout (what the teacher measured at noise 0)
  none     : raw cutouts, NOT prepared -- only for experiments, does not match measurement-time inputs
Galaxies: student_sim (if set) = galaxies different from the teacher's `sim`; the labels then come from the s_label
step (eDSM of `sim` applied to the teacher shapes measured on them). dataset.image_sim only re-renders the SAME
galaxies from another directory; every used row's catsim `indices` in <image dir>/img_k/gt_cat.csv must equal the
label catalog's, otherwise the step aborts.

Row i of images.npy == row i of catalog.csv (id = i). Label columns e1,e2 come from dataset.label:
rdsm | truth (e_true) | teacher (measured teacher shape).
"""
import numpy as np

from ml_wl.config import step_dir
from ml_wl.steps._common import d, load_job_args, sg_job

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
    sim, noise, img_id, cr, expect = args
    import pandas as pd
    from src.anacal.batch_calibration import prepare_q_images
    from src.anacal.cal_toolkit import load_img
    mode = noise["mode"]
    if mode == "none":
        cat = pd.read_csv(f"{sim['split_dir']}/img_{img_id}/gt_cat.csv")
        _check_alignment(cat, cr, expect, sim, img_id)
        raw = np.load(f"{sim['split_dir']}/img_{img_id}/cutouts.npy")[cr]
        return raw.astype(np.float32), np.zeros(len(cr), np.float32)
    kw = dict(noise_level=noise["level"], noise_mode="adaptive") if mode == "adaptive" else \
        dict(noise_level=0.0, noise_mode="fixed")
    cut, psf, cat, nl = load_img(sim["comp"], sim["mode"], img_id, abs_shear_value=sim["abs_shear"],
                                 directory=sim["dir"], **kw)
    _check_alignment(cat, cr, expect, sim, img_id)
    cat = cat.iloc[cr].reset_index(drop=True)
    cut, psf, nl = cut[cr], psf[cr], nl[cr]
    if mode == "adaptive":
        renoise = np.random.default_rng(20020620 + img_id).standard_normal(cut.shape) * nl.reshape(-1, 1, 1)
    else:
        renoise = np.zeros_like(cut)
    q = prepare_q_images(cut, psf, renoise, cat, sigma_arcsec=sim["psf_fwhm"] / 2.355, flim=FLIM, workers=1)
    return q[:, 0].astype(np.float32), nl.astype(np.float32)


def _check_alignment(cat, cr, expect, sim, img_id):
    got = cat["indices"].to_numpy()[cr]
    if not np.array_equal(got, expect):
        bad = int((got != expect).sum())
        raise ValueError(f"{sim['split_dir']}/img_{img_id}: {bad}/{len(cr)} rows are different galaxies than the "
                         "ones the RDSM labels were measured on (catsim index mismatch)")


def main():
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    import pandas as pd

    cfg, _ = load_job_args("dataset")
    ds = cfg["dataset"]
    sim = ds["sim"]
    out = d(cfg, "dataset")
    label_step = "s_label" if cfg["student_sim"] is not None else "rdsm"
    with np.load(step_dir(cfg, label_step) / "rdsm_catalog.npz") as z:
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
    tasks = [(sim, ds["noise"], i, npz["catalog_row"][rows[p]].astype(int), npz["indices"][rows[p]])
             for i, p in order.items()]
    images = np.lib.format.open_memmap(out / "images.npy", mode="w+", dtype=np.float32, shape=(len(rows), 64, 64))
    sigma = np.zeros(len(rows), np.float32)
    print(f"{len(rows)} objects from {len(tasks)} image dirs; split sizes "
          f"{ {s: len(v) for s, v in sel.items()} }; noise={ds['noise']}; images from {sim['split_dir']}", flush=True)
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
