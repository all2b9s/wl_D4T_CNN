"""Thin blended-catalog adapter: measure, prepare weighted inputs, report test results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from ml_wl.config import step_dir

RESPONSE_KEYS = tuple(f"ml_R{a}{b}" for a in (1, 2) for b in (1, 2))
COMPARE_KEYS = ("x", "y", "fpfs_w", "fpfs_dw_dg1", "fpfs_dw_dg2",
                "ml_e1", "ml_e2", "ml_R11", "ml_R22")


def job(cfg, action):
    config = Path(cfg["round_dir"]) / "config.resolved.json"
    return [dict(cwd="sg", lines=[f"python -u -m ml_wl.steps.blended {action} --config {config}"])]


def jobs(cfg):
    return job(cfg, "evaluate") if cfg.get("blended") is not None else []


def measurement_dir(cfg):
    return step_dir(cfg, "measure") / "g1_2" / "attnw_full"


def selected_splits(cfg):
    b = cfg["blended"]
    splits = split_exposures(b["index_range"], b["seed"])
    if b.get("sample_targets"):
        path = Path(b["old_catalog"]) / "measurement_manifest.csv"
        with path.open() as f:
            counts = {int(r["exp_id"]): int(r["n_det"]) for r in csv.DictReader(f)}
        for name, ids in splits.items():
            cumulative = np.r_[0, np.cumsum([counts[int(i)] for i in ids])]
            n = int(np.argmin(np.abs(cumulative - b["sample_targets"][name])))
            if n == 0:
                raise ValueError(f"no exposures selected for {name}")
            splits[name] = ids[:n]
    return splits


def compare_catalog(exposure, path, old_dir):
    with np.load(path) as z, np.load(Path(old_dir) / f"catalog_{exposure}.npz") as old:
        if len(np.unique(z["det_id"])) != len(z["det_id"]) or not np.array_equal(z["det_id"], old["det_id"]):
            raise ValueError(f"exposure {exposure}: detections changed or duplicated")
        differences = {}
        for key in COMPARE_KEYS:
            atol = 1e-3 if key in ("ml_R11", "ml_R22") else 1e-7
            try:
                np.testing.assert_allclose(z[key], old[key], rtol=1e-5, atol=atol)
            except AssertionError as exc:
                raise ValueError(f"exposure {exposure}, {key}: old/new mismatch") from exc
            finite = np.isfinite(z[key]) & np.isfinite(old[key])
            differences[key] = float(np.abs(z[key][finite] - old[key][finite]).max(initial=0))
    return differences


def measure(cfg):
    from src.blending.test_func import measure_catalogs_with_run

    b = cfg["blended"]
    checkpoint = Path(cfg["teacher_cnn"]["checkpoint"])
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if b.get("checkpoint_sha256", digest) != digest:
        raise ValueError("teacher checkpoint differs from the original AttnW measurement")
    splits = selected_splits(cfg)
    ids = sorted(np.concatenate(list(splits.values())).tolist())
    selection = step_dir(cfg, "catalog") / "split_exposures.npz"
    if selection.exists():
        with np.load(selection) as z:
            for name, values in splits.items():
                np.testing.assert_array_equal(z[name], values)
    else:
        np.savez(selection, **splits)
    out = measurement_dir(cfg)
    for exposure in ids:
        path = out / f"catalog_{exposure}.npz"
        if path.exists():
            with np.load(path) as z:
                if not set(RESPONSE_KEYS).issubset(z.files):
                    raise ValueError(f"{path}: incomplete response; use a fresh measure directory")
            compare_catalog(exposure, path, b["old_catalog"])
    summary = measure_catalogs_with_run(
        run_dir=checkpoint.parent, weights=checkpoint.name,
        output_root=step_dir(cfg, "measure"), fname="attnw_full",
        index_range=tuple(b["index_range"]), exposure_ids=ids,
        catalog_callback=lambda i, p: compare_catalog(i, p, b["old_catalog"]), **b["measurement"])
    (out / "measurement_config.json").write_text(json.dumps(summary, indent=2) + "\n")
    if b.get("prepare_catalog_in_measure", False):
        catalog(cfg)


def weighted_inputs(e, response, weight, dw):
    """Axes of response are (output component, shear component)."""
    return weight[:, None] * e, weight[:, None, None] * response + e[:, :, None] * dw[:, None, :]


def split_exposures(index_range, seed=42):
    ids = np.random.default_rng(seed).permutation(np.arange(*index_range))
    n = len(ids)
    return dict(zip(("train", "val", "test"), np.split(ids, [int(.4 * n), int(.6 * n)])))


def catalog(cfg):
    b = cfg["blended"]
    splits = selected_splits(cfg)
    labels = {int(i): code for code, ids in enumerate(splits.values()) for i in ids}
    parts, audit = [], {"input_transform": "detw_e", "seed": b["seed"],
                        "old_catalog": b["old_catalog"], "rows": 0, "rejected": 0,
                        "split_codes": {"train": 0, "val": 1, "test": 2},
                        "max_abs_difference": {k: 0.0 for k in COMPARE_KEYS},
                        "comparison_rtol": 1e-5,
                        "comparison_atol": {k: 1e-3 if k in ("ml_R11", "ml_R22") else 1e-7
                                            for k in COMPARE_KEYS}}
    for exposure in sorted(labels):
        path = measurement_dir(cfg) / f"catalog_{exposure}.npz"
        delta = compare_catalog(exposure, path, b["old_catalog"])
        for key, value in delta.items():
            audit["max_abs_difference"][key] = max(audit["max_abs_difference"][key], value)
        with np.load(path) as z:
            data = {k: z[k] for k in (*RESPONSE_KEYS, "ml_e1", "ml_e2", "fpfs_w",
                                      "fpfs_dw_dg1", "fpfs_dw_dg2", "det_id")}
        e = np.column_stack([data["ml_e1"], data["ml_e2"]])
        response = np.stack([data[k] for k in RESPONSE_KEYS], axis=1).reshape(-1, 2, 2)
        w = data["fpfs_w"]
        dw = np.column_stack([data["fpfs_dw_dg1"], data["fpfs_dw_dg2"]])
        x, rx = weighted_inputs(e, response, w, dw)
        valid = (np.isfinite(e).all(1) & np.isfinite(response).all((1, 2)) &
                 np.isfinite(w) & (w >= 0) & np.isfinite(dw).all(1) &
                 np.isfinite(x).all(1) & np.isfinite(rx).all((1, 2)))
        compact = dict(shapes=x.astype(np.float32), R_ana=rx[:, None].astype(np.float32),
                       raw_e=e.astype(np.float32), raw_R=response.astype(np.float32),
                       exposure_id=np.full(len(e), exposure, dtype=np.int32),
                       det_id=data["det_id"].astype(np.int32),
                       split=np.full(len(e), labels[exposure], dtype=np.uint8))
        parts.append({k: v[valid] for k, v in compact.items()})
        audit["rows"] += len(e)
        audit["rejected"] += int((~valid).sum())
    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    out = step_dir(cfg, "catalog")
    np.savez(out / "split_exposures.npz", **splits)
    temporary = out / "catalog.tmp.npz"
    np.savez_compressed(temporary, **merged)
    with np.load(temporary) as z:
        if len(z["det_id"]) != audit["rows"] - audit["rejected"]:
            raise ValueError("catalog row-count verification failed")
    temporary.replace(out / "catalog.npz")
    audit["splits"] = {}
    for code, (name, ids) in enumerate(splits.items()):
        mask = merged["split"] == code
        if not mask.any():
            raise ValueError(f"empty {name} sample")
        audit["splits"][name] = {"exposures": len(ids), "rows": int(mask.sum())}
    audit["split_sha256"] = hashlib.sha256((out / "split_exposures.npz").read_bytes()).hexdigest()
    (out / "catalog_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    if b.get("discard_measurements", False):
        # This directory belongs only to this round; the verified compact catalog
        # now contains every selected full response, so the batch copies are redundant.
        for path in measurement_dir(cfg).glob("catalog_*.npz"):
            path.unlink()
        (measurement_dir(cfg) / "measurement_manifest.csv").unlink(missing_ok=True)


def effective_noise(stats, scale):
    from scripts.training_size_test.global_response_tract_noise import estimate
    return estimate(stats[:, :, 1], stats[:, :, 3], stats[:, :, 0], scale)[0]


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(cfg):
    from src.blending.test_func import shape_noise_from_stats, bootstrap_shape_noise

    with np.load(step_dir(cfg, "catalog") / "catalog.npz") as source, np.load(step_dir(cfg, "rdsm") / "rdsm_catalog.npz") as z:
        if "det_id" in z.files:
            np.testing.assert_array_equal(source["det_id"], z["det_id"])
            np.testing.assert_array_equal(source["exposure_id"], z["exposure_id"])
        labels = source["split"]
        test = (labels == (2 if labels.dtype.kind in "iu" else "test")) & z["rdsm_valid"].astype(bool)
        ids = source["exposure_id"][test]
        signals = np.stack([source["raw_e"][test], source["shapes"][test],
                            np.column_stack([z["rdsm_e1"][test], z["rdsm_e2"][test]])], axis=1)
        response = np.stack([source["raw_R"][test], source["R_ana"][test, 0], z["rdsm_response"][test]], axis=1)
    finite = np.isfinite(signals).all((1, 2)) & np.isfinite(response).all((1, 2, 3))
    ids, signals, response = ids[finite], signals[finite], response[finite]
    exposures, inv = np.unique(ids, return_inverse=True)
    if len(exposures) < 2:
        raise ValueError("at least two nonempty test exposures are required")
    counts = np.bincount(inv)
    stats = np.empty((len(exposures), 3, 4, 2))
    cross = np.empty((len(exposures), 3, 2))
    for m in range(3):
        for c in range(2):
            e, r = signals[:, m, c], response[:, m, c, c]
            stats[:, m, 0, c] = counts
            stats[:, m, 1, c] = np.bincount(inv, weights=e)
            stats[:, m, 2, c] = np.bincount(inv, weights=e * e)
            stats[:, m, 3, c] = np.bincount(inv, weights=r)
            cross[:, m, c] = np.bincount(inv, weights=response[:, m, c, 1-c])
    total = stats.sum(0)
    _, mean_r, noise = shape_noise_from_stats(total)
    n_boot = 2000
    seed = cfg["blended"]["seed"]
    noise_draws = bootstrap_shape_noise(stats, n_boot=n_boot, seed=seed)
    scale = float(np.sqrt(counts.mean()))
    effective = effective_noise(stats, scale)
    effective_draws = np.empty((n_boot, 3, 2))
    cross_draws = np.empty((n_boot, 3, 2))
    rng = np.random.default_rng(seed)
    for i in range(n_boot):
        draw = rng.integers(len(exposures), size=len(exposures))
        effective_draws[i] = effective_noise(stats[draw], scale)
        cross_draws[i] = cross[draw].sum(0) / counts[draw].sum()
    rows, zero = [], []
    names = ("attnw", "detw_attnw", "rdsm")
    for m, name in enumerate(names):
        for c in range(2):
            ratio_draws = noise_draws[:, m, c] / noise_draws[:, 1, c]
            effective_ratio_draws = np.divide(
                effective_draws[:, m, c], effective_draws[:, 1, c],
                out=np.full(n_boot, np.nan), where=effective_draws[:, 1, c] != 0)
            rows.append(dict(method=name, component=c+1, n_objects=len(ids), n_exposures=len(exposures),
                mean_response=mean_r[m, c], shear=total[m, 1, c] / total[m, 3, c],
                shape_noise=noise[m, c], shape_noise_err=noise_draws[:, m, c].std(ddof=1),
                noise_ratio_to_detw=noise[m, c] / noise[1, c],
                ratio_ci_low=np.quantile(ratio_draws, .025), ratio_ci_high=np.quantile(ratio_draws, .975),
                effective_noise=effective[m, c], effective_noise_err=effective_draws[:, m, c].std(ddof=1),
                effective_ratio_to_detw=effective[m, c]/effective[1, c],
                effective_ratio_ci_low=np.nanquantile(effective_ratio_draws, .025),
                effective_ratio_ci_high=np.nanquantile(effective_ratio_draws, .975),
                effective_ratio_valid_draws=int(np.isfinite(effective_ratio_draws).sum())))
            r = response[:, m, c, 1-c]
            zero.append(dict(method=name, response=f"R{c+1}{2-c}", mean=r.mean(),
                rms=np.sqrt(np.mean(r*r)), abs_p50=np.quantile(np.abs(r), .5),
                abs_p95=np.quantile(np.abs(r), .95), abs_p99=np.quantile(np.abs(r), .99),
                mean_err=cross_draws[:, m, c].std(ddof=1),
                mean_ci_low=np.quantile(cross_draws[:, m, c], .025),
                mean_ci_high=np.quantile(cross_draws[:, m, c], .975),
                mean_over_diagonal=r.mean() / mean_r[m, c],
                rms_over_diagonal=np.sqrt(np.mean(r*r)) / abs(mean_r[m, c])))
    out = step_dir(cfg, "evaluate")
    write_csv(out / "metrics.csv", rows)
    write_csv(out / "offdiagonal_zero.csv", zero)
    np.savez(out / "exposure_statistics.npz", exposure_id=exposures, methods=names,
             statistics=stats, cross_response_sum=cross)
    (out / "report.txt").write_text(
        f"Test only: {len(exposures)} exposures, {len(ids)} common detections.\n"
        "Shear uses sum(signal)/sum(diagonal response); no matrix inverse.\n"
        "detw_attnw and rdsm signals already include detection weighting; no extra w.\n"
        "Off-diagonal diagnostics use individual values and exposure-bootstrap means.\n"
        "Effective-noise bootstrap keeps sqrt(mean count) fixed, as in the existing reports.\n"
        "2000 paired exposure bootstrap draws; intervals are 95% percentile intervals.\n"
        "No end-to-end image-shear bias measurement is made here.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("measure", "catalog", "evaluate", "checkpoint-norm"))
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.action == "checkpoint-norm":
        import torch
        path = step_dir(cfg, "rdsm") / "selection.json"
        selection = json.loads(path.read_text())
        checkpoint = torch.load(selection["checkpoint"], map_location="cpu", weights_only=False)
        if checkpoint["evaluation_split"] != "val":
            raise ValueError("blended checkpoint was not selected on validation data")
        norm = float(checkpoint["input_norm"])
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("invalid training norm")
        selection.update(input_norm=norm, evaluation_split="val", input_transform="detw_e")
        path.write_text(json.dumps(selection, indent=2) + "\n")
        print(norm)
    else:
        {"measure": measure, "catalog": catalog, "evaluate": evaluate}[args.action](cfg)


if __name__ == "__main__":
    main()
