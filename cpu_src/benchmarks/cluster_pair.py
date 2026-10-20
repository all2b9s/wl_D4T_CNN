"""Paired real-stamp CPU/GPU validation. Run only inside a Slurm allocation.

CPU process pools run before the first CUDA initialization. Every timing excludes
loading, export, q-stamp generation and plotting. Raw repetitions are retained.
"""
from __future__ import annotations
import argparse
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

_WORKER_MODEL = None
_WORKER_Q = None
_WORKER_PATH = None


def event(message, **values):
    print(json.dumps({"event": message, **values}), flush=True)


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def result_arrays(result):
    return {"e": result.e, "R": result.R,
            **({"dE_dx": result.dE_dx} if result.dE_dx is not None else {})}


def timed(fn, count, repeats, synchronize=None, warmup=None):
    warmup = fn if warmup is None else warmup
    warmup()
    warmup()
    samples = []
    for _ in range(repeats):
        if synchronize:
            synchronize()
        start = time.perf_counter()
        result = fn()
        if synchronize:
            synchronize()
        samples.append(time.perf_counter() - start)
    median = float(np.median(samples))
    return {"count": count, "seconds": samples, "median_seconds": median,
            "gal_per_sec": count / median, "unstable": max(samples) / min(samples) > 1.2}, result


def init_worker(counter, cores):
    import torch
    with counter.get_lock():
        index = counter.value
        counter.value += 1
    os.sched_setaffinity(0, {cores[index % len(cores)]})
    torch.set_num_threads(1)


def worker_measure(indices):
    from cpu_src.reference import measure
    result = measure(_WORKER_PATH, _WORKER_Q[indices], "float32", 2,
                     batch_size=1, _model=_WORKER_MODEL)
    return np.asarray(indices), result.e, result.R


def torch_processes(model, model_path, q, processes, repeats, cores):
    import torch
    global _WORKER_MODEL, _WORKER_Q, _WORKER_PATH
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU fork baseline must precede CUDA initialization")
    _WORKER_MODEL, _WORKER_Q, _WORKER_PATH = model, q, str(model_path)
    ctx = mp.get_context("fork")
    counter = ctx.Value("i", 0)
    count = min(len(q), 128) if processes == 1 else len(q)
    indices = np.array_split(np.arange(count), processes)
    started = time.perf_counter()
    with ctx.Pool(processes, initializer=init_worker, initargs=(counter, cores)) as pool:
        def run():
            pieces = pool.map(worker_measure, indices, chunksize=1)
            e, r = np.empty((count, 2), np.float32), np.empty((count, 2, 2), np.float32)
            for ids, ee, rr in pieces:
                e[ids], r[ids] = ee, rr
            return {"e": e, "R": r}
        warm_start = time.perf_counter()
        run()
        startup_warmup = time.perf_counter() - started
        timing, arrays = timed(run, count, repeats)
    timing.update({"processes": processes, "threads_per_process": 1,
                   "startup_and_first_warmup_seconds": startup_warmup,
                   "includes": "pool task dispatch, source inference, output assembly; excludes pool startup"})
    _WORKER_MODEL = None
    _WORKER_Q = None
    _WORKER_PATH = None
    return timing, arrays


def export_models(repo, work, output):
    from cpu_src.export import export_checkpoint
    paths, metadata = {}, {}
    source = repo / "models/blended_rdsm_runs/renoised_q_n0594_330k_3m"
    for label in ("s4", "attnw"):
        checkpoint = source / label / "last.pth"
        destination = work / f"{label}.npz"
        metadata[label] = export_checkpoint(checkpoint, destination)
        paths[label] = destination
        shutil.copy2(destination, output / destination.name)
        event("export", model=label, checkpoint=str(checkpoint), kwargs=metadata[label]["model"])
    return paths, metadata


def prepare_data(repo, work, output, model_path, metadata, count, seed):
    from cpu_src import Model
    model = Model.load(model_path)
    source = repo / "models/blended_rdsm_runs/renoised_q_n0594_330k_3m"
    splits = []
    for name in ("s4", "attnw"):
        with np.load(source / name / "split_exposures.npz", allow_pickle=False) as data:
            key = next(k for k in data.files if k in ("test", "test_exposures", "test_exp"))
            splits.append(set(map(int, data[key])))
    exposures = sorted(splits[0] & splits[1])[:4]
    if len(exposures) < 1:
        raise ValueError("No common held-out exposures")
    pre = metadata["preprocessing"]
    psf = np.asarray(np.load(repo / "logs/blended_test_psf.npy", allow_pickle=False), dtype=np.float64)
    if psf.ndim == 3:
        psf = psf[0]
    if psf.shape[0] % 2 or psf.shape[1] % 2:
        psf = psf[:-1, :-1]
    rng = np.random.default_rng(seed)
    raw_images, cached_q0, ids, exposure_ids, raw_metadata = [], [], [], [], []
    per_exposure = (count + len(exposures) - 1) // len(exposures)
    for exposure in exposures:
        name = f"blended_{exposure:05d}.npz"
        source_raw = Path("/work/hdd/bfcn/wenyinli/blended_training_stamp") / name
        source_q = Path("/work/hdd/bfcn/wenyinli/blended_training_q_n0594") / name
        # Stage entire shards outside timing, then do all further reads locally.
        local_raw, local_q = work / ("raw_" + name), work / ("q0_" + name)
        shutil.copy2(source_raw, local_raw)
        shutil.copy2(source_q, local_q)
        with np.load(local_raw, allow_pickle=False) as raw, np.load(local_q, allow_pickle=False) as cached:
            if not np.array_equal(raw["sample_id"], cached["sample_id"]):
                raise ValueError("Raw/cached shard sample IDs differ")
            chosen = np.sort(rng.choice(len(raw["sample_id"]), min(per_exposure, len(raw["sample_id"])), replace=False))
            raw_images.append(np.array(raw["blended_stamps"][chosen], dtype=np.float32))
            cached_q0.append(np.array(cached["q_images"][chosen], dtype=np.float32))
            ids.append(np.array(raw["sample_id"][chosen], dtype=np.int64))
            exposure_ids.append(np.full(len(chosen), exposure, dtype=np.int64))
            raw_metadata.append({"exposure": exposure, "source": str(source_raw), "selected": chosen.tolist()})
    images = np.concatenate(raw_images)[:count]
    cached = np.concatenate(cached_q0)[:count]
    sample_ids = np.concatenate(ids)[:count]
    exp_ids = np.concatenate(exposure_ids)[:count]
    if len(images) != count:
        raise ValueError(f"Requested {count}, obtained {len(images)} real stamps")
    noise_seed = metadata["config"]["image_preprocessing"]["seed"]
    noise_std = metadata["config"]["image_preprocessing"]["noise_std"]
    center = np.array([[images.shape[-1] // 2, images.shape[-2] // 2]], dtype=np.float64)
    def one(index):
        generator = np.random.default_rng(np.random.SeedSequence([noise_seed, int(sample_ids[index])]))
        noise = generator.normal(0, noise_std, images[index].shape).astype(np.float32)
        return model.prepare_qstamps(images[index], psf, center, noise, nthreads=1)[0].astype(np.float32)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=16) as executor:
        q = np.stack(list(executor.map(one, range(count))))
    elapsed = time.perf_counter() - started
    delta = np.abs(q[:, 0] - cached)
    matched = bool(np.allclose(q[:, 0], cached, atol=1e-6, rtol=1e-5))
    if not matched:
        raise ValueError(f"Reconstructed primal differs from training cache: max_abs={delta.max()}")
    provenance = {"kind": "real held-out blended stamps", "samples": count,
                  "exposures": exposures, "sampling_seed": seed, "sample_id": sample_ids.tolist(),
                  "shards": raw_metadata, "preprocessing": pre, "noise_seed": noise_seed,
                  "noise_std": noise_std, "cached_q0_matches": matched,
                  "cached_q0_max_abs": float(delta.max()), "q_preparation_16_threads_seconds": elapsed}
    np.savez(output / "qstamps.npz", q=q, sample_id=sample_ids, exposure_id=exp_ids)
    save_json(output / "data.json", provenance)
    event("data_ready", samples=count, exposures=exposures, q0_max_abs=float(delta.max()), preparation_seconds=elapsed)
    return np.ascontiguousarray(q), provenance


def compare(actual, reference):
    report = {}
    for name in ("e", "R", "dE_dx"):
        if name not in actual or name not in reference:
            continue
        a, b = actual[name].astype(np.float64), reference[name].astype(np.float64)
        rtol = 1e-5 if name == "e" else 1e-4
        error = a - b
        scale = 1e-6 + rtol * np.abs(b)
        report[name] = {"max_abs": float(np.max(np.abs(error))), "rms": float(np.sqrt(np.mean(error**2))),
                        "mean_signed": float(error.mean()), "p99_abs": float(np.quantile(np.abs(error), .99)),
                        "max_normalized_error": float(np.max(np.abs(error) / scale)),
                        "within_tolerance_fraction": float(np.mean(np.abs(error) <= scale)),
                        "allclose": bool(np.all(np.abs(error) <= scale)), "atol": 1e-6, "rtol": rtol}
        if name in ("e", "R"):
            bias = error.mean(axis=0)
            report[name].update({"mean_bias": bias.tolist(), "max_abs_mean_bias": float(np.max(np.abs(bias))),
                                 "mean_bias_limit": 1e-4,
                                 "mean_bias_passed": bool(np.isfinite(error).all() and np.all(np.abs(bias) < 1e-4))})
    return report


def cpu_phase(paths, q, output, repeats, cores, report):
    import torch
    from cpu_src import Model
    from cpu_src.reference import torch_model
    torch.set_num_threads(1)
    results = {}
    for name, path in paths.items():
        model = Model.load(path)
        reference = torch_model(path)
        rows = []
        for workers in (1, 8, 36, 72):
            if workers > len(cores):
                continue
            count = min(128, len(q)) if workers == 1 else len(q)
            os.sched_setaffinity(0, set(cores[:workers]))
            run = lambda: model.measure_qstamps(q[:count], 2, workers)
            warm = lambda: model.measure_qstamps(q[:min(count, max(32, 2 * workers))], 2, workers)
            row, value = timed(run, count, repeats, warmup=warm)
            row["threads"] = workers
            rows.append(row)
            event("cpu_cpp", model=name, **row)
            if workers == max(x for x in (1, 8, 36, 72) if x <= len(cores)):
                results[name] = {"cpp": result_arrays(value)}
        os.sched_setaffinity(0, set(cores))
        baseline = rows[0]["gal_per_sec"]
        for row in rows:
            row["parallel_efficiency"] = row["gal_per_sec"] / (row["threads"] * baseline)
        process_rows = []
        for processes in (1, 8, 72):
            if processes > len(cores):
                continue
            gc.collect()
            row, value = torch_processes(reference, path, q, processes, repeats, cores)
            process_rows.append(row)
            event("cpu_torch_processes", model=name, **row)
            if processes == max(x for x in (1, 8, 72) if x <= len(cores)):
                results[name]["torch_cpu"] = value
        results[name]["cpu_comparison"] = compare(results[name]["cpp"], results[name]["torch_cpu"])
        np.savez(output / f"{name}_cpu.npz", **{f"{backend}_{key}": array for backend in ("cpp", "torch_cpu")
                                                for key, array in results[name][backend].items()})
        report["models"][name] = {"cpu_cpp": rows, "cpu_torch_processes": process_rows,
                                  "cpp_vs_torch_cpu": results[name]["cpu_comparison"]}
        save_json(output / "progress.json", report)
        del reference, model
        gc.collect()
    return results


def gpu_device_measure(model, q, batch_size):
    import torch
    es, rs = [], []
    for start in range(0, len(q), batch_size):
        batch = q[start:start + batch_size]
        x = batch[:, :1].detach().requires_grad_(True)
        pred = model(x)
        gradients = []
        for component in range(2):
            gx, = torch.autograd.grad(pred[:, component].sum(), x, retain_graph=component == 0)
            gradients.append((gx * batch[:, 1:3]).sum((-1, -2)))
        es.append(pred.detach())
        rs.append(torch.stack(gradients, 1).detach())
    return torch.cat(es), torch.cat(rs)


def gpu_phase(paths, q, output, repeats, cores, report, results):
    import torch
    from cpu_src import Model
    from cpu_src.reference import torch_model, measure
    os.sched_setaffinity(0, set(cores))
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("Allocated GPU is unavailable")
    report["environment"]["gpu"] = torch.cuda.get_device_name(0)
    report["environment"]["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
    original = {"matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_tf32": torch.backends.cudnn.allow_tf32, "cudnn_benchmark": torch.backends.cudnn.benchmark}
    report["environment"]["original_cuda_flags"] = original
    torch.backends.cudnn.benchmark = False
    for name, path in paths.items():
        model = torch_model(path, device="cuda")
        mode_rows = {}
        for mode in ("strict_fp32", "production_default"):
            torch.backends.cuda.matmul.allow_tf32 = False if mode == "strict_fp32" else original["matmul_tf32"]
            torch.backends.cudnn.allow_tf32 = False if mode == "strict_fp32" else original["cudnn_tf32"]
            rows = []
            for batch_size in (16, 64, 128, 256):
                try:
                    torch.cuda.reset_peak_memory_stats()
                    run = lambda: measure(path, q, "float32", 2, "cuda", batch_size, _model=model)
                    warm = lambda: measure(path, q[:min(len(q), 2 * batch_size)], "float32", 2, "cuda", batch_size, _model=model)
                    row, value = timed(run, len(q), repeats, torch.cuda.synchronize, warm)
                    row.update({"batch_size": batch_size, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                                "cudnn_tf32": torch.backends.cudnn.allow_tf32})
                    rows.append(row)
                    event("gpu", model=name, mode=mode, **row)
                except torch.cuda.OutOfMemoryError:
                    model.feats = model.f8 = model.f_mean_e1 = model.f_mean_e2 = None
                    torch.cuda.empty_cache()
                    rows.append({"batch_size": batch_size, "oom": True})
                    event("gpu_oom", model=name, mode=mode, batch_size=batch_size)
            valid = [r for r in rows if not r.get("oom")]
            if not valid:
                raise RuntimeError("No successful GPU batch")
            best = max(valid, key=lambda r: r["gal_per_sec"])
            value = measure(path, q, "float32", 2, "cuda", best["batch_size"], _model=model)
            arrays = result_arrays(value)
            results[name][mode] = arrays
            device_q = torch.as_tensor(q, device="cuda", dtype=torch.float32)
            gpu_device_measure(model, device_q[:best["batch_size"]], best["batch_size"])
            torch.cuda.synchronize()
            device_times = []
            for _ in range(repeats):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                device_result = gpu_device_measure(model, device_q, best["batch_size"])
                b.record()
                b.synchronize()
                device_times.append(a.elapsed_time(b) / 1000.)
            resident_comparison = compare({"e": device_result[0].cpu().numpy(),
                                          "R": device_result[1].cpu().numpy()}, arrays)
            mode_rows[mode] = {"batches": rows, "best": best,
                               "device_resident_seconds": device_times,
                               "device_resident_gal_per_sec": len(q) / float(np.median(device_times)),
                               "device_resident_vs_host_outputs": resident_comparison}
            comparison = compare(results[name]["cpp"], arrays)
            report["models"][name][f"cpp_vs_gpu_{mode}"] = comparison
            event("comparison", model=name, mode=mode, errors=comparison)
            del device_q, device_result
        report["models"][name]["gpu"] = mode_rows
        # Position response is a separate correctness check, not the T=2 throughput definition.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        q4 = q[:min(64, len(q))]
        cpu4 = Model.load(path).measure_qstamps(q4, 4, min(72, len(cores)))
        gpu4 = measure(path, q4, "float32", 4, "cuda", 64, _model=model)
        report["models"][name]["T4_comparison"] = compare(result_arrays(cpu4), result_arrays(gpu4))
        np.savez(output / f"{name}_paired.npz", **{f"{backend}_{key}": array for backend in
                  ("cpp", "torch_cpu", "strict_fp32", "production_default") for key, array in results[name][backend].items()},
                  cpp_T4_dE_dx=cpu4.dE_dx, gpu_T4_dE_dx=gpu4.dE_dx)
        save_json(output / "progress.json", report)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = original["matmul_tf32"]
    torch.backends.cudnn.allow_tf32 = original["cudnn_tf32"]
    torch.backends.cudnn.benchmark = original["cudnn_benchmark"]


def plots(output, report, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    labels = ["e1", "e2", "R11", "R12", "R21", "R22"]
    def columns(value):
        return np.column_stack((value["e"], value["R"].reshape(-1, 4)))
    for name, value in results.items():
        for mode in ("strict_fp32", "production_default"):
            x, y = columns(value[mode]), columns(value["cpp"])
            fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
            residual, residual_axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
            normalized_fig, normalized_axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
            for i, (ax, rx, nx) in enumerate(zip(axes.flat, residual_axes.flat, normalized_axes.flat)):
                low, high = min(x[:, i].min(), y[:, i].min()), max(x[:, i].max(), y[:, i].max())
                width = max(high - low, 1e-7)
                ax.scatter(x[:, i], y[:, i], s=10, alpha=.5, edgecolors="none")
                ax.plot([low, high], [low, high], "k--", lw=1)
                ax.set(xlabel="PyTorch GPU " + labels[i], ylabel="C++ CPU " + labels[i],
                       xlim=(low - .03 * width, high + .03 * width), ylim=(low - .03 * width, high + .03 * width))
                ax.set_aspect("equal", adjustable="box")
                difference = y[:, i].astype(float) - x[:, i].astype(float)
                ax.text(.03, .96, f"max |delta|={np.max(np.abs(difference)):.2e}\nRMS={np.sqrt(np.mean(difference**2)):.2e}",
                        transform=ax.transAxes, va="top", fontsize=9)
                rx.scatter(x[:, i], difference, s=10, alpha=.5, edgecolors="none")
                rx.axhline(0, color="black", lw=.8)
                tolerance = 1e-6 + (1e-5 if i < 2 else 1e-4) * np.abs(x[:, i])
                failed = np.abs(difference) > tolerance
                ax.scatter(x[failed, i], y[failed, i], s=22, facecolors="none", edgecolors="tab:red")
                rx.scatter(x[failed, i], difference[failed], s=22, facecolors="none", edgecolors="tab:red")
                nx.scatter(x[:, i], difference / tolerance, s=10, alpha=.5, edgecolors="none")
                nx.scatter(x[failed, i], (difference / tolerance)[failed], s=22, color="tab:red")
                nx.axhline(0, color="black", lw=.8)
                nx.axhline(1, color="tab:red", lw=.8)
                nx.axhline(-1, color="tab:red", lw=.8)
                nx.set(xlabel="PyTorch GPU " + labels[i], ylabel="(CPU - GPU) / tolerance",
                       title=f"{int(failed.sum())}/{len(x)} outside tolerance")
                order = np.argsort(x[:, i])
                rx.plot(x[order, i], tolerance[order], color="tab:red", lw=.8)
                rx.plot(x[order, i], -tolerance[order], color="tab:red", lw=.8)
                rx.set(xlabel="PyTorch GPU " + labels[i], ylabel="CPU - GPU")
                rx.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            fig.suptitle(f"{name}: {len(x)} real held-out stamps, fp32, T=2, {mode}")
            residual.suptitle(f"{name}: residuals; red = output acceptance tolerance ({mode})")
            normalized_fig.suptitle(f"{name}: normalized residuals; acceptance = [-1, 1] ({mode})")
            for extension in ("png", "pdf"):
                fig.savefig(figure_dir / f"{name}_{mode}_one_to_one.{extension}", dpi=180)
                residual.savefig(figure_dir / f"{name}_{mode}_residuals.{extension}", dpi=180)
                normalized_fig.savefig(figure_dir / f"{name}_{mode}_normalized_residuals.{extension}", dpi=180)
            plt.close(fig)
            plt.close(residual)
            plt.close(normalized_fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, (name, item) in zip(axes, report["models"].items()):
        for key, label, worker_key in (("cpu_cpp", "C++ dual CPU", "threads"),
                                       ("cpu_torch_processes", "PyTorch CPU autograd", "processes")):
            rows = item[key]
            ax.plot([r[worker_key] for r in rows], [r["gal_per_sec"] for r in rows], "o-", label=label)
        for mode, style in (("strict_fp32", "--"), ("production_default", ":")):
            speed = item["gpu"][mode]["best"]["gal_per_sec"]
            ax.axhline(speed, linestyle=style, label="1 GPU " + mode)
        ax.errorbar([72], [540], yerr=[40], fmt="*", color="green", markersize=12, capsize=4, label="plan: 500-580 at 72 cores")
        ax.set(title=name, xlabel="CPU cores / single-thread CPU processes", ylabel="galaxies/s (e + full R)")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.savefig(figure_dir / "throughput.png", dpi=180)
    fig.savefig(figure_dir / "throughput.pdf")
    plt.close(fig)


def write_report(output, report):
    lines = ["# GH200 paired CPU/GPU validation", "", f"Slurm job: {report['environment']['job_id']}; node: {report['environment']['hostname']}.",
             "", f"Input: {report['data']['samples']} real blended stamps from common held-out exposures {report['data']['exposures']}; model computations use fp32.",
             "", "Both models use last.pth from renoised_q_n0594_330k_3m; s4 has nl_sigma=gw_sigma=4.",
             "", "## Throughput (T=2: e and all four shear responses)", "",
             "| Model | C++ 1 core | C++ 72 cores | 72-core efficiency | Torch CPU 72x1 | GPU strict fp32 | GPU production default |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, item in report["models"].items():
        one, many = item["cpu_cpp"][0], item["cpu_cpp"][-1]
        torch_cpu = item["cpu_torch_processes"][-1]
        strict, default = (item["gpu"][m]["best"] for m in ("strict_fp32", "production_default"))
        lines.append(f"| {name} | {one['gal_per_sec']:.1f} | {many['gal_per_sec']:.1f} | {many['parallel_efficiency']:.1%} | {torch_cpu['gal_per_sec']:.1f} | {strict['gal_per_sec']:.1f} | {default['gal_per_sec']:.1f} |")
        item["expectation_check"] = {"original_72_core_estimate": [500, 580],
            "meets_500_gal_per_sec": many["gal_per_sec"] >= 500,
            "meets_70_percent_parallel_efficiency": many["parallel_efficiency"] >= .7,
            "speedup_over_same_node_torch_cpu_72x1": many["gal_per_sec"] / torch_cpu["gal_per_sec"],
            "fraction_of_strict_gpu_throughput": many["gal_per_sec"] / strict["gal_per_sec"]}
    lines += ["", "## Float32 acceptance", "",
              "Each e/R component must satisfy abs(mean(CPU - reference)) < 1e-4 over matched sources. Components are assessed separately; this is signed mean bias, not mean absolute error.", "",
              "| Model / GPU mode | Largest e component mean bias (absolute) | Largest R component mean bias (absolute) | Pass |",
              "|---|---:|---:|---|"]
    for name, item in report["models"].items():
        for mode in ("strict_fp32", "production_default"):
            result = item[f"cpp_vs_gpu_{mode}"]
            if all("mean_bias_passed" in result[key] for key in ("e", "R")):
                passed = all(result[key]["mean_bias_passed"] for key in ("e", "R"))
                lines.append(f"| {name} / {mode} | {result['e']['max_abs_mean_bias']:.3g} | {result['R']['max_abs_mean_bias']:.3g} | {passed} |")
            else:
                lines.append(f"| {name} / {mode} | unavailable in old report | unavailable | reassess saved predictions |")
    lines += ["", "## Pointwise diagnostics", "", "| Model / GPU mode | e max abs | R max abs | e pass fraction | R pass fraction |", "|---|---:|---:|---:|---:|"]
    for name, item in report["models"].items():
        for mode in ("strict_fp32", "production_default"):
            result = item[f"cpp_vs_gpu_{mode}"]
            lines.append(f"| {name} / {mode} | {result['e']['max_abs']:.3g} | {result['R']['max_abs']:.3g} | {result['e']['within_tolerance_fraction']:.3%} | {result['R']['within_tolerance_fraction']:.3%} |")
    lines += ["", "Diagnostic tolerances: e atol=1e-6/rtol=1e-5; responses atol=1e-6/rtol=1e-4. Individual outliers do not determine the float32 e/R acceptance above.",
              "", "## Measurement protocol and limits", "",
              "- All work ran sequentially within one interactive allocation: one GPU and 72 cores on its CPU domain. Other quarters of the physical node may be shared.",
              "- The same five-plane q-stamps feed every backend. Reconstructed primal images were checked against the training cache, using its per-sample noise seeds.",
              "- CPU: C++ workers over sources; Torch uses persistent, core-pinned processes with one intra-op thread each. Fork pools complete before CUDA initialization.",
              "- Host-to-output timings exclude export, file I/O, q preparation, pool creation, and plots; include input/output handling. GPU calls are synchronized. Device-resident event timings are additionally in results.json.",
              "- Each setting has two warmups and retains every measured repetition. One-core tests use up to 128 samples; parallel/GPU tests use the entire sample. Inspect unstable flags before interpreting a speedup.",
              "- GPU strict mode disables TF32 for both cuDNN and matmul. Production mode restores the original TF32 flags; both remain dtype=float32 and AMP is off.",
              "- The original 500-580 gal/s CPU figure was a forecast, not an acceptance measurement. Results are judged against it without changing the target.",
              "- This checks per-source numerical consistency and throughput, not real-exposure m/c science validation. Figures and raw predictions are retained.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    save_json(output / "results.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--resume-cpu", action="store_true",
                        help="Reuse saved q-stamps, model exports and CPU measurements; repeat GPU phase")
    parser.add_argument("--plots-only", action="store_true", help="Redraw saved paired predictions without rerunning inference")
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Run this workload inside a Slurm allocation, not a login node")
    args.output.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    if args.plots_only:
        report = json.loads((args.output / "results.json").read_text())
        results = {}
        for name in report["models"]:
            with np.load(args.output / f"{name}_paired.npz", allow_pickle=False) as data:
                results[name] = {backend: {key: np.array(data[f"{backend}_{key}"])
                                 for key in ("e", "R")} for backend in ("cpp", "strict_fp32", "production_default")}
        plots(args.output, report, results)
        event("plots_complete", output=str(args.output / "figures"))
        return
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cores = sorted(os.sched_getaffinity(0))
    if len(cores) < 72:
        raise RuntimeError(f"This validation requests 72 allocated cores, received {len(cores)}")
    report = {"environment": {"job_id": os.environ["SLURM_JOB_ID"], "hostname": platform.node(),
                "cpu_affinity": cores, "torch": torch.__version__, "numpy": np.__version__,
                "precision": "float32", "T": 2, "repeats": args.repeats,
                "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,memory.total", "--format=csv"], text=True)},
              "models": {}}
    event("started", **report["environment"])
    if args.resume_cpu:
        report = json.loads((args.output / "progress.json").read_text())
        if report["environment"]["job_id"] != os.environ["SLURM_JOB_ID"]:
            raise ValueError("Resume CPU measurements only within the same allocation")
        from cpu_src.model_io import load_model
        paths = {name: args.output / f"{name}.npz" for name in ("s4", "attnw")}
        with np.load(args.output / "qstamps.npz", allow_pickle=False) as data:
            q = np.ascontiguousarray(data["q"])
            if data["sample_id"].tolist() != report["data"]["sample_id"]:
                raise ValueError("Saved q-stamp sample IDs differ from the CPU run")
        results = {}
        for name, path in paths.items():
            if load_model(path)[0] != report["checkpoints"][name]:
                raise ValueError("Saved model metadata differs from the CPU run")
            with np.load(args.output / f"{name}_cpu.npz", allow_pickle=False) as data:
                results[name] = {backend: {key: np.array(data[f"{backend}_{key}"])
                                 for key in ("e", "R")} for backend in ("cpp", "torch_cpu")}
        report["environment"]["gpu_resumed_from_saved_cpu"] = True
        event("resumed_cpu", samples=len(q))
    else:
        paths, metadata = export_models(args.repo, args.work, args.output)
        if metadata["s4"]["preprocessing"] != metadata["attnw"]["preprocessing"]:
            raise ValueError("Paired models require the same q preprocessing")
        report["checkpoints"] = metadata
        q, provenance = prepare_data(args.repo, args.work, args.output, paths["s4"], metadata["s4"], args.samples, args.seed)
        report["data"] = provenance
        save_json(args.output / "progress.json", report)
        results = cpu_phase(paths, q, args.output, args.repeats, cores, report)
    gpu_phase(paths, q, args.output, args.repeats, cores, report, results)
    plots(args.output, report, results)
    write_report(args.output, report)
    event("complete", report=str(args.output / "REPORT.md"))


if __name__ == "__main__":
    main()
