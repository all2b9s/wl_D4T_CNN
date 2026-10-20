"""Diagnose real-stamp outliers against fp64, after cluster_pair completes."""
import argparse
import gc
import json
import os
from pathlib import Path
import numpy as np

from .cluster_pair import compare, result_arrays, save_json, event


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Run on an allocated compute node")
    import torch
    from cpu_src import Model
    from cpu_src.reference import torch_model, measure
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = args.output
    report = json.loads((output / "results.json").read_text())
    with np.load(output / "qstamps.npz", allow_pickle=False) as data:
        q, ids = data["q"], data["sample_id"]
    diagnosis = {}
    for name in ("s4", "attnw"):
        path = output / f"{name}.npz"
        with np.load(output / f"{name}_paired.npz", allow_pickle=False) as data:
            saved = {backend: {key: np.array(data[f"{backend}_{key}"]) for key in ("e", "R")}
                     for backend in ("cpp", "torch_cpu", "strict_fp32", "production_default")}
        component_rows, scores = {}, []
        for key in ("e", "R"):
            actual, ref = saved["cpp"][key].astype(float), saved["strict_fp32"][key].astype(float)
            difference = actual - ref
            threshold = 1e-6 + (1e-5 if key == "e" else 1e-4) * np.abs(ref)
            normalized = np.abs(difference) / threshold
            scores.append(normalized.reshape(len(q), -1).max(1))
            violations = []
            for index in np.argwhere(normalized > 1):
                index = tuple(map(int, index))
                violations.append({"index": index, "sample_id": int(ids[index[0]]),
                                   "cpu": float(actual[index]), "gpu": float(ref[index]),
                                   "difference": float(difference[index]), "threshold": float(threshold[index])})
            component_rows[key] = {"violations": violations,
                                  "component_mean_difference": difference.mean(0).tolist(),
                                  "component_rms": np.sqrt(np.mean(difference**2, 0)).tolist()}
        selected = np.argsort(np.maximum.reduce(scores))[-8:]
        subset = q[selected]
        cpu64 = result_arrays(Model.load(path, precision="float64").measure_qstamps(subset, 4, 8))
        gpu64_model = torch_model(path, "float64", "cuda")
        gpu64 = result_arrays(measure(path, subset, "float64", 4, "cuda", 8, _model=gpu64_model))
        precision64 = {}
        for key in cpu64:
            delta = cpu64[key] - gpu64[key]
            precision64[key] = {"max_abs": float(np.abs(delta).max()),
                               "allclose_1e_12": bool(np.allclose(cpu64[key], gpu64[key], atol=1e-12, rtol=1e-12))}
        del gpu64_model
        gc.collect()
        torch.cuda.empty_cache()
        model = torch_model(path, "float32", "cuda")
        repeated = {}
        for mode in ("strict_fp32", "production_default"):
            torch.backends.cudnn.allow_tf32 = mode == "production_default"
            batch_size = report["models"][name]["gpu"][mode]["best"]["batch_size"]
            values = [result_arrays(measure(path, q, "float32", 2, "cuda", batch_size, _model=model)) for _ in range(2)]
            repeated[mode] = {"repeat_vs_repeat": compare(values[0], values[1]),
                              "repeat_vs_saved": compare(values[0], saved[mode])}
        torch.backends.cudnn.allow_tf32 = False
        diagnosis[name] = {"strict_fp32_components": component_rows,
                           "selected_indices": selected.tolist(), "selected_sample_ids": ids[selected].tolist(),
                           "fp64_cpu_vs_gpu_T4": precision64,
                           "fp32_cpu_vs_fp64": compare({key: value[selected] for key, value in saved["cpp"].items()}, gpu64),
                           "fp32_gpu_vs_fp64": compare({key: value[selected] for key, value in saved["strict_fp32"].items()}, gpu64),
                           "gpu_repeats": repeated}
        np.savez(output / f"{name}_fp64_diagnosis.npz", indices=selected, sample_id=ids[selected],
                 **{f"cpu64_{key}": value for key, value in cpu64.items()},
                 **{f"gpu64_{key}": value for key, value in gpu64.items()})
        save_json(output / "diagnosis.json", diagnosis)
        event("diagnosed", model=name, fp64=precision64, violations=component_rows)
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
