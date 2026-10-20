import json
from pathlib import Path
import numpy as np
from .api import Model
from .model_io import sha256

def error_stats(actual, expected, atol, rtol, mean_bias_limit=None):
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape:
        raise ValueError("Paired outputs must have identical shapes")
    delta = actual - expected
    difference = np.abs(delta)
    pointwise = bool(np.isfinite(delta).all() and np.allclose(actual,expected,atol=atol,rtol=rtol))
    report = {"passed": pointwise, "pointwise_passed": pointwise,
            "max_abs": float(difference.max(initial=0)),
            "rms": float(np.sqrt(np.mean(difference**2))) if difference.size else 0.,
            "p99_abs": float(np.quantile(difference,.99)) if difference.size else 0.,
            "atol": atol, "rtol": rtol}
    if mean_bias_limit is not None:
        if not np.isfinite(mean_bias_limit) or mean_bias_limit <= 0:
            raise ValueError("Mean bias limit must be finite and positive")
        bias = delta.mean(axis=0) if delta.size else None
        finite = bool(delta.size and np.isfinite(delta).all())
        maximum = float(np.max(np.abs(bias))) if finite else None
        passed = finite and maximum < mean_bias_limit
        report.update({"passed": passed, "mean_bias_passed": passed,
                       "mean_bias": bias.tolist() if finite else None,
                       "max_abs_mean_bias": maximum, "mean_bias_limit": mean_bias_limit,
                       "acceptance": "absolute signed mean difference per component, over sources"})
    return report

def validate_golden(model_path, golden_path, precision="float32", isa="auto", nthreads=1):
    model = Model.load(model_path,precision,isa)
    with np.load(golden_path,allow_pickle=False) as data:
        provenance = json.loads(data["metadata"].tobytes())
        if provenance["model_sha256"] != sha256(model_path):
            raise ValueError("Golden data belongs to a different exported model")
        result = model.measure_qstamps(data["q"],4,nthreads)
        if precision == "float64":
            atol, ertol, rrtol = 1e-12,1e-12,1e-12
        else:
            atol, ertol, rrtol = 1e-6,1e-5,1e-4
        report = {"precision": precision,"isa": model.isa,"golden": provenance,
                  "e": error_stats(result.e,data["e"],atol,ertol,1e-4 if precision=="float32" else None),
                  "R": error_stats(result.R,data["R"],atol,rrtol,1e-4 if precision=="float32" else None),
                  "dE_dx": error_stats(result.dE_dx,data["dE_dx"],atol,rrtol)}
        trace_path = Path(golden_path).with_name(Path(golden_path).stem+"_trace.npz")
        if trace_path.is_file():
            report["trace"] = {}
            with np.load(trace_path,allow_pickle=False) as trace:
                for i in range(min(4,len(data["q"]))):
                    native = model.trace(data["q"][i],4)
                    for key, value in native.items():
                        name = f"{i}/{key}"
                        if name in trace:
                            report["trace"][name] = error_stats(value[0],trace[name],atol if precision == "float64" else 1e-5,rrtol)
        report["acceptance"] = "per-component mean bias < 1e-4 for e and R" if precision=="float32" else "pointwise float64 validation"
        report["passed"] = all(report[k]["passed"] for k in (("e","R") if precision=="float32" else ("e","R","dE_dx")))
        report["trace_passed"] = all(x["passed"] for x in report.get("trace",{}).values())
    return report

def paired_catalogs(left,right):
    """Compare identical catalog samples before any calibration comparison."""
    def read(path):
        with np.load(path,allow_pickle=False) as data:
            return {key: np.array(data[key]) for key in data.files}
    a,b=read(left),read(right)
    if len(a["det_id"])!=len(b["det_id"]) or any(not np.array_equal(a[k],b[k]) for k in ("det_id","x","y")):
        raise ValueError("Catalog detection rows differ; this is not a paired comparison")
    keys=["ml_e1","ml_e2","ml_R11","ml_R12","ml_R21","ml_R22"]
    return {k:error_stats(a[k],b[k],1e-6,1e-4 if "R" in k else 1e-5,mean_bias_limit=1e-4) for k in keys}
