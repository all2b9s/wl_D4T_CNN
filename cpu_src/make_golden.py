import json
from pathlib import Path
import numpy as np
from .model_io import load_model, sha256

def make_golden(model_path, output, qstamps=None, count=512, precision="float64", seed=42, device="cpu"):
    from .reference import measure, synthetic_q, primal_trace
    if type(count) is not int or count < 1:
        raise ValueError("count must be positive")
    meta, _ = load_model(model_path)
    if qstamps:
        with np.load(qstamps, allow_pickle=False) as archive:
            q = np.array(archive["q"][:count], dtype=precision)
        origin = str(Path(qstamps).resolve())
    else:
        q = synthetic_q(count, meta["stamp_size"], seed).astype(precision)
        origin = "synthetic (numerical validation only; not a science sample)"
    reference = measure(model_path, q, precision, 4, device)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance = {"origin": origin, "model_sha256": sha256(model_path), "precision": precision, "seed": seed}
    np.savez(output, q=q, e=reference.e, R=reference.R, dE_dx=reference.dE_dx,
             metadata=np.frombuffer(json.dumps(provenance).encode(),dtype=np.uint8))
    traces = {}
    for i in range(min(4,len(q))):
        for key, value in primal_trace(model_path,q[i],precision).items():
            traces[f"{i}/{key}"] = value
    np.savez(output.with_name(output.stem+"_trace.npz"), **traces)
    return provenance
