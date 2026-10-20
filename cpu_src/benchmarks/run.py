import os
import platform
import resource
import time
import numpy as np
from cpu_src import Model
from cpu_src.model_io import load_model

def benchmark(model_path, qstamps=None, count=16, threads=(1,), n_tangent=2, precision="float32", isa="auto", repeats=3, backend="cpp"):
    if backend not in ("cpp","torch"):
        raise ValueError("backend must be cpp or torch")
    if backend=="cpp":
        model=Model.load(model_path,precision,isa)
        meta=model.metadata
        selected_isa=model.isa
    else:
        import torch
        from cpu_src.reference import torch_model,measure
        torch.set_num_threads(1)
        reference_model=torch_model(model_path,precision)
        meta,_=load_model(model_path)
        selected_isa="torch_cpu_autograd"
    if qstamps:
        with np.load(qstamps,allow_pickle=False) as data:
            q=np.array(data["q"][:count])
    else:
        rng=np.random.default_rng(42)
        q=rng.normal(0,.005,(count,5,meta["stamp_size"],meta["stamp_size"]))
    if len(q)==0 or repeats<1:
        raise ValueError("Benchmark requires nonempty input and positive repeats")
    rows=[]
    for workers in threads:
        if backend=="cpp":
            run=lambda sample: model.measure_qstamps(sample,n_tangent,workers)
        else:
            torch.set_num_threads(workers)
            run=lambda sample: measure(model_path,sample,precision,n_tangent,batch_size=1,_model=reference_model)
        run(q[:1])
        samples=[]
        for _ in range(repeats):
            start=time.perf_counter()
            run(q)
            samples.append(time.perf_counter()-start)
        elapsed=float(np.median(samples))
        rows.append({"threads":workers,"galaxies":len(q),"seconds":samples,"median_seconds":elapsed,"gal_per_sec":len(q)/elapsed,
                     "unstable_timing":bool(max(samples)/min(samples)>1.3)})
    baseline=next((r["gal_per_sec"] for r in rows if r["threads"]==1),None)
    for row in rows:
        row["parallel_efficiency"] = row["gal_per_sec"]/(row["threads"]*baseline) if baseline else None
    qualified=not any(r["unstable_timing"] or (r["parallel_efficiency"] is not None and r["parallel_efficiency"]>1.2) for r in rows)
    return {"timing_stable":qualified,"performance_claim":"local smoke only; dedicated-node measurement required",
            "machine":platform.machine(),"platform":platform.platform(),"precision":precision,"isa":selected_isa,"backend":backend,
            "threading":"over sources" if backend=="cpp" else "torch intra-op; source batch size 1",
            "n_tangent":n_tangent,"cpu_affinity":sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None,
            "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"results":rows,
            "timing_scope":"full network and Python input validation; excludes q-stamp preparation"}
