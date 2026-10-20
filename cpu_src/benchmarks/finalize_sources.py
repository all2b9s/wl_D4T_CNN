"""Synthetic source-API batching/FFT check, separate from real-stamp throughput."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
from cpu_src import Model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):raise RuntimeError("Use a compute allocation")
    threads=len(os.sched_getaffinity(0))
    rng=np.random.default_rng(20261019)
    image=rng.normal(0,.02,(384,384)).astype(np.float32)
    noise=rng.normal(0,.02,image.shape).astype(np.float32)
    y,x=np.mgrid[:64,:64];psf=np.exp(-((x-32)**2+(y-32)**2)/(2*1.8**2));psf/=psf.sum()
    positions=rng.uniform(40,344,(512,2))
    report={"dataset":"synthetic 384x384 canvas, 512 non-edge positions; not a science sample",
            "threads":threads,"models":{}}
    for name in ("s4","attnw"):
        model=Model.load(args.data/f"{name}.npz")
        q=model.prepare_qstamps(image,psf,positions,noise,threads)
        expected=model.measure_qstamps(q,4,threads)
        result=model.measure_sources(image,psf,positions,noise,4,threads)
        for key in ("e","R","dE_dx"):
            np.testing.assert_array_equal(getattr(result,key),getattr(expected,key))
        rows=[]
        for batch in (64,None):
            durations=[]
            model.measure_sources(image,psf,positions,noise,2,threads,batch)
            for _ in range(3):
                start=time.perf_counter()
                model.measure_sources(image,psf,positions,noise,2,threads,batch)
                durations.append(time.perf_counter()-start)
            rows.append({"batch_size":batch,"effective_batch_size":batch or max(64,8*threads),
                         "seconds":durations,"gal_per_sec":512/float(np.median(durations))})
        report["models"][name]={"source_vs_qstamps_T4_identical":True,"timings":rows}
    (args.output/"source_api.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__":main()
