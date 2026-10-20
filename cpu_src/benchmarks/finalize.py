"""Real-stamp acceptance and CPU timing; run inside a Slurm interactive step."""
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
from cpu_src import Model, _native
from cpu_src.validation import error_stats


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--repeats",type=int,default=3)
    args=parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Use an interactive compute allocation")
    args.output.mkdir(parents=True,exist_ok=True)
    cores=sorted(os.sched_getaffinity(0))
    q=np.load(args.data/"qstamps.npz",allow_pickle=False)["q"]
    package=Path(__file__).resolve().parents[1]
    files=[package/"api.py",*sorted((package/"native").glob("*.cpp")),
           *sorted((package/"native").glob("*.inc")),*sorted((package/"include/d4cnn").glob("*.h"))]
    report={"job_id":os.environ["SLURM_JOB_ID"],"hostname":platform.node(),"cores":cores,
            "native_sha256":digest(_native.__file__),"source_sha256":{str(p.relative_to(package)):digest(p) for p in files},
            "q_sha256":digest(args.data/"qstamps.npz"),"count":len(q),"repeats":args.repeats,
            "criterion":"abs(mean(CPU-reference, axis=0)) < 1e-4 for each e/R component",
            "reference":"saved strict-FP32 GPU, paired_3417691; identical model and source ordering",
            "timing":"Python API including validation and output, excluding model load and q preparation",
            "models":{}}
    passed=True

    def stats(result,reference):
        return {key:error_stats(getattr(result,key),reference[key],1e-6,1e-5 if key=="e" else 1e-4,1e-4)
                for key in ("e","R")}

    def timed(fn,count):
        fn()
        durations=[]
        for _ in range(args.repeats):
            start=time.perf_counter();fn();durations.append(time.perf_counter()-start)
        return {"count":count,"seconds":durations,"gal_per_sec":count/float(np.median(durations)),
                "stable":max(durations)/min(durations)<1.2}

    for name in ("s4","attnw"):
        model=Model.load(args.data/f"{name}.npz")
        reference=np.load(args.data/f"{name}_paired.npz",allow_pickle=False)
        ref={key:reference[f"strict_fp32_{key}"] for key in ("e","R")}
        result=model.measure_qstamps(q,2,len(cores))
        item={"model_sha256":digest(args.data/f"{name}.npz"),"metadata":result.metadata,
              "T2":stats(result,ref),"timings":[],"batch_sweep":[],"streamed_batches":[]}
        passed &= all(row["passed"] for row in item["T2"].values())
        np.savez(args.output/f"{name}_T2.npz",e=result.e,R=result.R)
        count4=len(reference["gpu_T4_dE_dx"])
        result4=model.measure_qstamps(q[:count4],4,len(cores))
        item["T4"]={"count":count4,**stats(result4,{key:value[:count4] for key,value in ref.items()}),
                    "dE_dx_diagnostic":error_stats(result4.dE_dx,reference["gpu_T4_dE_dx"],1e-6,1e-4),
                    "T2_e_identical":bool(np.array_equal(result.e[:count4],result4.e)),
                    "T2_R_identical":bool(np.array_equal(result.R[:count4],result4.R))}
        passed &= item["T4"]["e"]["passed"] and item["T4"]["R"]["passed"]
        np.savez(args.output/f"{name}_T4.npz",e=result4.e,R=result4.R,dE_dx=result4.dE_dx)
        for workers in (1,8,36,72):
            if workers>len(cores):continue
            os.sched_setaffinity(0,set(cores[:workers]))
            inputs=q[:32 if workers==1 else 128 if workers==8 else len(q)]
            row={"threads":workers,**timed(lambda:model.measure_qstamps(inputs,2,workers),len(inputs))}
            item["timings"].append(row)
            print(json.dumps({"model":name,"scaling":row}),flush=True)
        os.sched_setaffinity(0,set(cores))
        for count in (576,1152,2304):
            inputs=np.ascontiguousarray(q[np.arange(count)%len(q)])
            row={"threads":len(cores),"repeated_sources":True,
                 **timed(lambda:model.measure_qstamps(inputs,2,len(cores)),count)}
            item["batch_sweep"].append(row)
            print(json.dumps({"model":name,"batch":row}),flush=True)
        for batch_size in (64,576):
            def stream():
                for start in range(0,len(q),batch_size):
                    model.measure_qstamps(q[start:start+batch_size],2,len(cores))
            item["streamed_batches"].append({"batch_size":batch_size,**timed(stream,len(q))})
        report["models"][name]=item
        report["passed"]=bool(passed)
        (args.output/"results.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({"model":name,"T2":item["T2"],"T4":item["T4"]}),flush=True)
    if not passed:raise SystemExit("e/R component-mean acceptance failed")


if __name__=="__main__":main()
