"""Fresh independent Torch CPU baseline and strict-FP32 GPU reference."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from cpu_src.reference import torch_model, measure
from cpu_src.validation import error_stats
from cpu_src.benchmarks.cluster_pair import torch_processes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):raise RuntimeError("Use an interactive compute allocation")
    cores=sorted(os.sched_getaffinity(0))
    torch.set_num_threads(1)
    q=np.load(args.data/"qstamps.npz",allow_pickle=False)["q"]
    report={"torch_version":torch.__version__,"cores":cores,"models":{},"passed":True}
    # Complete every CPU fork pool before initializing CUDA or native executors.
    for name in ("s4","attnw"):
        path=args.data/f"{name}.npz"
        model=torch_model(path)
        rows=[]
        for processes in (1,len(cores)):
            row,arrays=torch_processes(model,path,q[:32] if processes==1 else q,processes,3,cores)
            rows.append(row)
            print(json.dumps({"model":name,"torch_cpu":row}),flush=True)
        report["models"][name]={"cpu_timings":rows}
        np.savez(args.output/f"{name}_torch_cpu.npz",**arrays)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    for name in ("s4","attnw"):
        path=args.data/f"{name}.npz"
        model=torch_model(path,device="cuda")
        fresh=measure(path,q,"float32",4,"cuda",64,_model=model)
        np.savez(args.output/f"{name}_fresh_gpu.npz",e=fresh.e,R=fresh.R,dE_dx=fresh.dE_dx)
        result=np.load(args.output/f"{name}_T2.npz",allow_pickle=False)
        checks={key:error_stats(result[key],getattr(fresh,key),1e-6,1e-5 if key=="e" else 1e-4,1e-4)
                for key in ("e","R")}
        report["models"][name]["fresh_strict_gpu"]=checks
        report["passed"] &= all(row["passed"] for row in checks.values())
        print(json.dumps({"model":name,"fresh_strict_gpu":checks}),flush=True)
        del model
        torch.cuda.empty_cache()
    report["gpu"]=torch.cuda.get_device_name(0)
    (args.output/"reference.json").write_text(json.dumps(report,indent=2)+"\n")
    if not report["passed"]:raise SystemExit("Fresh GPU mean-bias check failed")


if __name__=="__main__":main()
