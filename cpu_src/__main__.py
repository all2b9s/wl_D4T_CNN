"""Run from single_galaxies: python -m cpu_src --help."""
import argparse
import json
from pathlib import Path

def emit(value, output=None):
    text=json.dumps(value,indent=2)
    print(text)
    if output:
        path=Path(output)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(text+"\n")

def main(argv=None):
    parser=argparse.ArgumentParser(description="D4CNN CPU inference and AnaCal pipeline")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("export",help="Export a trusted checkpoint; requires torch")
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--config")
    p.add_argument("--architecture",choices=["Forward8_fixW_CNN","Forward8_AttnW_CNN"])
    p.add_argument("--model-kwargs",default="{}",help="JSON object; conflicts with config are rejected")
    p.add_argument("--stamp-size",type=int,default=64)
    p.add_argument("--pixel-scale",type=float)
    p.add_argument("--sigma-arcsec",type=float)
    p.add_argument("--flim",type=float)
    p=sub.add_parser("golden",help="Create autograd golden data; requires torch")
    p.add_argument("--model",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--qstamps",help="NPZ with q[N,5,H,H]; omitted => synthetic")
    p.add_argument("--count",type=int,default=512)
    p.add_argument("--precision",choices=["float32","float64"],default="float64")
    p.add_argument("--device",default="cpu")
    p.add_argument("--seed",type=int,default=42)
    for name in ("measure","calibrate","run"):
        p=sub.add_parser(name,help="Standalone blended pipeline: "+name)
        p.add_argument("--config",required=True)
    p=sub.add_parser("infer",help="Measure a saved q-stamp NPZ without torch")
    p.add_argument("--model",required=True)
    p.add_argument("--qstamps",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--n-tangent",type=int,choices=[2,4],default=2)
    p.add_argument("--nthreads",type=int,default=1)
    p.add_argument("--precision",choices=["float32","float64"],default="float32")
    p.add_argument("--isa",default="auto")
    for name in ("validate","benchmark"):
        p=sub.add_parser(name)
        p.add_argument("--model",required=True)
        p.add_argument("--precision",choices=["float32","float64"],default="float32")
        p.add_argument("--isa",default="auto")
        p.add_argument("--output")
        if name=="validate":
            p.add_argument("--golden",required=True)
            p.add_argument("--nthreads",type=int,default=1)
        else:
            p.add_argument("--backend",choices=["cpp","torch"],default="cpp")
            p.add_argument("--qstamps")
            p.add_argument("--count",type=int,default=16)
            p.add_argument("--threads",nargs="+",type=int,default=[1])
            p.add_argument("--n-tangent",type=int,choices=[2,4],default=2)
            p.add_argument("--repeats",type=int,default=3)
    p=sub.add_parser("compare",help="Compare two paired catalog NPZ files")
    p.add_argument("left")
    p.add_argument("right")
    p.add_argument("--output")
    p=sub.add_parser("demo",help="Generate and run two tiny synthetic FITS exposures; no torch")
    p.add_argument("--model",required=True)
    p.add_argument("--output",required=True,help="New empty directory for inputs, config and catalogs")
    p=sub.add_parser("compare-science",help="Paired exposure bootstrap of two pipeline manifests")
    p.add_argument("left_manifest")
    p.add_argument("right_manifest")
    p.add_argument("--repeats",type=int,default=200)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--output")
    a=parser.parse_args(argv)
    if a.command=="export":
        from .export import export_checkpoint
        emit(export_checkpoint(a.checkpoint,a.output,a.config,a.architecture,json.loads(a.model_kwargs),
                               a.stamp_size,a.pixel_scale,a.sigma_arcsec,a.flim))
    elif a.command=="golden":
        from .make_golden import make_golden
        emit(make_golden(a.model,a.output,a.qstamps,a.count,a.precision,a.seed,a.device))
    elif a.command in ("measure","calibrate","run"):
        from .pipeline.config import PipelineConfig
        c=PipelineConfig.load(a.config)
        if a.command in ("measure","run"):
            from .pipeline.runner import run_measurement
            run_measurement(c)
        if a.command in ("calibrate","run"):
            from .pipeline.calibration import calibrate
            emit(calibrate(c))
    elif a.command=="infer":
        import numpy as np
        from . import Model
        model=Model.load(a.model,a.precision,a.isa)
        with np.load(a.qstamps,allow_pickle=False) as data:
            result=model.measure_qstamps(data["q"],a.n_tangent,a.nthreads)
        arrays={"e":result.e,"R":result.R,"metadata":np.frombuffer(json.dumps(result.metadata).encode(),dtype=np.uint8)}
        if result.dE_dx is not None:
            arrays["dE_dx"]=result.dE_dx
        Path(a.output).parent.mkdir(parents=True,exist_ok=True)
        np.savez(a.output,**arrays)
    elif a.command=="validate":
        from .validation import validate_golden
        report=validate_golden(a.model,a.golden,a.precision,a.isa,a.nthreads)
        emit(report,a.output)
        if not report["passed"] or (a.precision=="float64" and not report["trace_passed"]):
            raise SystemExit(1)
    elif a.command=="benchmark":
        from .benchmarks.run import benchmark
        emit(benchmark(a.model,a.qstamps,a.count,a.threads,a.n_tangent,a.precision,a.isa,a.repeats,a.backend),a.output)
    elif a.command=="demo":
        from .pipeline.demo import create_demo
        from .pipeline.runner import run_measurement
        from .pipeline.calibration import calibrate
        c=create_demo(a.model,a.output)
        run_measurement(c)
        emit(calibrate(c))
    elif a.command=="compare-science":
        from .science_validation import compare_runs
        emit(compare_runs(a.left_manifest,a.right_manifest,a.repeats,a.seed),a.output)
    elif a.command=="compare":
        from .validation import paired_catalogs
        report=paired_catalogs(a.left,a.right)
        emit(report,a.output)
        if not all(x["passed"] for x in report.values()):
            raise SystemExit(1)

if __name__=="__main__":
    main()
