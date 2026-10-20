"""Isolated Grace profiling/ablation builds; never replaces the installed native extension.

Use a Slurm compute node. --base is a local source snapshot, --work local scratch,
--data the previous paired validation directory, and --output persistent artifacts.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import time

PROBE = r'''
#pragma once
#include <array>
#include <atomic>
#include <chrono>
namespace probe {
constexpr int N=14;
inline std::array<std::atomic<unsigned long long>,N> sums{}, counts{};
inline thread_local std::array<unsigned long long,N> local{}, calls{};
inline auto now(){return std::chrono::steady_clock::now();}
inline void add(int key, std::chrono::steady_clock::time_point start){
 local[key]+=std::chrono::duration_cast<std::chrono::nanoseconds>(now()-start).count(); ++calls[key];
}
struct Scope {int key; decltype(now()) start; Scope(int k):key(k),start(now()){} ~Scope(){add(key,start);}};
inline void clear(){local.fill(0);calls.fill(0);}
inline void commit(){for(int i=0;i<N;++i){sums[i].fetch_add(local[i],std::memory_order_relaxed);counts[i].fetch_add(calls[i],std::memory_order_relaxed);}}
inline void reset(){for(int i=0;i<N;++i){sums[i]=0;counts[i]=0;}}
}
'''
NAMES = ["total", "normalize", "field_allocate", "orbit_input", "block_allocate",
         "conv_first", "conv_hidden", "layernorm", "gelu", "residual", "orbit_reduce",
         "attention", "pool", "heads"]

SPATIAL = r'''
// Experimental two-pixel / 16-output-channel tile: 24 vector accumulators.
static void spatial2(const float*x,const float*w,const float*b,float*y,int h,int ci,int co) {
 int p=h*h;
 for(int yy=0;yy<h;++yy)for(int xx=0;xx<h;xx+=2)for(int o=0;o<co;o+=16){
  float32x4_t acc[2][3][4];
  for(int a=0;a<2;++a)for(int d=0;d<3;++d)for(int l=0;l<4;++l)
   acc[a][d][l]=d==0?vld1q_f32(b+o+4*l):vdupq_n_f32(0);
  for(int ky=0;ky<3;++ky){int iy=yy+ky-1;if(iy<0||iy>=h)continue;
   for(int kx=0;kx<3;++kx){int ix=xx+kx-1;
    for(int i=0;i<ci;++i){
     float values[2][3];
     for(int a=0;a<2;++a)for(int d=0;d<3;++d)
      values[a][d]=(ix+a>=0&&ix+a<h)?x[(d*p+iy*h+ix+a)*ci+i]:0.f;
     for(int l=0;l<4;++l){auto ww=vld1q_f32(w+((ky*3+kx)*ci+i)*co+o+4*l);
      for(int a=0;a<2;++a)for(int d=0;d<3;++d)acc[a][d][l]=vfmaq_n_f32(acc[a][d][l],ww,values[a][d]);}
    }
   }
  }
  for(int a=0;a<2;++a)for(int d=0;d<3;++d)for(int l=0;l<4;++l)
   vst1q_f32(y+(d*p+yy*h+xx+a)*co+o+4*l,acc[a][d][l]);
 }
}
'''

VECTOR_GELU = r'''
  // Experimental numerical evaluation of erf-GELU. The scalar tanh path is unchanged.
  if constexpr(std::is_same_v<S,float>){if(!tanh){
   int n=x.p*x.c,i=0;
   for(;i+4<=n;i+=4){
    auto v=vld1q_f32(x.a.data()+i);
    auto z=vmulq_n_f32(v,0.7071067811865475f);
    auto t=vdivq_f32(vdupq_n_f32(1.f),vaddq_f32(vdupq_n_f32(1.f),vmulq_n_f32(vabsq_f32(z),0.3275911f)));
    auto poly=vdupq_n_f32(1.061405429f);
    poly=vfmaq_f32(vdupq_n_f32(-1.453152027f),poly,t);
    poly=vfmaq_f32(vdupq_n_f32(1.421413741f),poly,t);
    poly=vfmaq_f32(vdupq_n_f32(-0.284496736f),poly,t);
    poly=vfmaq_f32(vdupq_n_f32(0.254829592f),poly,t);
    poly=vmulq_f32(poly,t);
    auto ex=_ZGVnN4v_expf(vmulq_n_f32(vmulq_f32(v,v),-0.5f));
    auto ea=vsubq_f32(vdupq_n_f32(1.f),vmulq_f32(poly,ex));
    auto erfv=vbslq_f32(vcltq_f32(v,vdupq_n_f32(0.f)),vnegq_f32(ea),ea);
    auto phi=vmulq_n_f32(vaddq_f32(vdupq_n_f32(1.f),erfv),0.5f);
    auto df=vfmaq_f32(phi,vmulq_n_f32(v,0.3989422804014327f),ex);
    for(int d=1;d<D;++d)vst1q_f32(x.a.data()+d*n+i,vmulq_f32(vld1q_f32(x.a.data()+d*n+i),df));
    vst1q_f32(x.a.data()+i,vmulq_f32(v,phi));
   }
   for(;i<n;++i){S v=x.a[i],phi=S(.5)*(S(1)+std::erf(v*S(0.7071067811865475))),df=phi+v*std::exp(S(-.5)*v*v)*S(0.3989422804014327);
    for(int d=1;d<D;++d)x.a[d*n+i]*=df;x.a[i]=v*phi;}
   return;
  }}
'''

VECTOR_LN = r'''
  if constexpr(std::is_same_v<S,float>){if(x.c==32){
   int stride=x.p*32;
   for(int p=0;p<x.p;++p){float* base=x.a.data()+p*32;
    float32x4_t sum=vdupq_n_f32(0.f),center[8];
    for(int j=0;j<8;++j)sum=vaddq_f32(sum,vld1q_f32(base+4*j));
    float mu=vaddvq_f32(sum)/32.f;sum=vdupq_n_f32(0.f);
    for(int j=0;j<8;++j){center[j]=vsubq_f32(vld1q_f32(base+4*j),vdupq_n_f32(mu));sum=vfmaq_f32(sum,center[j],center[j]);}
    float inv=1.f/std::sqrt(vaddvq_f32(sum)/32.f+1e-5f);
    for(int d=1;d<D;++d){float* tangent=base+d*stride;sum=vdupq_n_f32(0.f);
     for(int j=0;j<8;++j)sum=vaddq_f32(sum,vld1q_f32(tangent+4*j));
     float dm=vaddvq_f32(sum)/32.f;sum=vdupq_n_f32(0.f);
     for(int j=0;j<8;++j)sum=vfmaq_f32(sum,center[j],vsubq_f32(vld1q_f32(tangent+4*j),vdupq_n_f32(dm)));
     float cov=vaddvq_f32(sum)/32.f;
     for(int j=0;j<8;++j){auto dt=vsubq_f32(vld1q_f32(tangent+4*j),vdupq_n_f32(dm));
      dt=vsubq_f32(dt,vmulq_n_f32(vmulq_n_f32(vmulq_n_f32(center[j],cov),inv),inv));
      vst1q_f32(tangent+4*j,vmulq_f32(vmulq_n_f32(vld1q_f32(gamma.data()+4*j),inv),dt));}
    }
    for(int j=0;j<8;++j)vst1q_f32(base+4*j,vaddq_f32(vmulq_f32(vmulq_n_f32(center[j],inv),vld1q_f32(gamma.data()+4*j)),vld1q_f32(beta.data()+4*j)));
   }
   return;
  }}
'''


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Source pattern must occur exactly once: {before[:90]}")
    return text.replace(before, after, 1)


def instrument(root):
    header = root / "cpu_src/include/d4cnn/model.h"
    text = header.read_text()
    text = '#include "profile_probe.h"\n' + text
    changes = [
        ('int h=cfg.size,co=cfg.channels; X y(h*h,co);',
         'auto alloc_start=probe::now(); int h=cfg.size,co=cfg.channels; X y(h*h,co); probe::add(4,alloc_start);'),
        ('if constexpr(std::is_same_v<S,float>)conv(', 'auto conv_start=probe::now(); if constexpr(std::is_same_v<S,float>)conv('),
        ('save(trace,key+".conv",y);layernorm(y,key+".norm");save(trace,key+".norm",y);',
         'probe::add(layer==0?5:6,conv_start);save(trace,key+".conv",y);auto ln_start=probe::now();layernorm(y,key+".norm");probe::add(7,ln_start);save(trace,key+".norm",y);'),
        ('activate(y);save(trace,key+".gelu",y);',
         'auto act_start=probe::now();activate(y);probe::add(8,act_start);save(trace,key+".gelu",y);auto residual_start=probe::now();'),
        ('save(trace,key+".out",y);return y;', 'probe::add(9,residual_start);save(trace,key+".out",y);return y;'),
        ('int h=cfg.size,s=std::min(cfg.grid,h)', 'probe::Scope timer(11);int h=cfg.size,s=std::min(cfg.grid,h)'),
        ('X head(X x,const std::string& key)const{', 'X head(X x,const std::string& key)const{probe::Scope timer(13);'),
        ('int h=cfg.size,p=h*h,c=cfg.channels;X normalized',
         'probe::Scope total_timer(0);auto norm_start=probe::now();int h=cfg.size,p=h*h,c=cfg.channels;X normalized'),
        ('save(trace,"normalized",normalized);X scalar(p,c),e1(p,c),e2(p,c);',
         'save(trace,"normalized",normalized);probe::add(1,norm_start);auto field_start=probe::now();X scalar(p,c),e1(p,c),e2(p,c);probe::add(2,field_start);'),
        ('for(int g=0;g<8;++g){X x(p,1);', 'for(int g=0;g<8;++g){auto orbit_start=probe::now();X x(p,1);'),
        ('for(int l=0;l<cfg.layers;++l)x=block(x,l,g==0?trace:nullptr);',
         'probe::add(3,orbit_start);for(int l=0;l<cfg.layers;++l)x=block(x,l,g==0?trace:nullptr);auto reduce_start=probe::now();'),
        ('scalar.at(d,canonical,j)+=v;e1.at(d,canonical,j)+=S(sign1[g])*v;e2.at(d,canonical,j)+=S(sign2[g])*v;}}',
         'scalar.at(d,canonical,j)+=v;e1.at(d,canonical,j)+=S(sign1[g])*v;e2.at(d,canonical,j)+=S(sign2[g])*v;}}probe::add(10,reduce_start);'),
        ('X v1(1,c),v2(1,c);', 'auto pool_start=probe::now();X v1(1,c),v2(1,c);'),
        ('save(trace,"pooled_e1",v1);', 'probe::add(12,pool_start);save(trace,"pooled_e1",v1);'),
    ]
    for before, after in changes:
        text = replace_once(text, before, after)
    header.write_text(text)
    (header.parent / "profile_probe.h").write_text(PROBE)
    binding = root / "cpu_src/native/bindings.cpp"
    text = binding.read_text()
    text = replace_once(text, 'const S* ptr=in+size_t(i)*5*cfg.size*cfg.size;',
                        'probe::clear();const S* ptr=in+size_t(i)*5*cfg.size*cfg.size;')
    text = replace_once(text, '   });\n  }return result;', '   probe::commit();});\n  }return result;')
    text = replace_once(text, 'PYBIND11_MODULE(_native,m){', '''PYBIND11_MODULE(_native,m){
 m.def("profile_reset",&probe::reset);
 m.def("profile_stats",[](){py::dict out;std::vector<unsigned long long> ns,calls;
  for(int i=0;i<probe::N;++i){ns.push_back(probe::sums[i].load());calls.push_back(probe::counts[i].load());}
  out["ns"]=ns;out["calls"]=calls;return out;});''')
    binding.write_text(text)


def build_variant(args, name, flags, kind):
    import pybind11
    root = args.work / name
    shutil.copytree(args.base / "cpu_src", root / "cpu_src", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("runs", "build", ".build-tmp", "__pycache__"))
    if kind in ("profile", "vector_profile", "combined_profile"):
        instrument(root)
    if kind in ("vector", "vector_profile", "combined", "combined_profile", "ln"):
        path = root / "cpu_src/include/d4cnn/model.h"
        text = '#include <arm_neon.h>\nextern "C" float32x4_t _ZGVnN4v_expf(float32x4_t);\n' + path.read_text()
        if kind != "ln":
            text = replace_once(text, 'void activate(X& x,bool tanh=false)const{',
                                'void activate(X& x,bool tanh=false)const{\n' + VECTOR_GELU)
        if kind in ("combined", "combined_profile", "ln"):
            before = 'const auto& gamma=get(key+".weight");const auto& beta=get(key+".bias");'
            text = replace_once(text, before, before + '\n' + VECTOR_LN)
        path.write_text(text)
    if kind in ("nocopy", "combined", "combined_profile"):
        path = root / "cpu_src/include/d4cnn/model.h"
        path.write_text(replace_once(path.read_text(),
            'X skip=x.c==co?x:linear(x,key+".skip",co,false);add(y,skip,S(cfg.residual));',
            'if(x.c==co)add(y,x,S(cfg.residual));else {X skip=linear(x,key+".skip",co,false);add(y,skip,S(cfg.residual));}'))
    if kind in ("tile16", "spatial2"):
        path = root / "cpu_src/native/kernels.cpp"
        text = path.read_text()
        if kind == "tile16":
            text = replace_once(text, 'neon_tiles<3,8>', 'neon_tiles<3,4>')
        else:
            text = replace_once(text, 'static void conv_neon(', SPATIAL + '\nstatic void conv_neon(')
            text = replace_once(text, 'if(nd==3)neon_tiles<3,8>',
                                'if(nd==3&&ci==32&&co==32&&h%2==0)spatial2(x,w,b,y,h,ci,co);else if(nd==3)neon_tiles<3,8>')
        path.write_text(text)
    command = ["g++", "-std=c++17", "-O3", "-fPIC", "-fvisibility=hidden", "-pthread", "-pipe"] + flags
    command += ["-I" + str(root / "cpu_src/include"), "-I" + pybind11.get_include(),
                "-I" + sysconfig.get_paths()["include"]]
    output = root / "cpu_src" / ("_native" + sysconfig.get_config_var("EXT_SUFFIX"))
    with (args.output / f"build_{name}.log").open("w") as log:
        link = ["-lmvec"] if kind in ("vector", "vector_profile", "combined", "combined_profile") else []
        subprocess.run(command + ["-shared"] + [str(root / "cpu_src/native" / f) for f in
                       ("bindings.cpp", "convolution.cpp", "kernels.cpp")] + ["-o", str(output)] + link,
                       check=True, stdout=log, stderr=subprocess.STDOUT)
    # Retain assembly for source-to-machine-code analysis.
    if name in ("rebuild", "tuned", "fastmath", "spatial2"):
        subprocess.run(command + ["-S", str(root / "cpu_src/native/kernels.cpp"), "-o",
                       str(args.output / f"{name}_kernels.s")], check=True)
    (args.output / f"{name}_build.json").write_text(json.dumps({"command": command, "kind": kind}, indent=2))
    print(json.dumps({"built": name}), flush=True)
    return root


def accuracy(a, b):
    import numpy as np
    delta = a.astype(np.float64) - b.astype(np.float64)
    bias = delta.mean(0)
    return {"mean_bias": bias.tolist(), "max_abs_mean_bias": float(np.abs(bias).max()),
            "mean_bias_passed": bool(np.isfinite(delta).all() and (np.abs(bias) < 1e-4).all()),
            "max_abs": float(np.abs(delta).max()), "rms": float(np.sqrt(np.mean(delta**2)))}


def worker(args):
    sys.path.insert(0, str(args.base))
    import numpy as np
    from cpu_src import Model, _native
    q = np.load(args.data / "qstamps.npz", allow_pickle=False)["q"]
    cores = sorted(os.sched_getaffinity(0))
    report = {"variant": args.variant, "cores": cores, "models": {}}
    for name in ("s4", "attnw"):
        model = Model.load(args.data / f"{name}.npz")
        rows = []
        settings = [(1, 32), (72, 512)]
        if args.balanced:
            settings = [(72, n) for n in (512, 576, 1152, 2304)]
        for workers, count in settings:
            if workers > len(cores):
                continue
            inputs = np.ascontiguousarray(np.tile(q, (int(np.ceil(count / len(q))), 1, 1, 1))[:count])
            os.sched_setaffinity(0, set(cores[:workers]))
            model.measure_qstamps(inputs[:min(count, workers * 2)], 2, workers)
            times = []
            if hasattr(_native, "profile_reset"):
                _native.profile_reset()
            for _ in range(3):
                start = time.perf_counter()
                result = model.measure_qstamps(inputs, 2, workers)
                times.append(time.perf_counter() - start)
            row = {"threads": workers, "count": count, "seconds": times,
                   "gal_per_sec": count / float(np.median(times))}
            if hasattr(_native, "profile_stats"):
                stats = _native.profile_stats()
                row["profile"] = {key: {"ms_per_source": ns / (3 * count) / 1e6,
                                       "calls_per_source": calls / (3 * count)}
                                  for key, ns, calls in zip(NAMES, stats["ns"], stats["calls"])}
            if count == 512:
                saved = np.load(args.data / f"{name}_paired.npz", allow_pickle=False)
                row["accuracy_vs_strict_gpu"] = {"e": accuracy(result.e, saved["strict_fp32_e"]),
                                                "R": accuracy(result.R, saved["strict_fp32_R"])}
                np.savez(args.output / f"{args.variant}_{name}_predictions.npz", e=result.e, R=result.R)
            rows.append(row)
            print(json.dumps({"variant": args.variant, "model": name, **row}), flush=True)
        report["models"][name] = rows
        os.sched_setaffinity(0, set(cores))
    (args.output / f"{args.variant}.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "work", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--experiments", nargs="+", help="Build/run only these named variants, without the original baseline")
    parser.add_argument("--skip-balanced", action="store_true")
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Use an allocated compute node")
    args.work.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    configurations = [
        ("rebuild", [], "none"), ("profile", [], "profile"), ("nocopy", [], "nocopy"),
        ("tuned", ["-mcpu=neoverse-v2", "-fno-math-errno"], "none"),
        ("profile_tuned", ["-mcpu=neoverse-v2", "-fno-math-errno"], "profile"),
        ("fastmath", ["-mcpu=neoverse-v2", "-ffast-math"], "none"),
        ("tile16", [], "tile16"), ("spatial2", [], "spatial2"),
        ("vector_gelu", [], "vector"), ("vector_gelu_profile", [], "vector_profile"),
        ("vector_ln", [], "ln"), ("combined", [], "combined"), ("combined_profile", [], "combined_profile"),
        ("tuned_nosched", ["-mcpu=neoverse-v2", "-fno-math-errno", "-fno-schedule-insns"], "none")]
    if args.experiments:
        known = {row[0] for row in configurations}
        if set(args.experiments) - known:
            raise ValueError("Unknown experiment")
        configurations = [row for row in configurations if row[0] in args.experiments]
    with ThreadPoolExecutor(max_workers=2) as pool:
        roots = list(pool.map(lambda row: build_variant(args, *row), configurations))
    runs = ([] if args.experiments else [("baseline", args.base)]) + [(row[0], root) for row, root in zip(configurations, roots)]
    for name, root in runs:
        command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--worker", "--variant", name,
                   "--base", str(root), "--work", str(args.work), "--data", str(args.data), "--output", str(args.output)]
        subprocess.run(command, check=True)
    # Balance work over 72 cores to distinguish wave-tail losses from resource contention.
    for name, root in (() if args.skip_balanced else (("baseline_balanced", args.base), ("tuned_balanced", args.work / "tuned"))):
        subprocess.run([sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--worker", "--balanced",
                        "--variant", name, "--base", str(root), "--work", str(args.work), "--data", str(args.data),
                        "--output", str(args.output)], check=True)


if __name__ == "__main__":
    main()
