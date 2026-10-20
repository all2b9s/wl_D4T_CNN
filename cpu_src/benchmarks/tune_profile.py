"""Instrument an isolated copy of the optimized kernels, or measure its stages."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

NAMES=["total","first_block","hidden_blocks","winograd_input","winograd_gemm",
       "winograd_output","epilogue","attention","heads","linear","softmax"]
PROBE=r'''
#pragma once
#include <array>
#include <atomic>
#include <chrono>
namespace tune_probe {
constexpr int N=11;
inline std::array<std::atomic<unsigned long long>,N> sums{},counts{};
inline thread_local std::array<unsigned long long,N> local{},calls{};
inline auto now(){return std::chrono::steady_clock::now();}
inline void add(int k,decltype(now()) start){local[k]+=std::chrono::duration_cast<std::chrono::nanoseconds>(now()-start).count();++calls[k];}
struct Scope{int k;decltype(now()) start;Scope(int k_):k(k_),start(now()){}~Scope(){add(k,start);}};
inline void clear(){local.fill(0);calls.fill(0);}
inline void commit(){for(int k=0;k<N;++k){sums[k].fetch_add(local[k]);counts[k].fetch_add(calls[k]);}}
inline void reset(){for(int k=0;k<N;++k){sums[k]=0;counts[k]=0;}}
}
'''


def instrument(base,output):
    shutil.copytree(base/"cpu_src",output/"cpu_src",ignore=shutil.ignore_patterns("build","runs",".build-tmp","__pycache__"))
    def edit(path,replacements):
        text=path.read_text()
        for old,new in replacements:
            if text.count(old)!=1:raise ValueError(f"Expected unique pattern: {old}")
            text=text.replace(old,new)
        path.write_text(text)
    root=output/"cpu_src"
    (root/"include/d4cnn/tuning_probe.h").write_text(PROBE)
    edit(root/"include/d4cnn/model.h",[
        ('#include "convolution.h"','#include "tuning_probe.h"\n#include "convolution.h"'),
        ('X y(x.p,out);const auto& ww=', 'tune_probe::Scope clock(9);X y(x.p,out);const auto& ww='),
        ('void softmax(X& x)const{','void softmax(X& x)const{tune_probe::Scope clock(10);'),
        ('int h=cfg.size,co=cfg.channels;', 'tune_probe::Scope clock(layer==0?1:2);int h=cfg.size,co=cfg.channels;'),
        ('int h=cfg.size,s=std::min(cfg.grid,h)', 'tune_probe::Scope clock(7);int h=cfg.size,s=std::min(cfg.grid,h)'),
        ('X head(X x,const std::string& key)const{', 'X head(X x,const std::string& key)const{tune_probe::Scope clock(8);'),
        ('thread_local Workspace ws;', 'tune_probe::Scope clock(0);thread_local Workspace ws;'),
    ])
    edit(root/"include/d4cnn/winograd.h",[
        (' const int stride=h*h*32;', ' const int stride=h*h*32;'),
        ('for(int oy=0;oy<h;oy+=2)for(int ox=0;ox<h;ox+=2){','for(int oy=0;oy<h;oy+=2)for(int ox=0;ox<h;ox+=2){auto tic=tune_probe::now();'),
        ('  constexpr int lanes=', '  tune_probe::add(3,tic);tic=tune_probe::now();\n  constexpr int lanes='),
        ('  for(int d=0;d<D;++d)for(int c=0;c<32;c+=4){\n   float32x4_t t[2][4];',
         '  tune_probe::add(4,tic);tic=tune_probe::now();\n  for(int d=0;d<D;++d)for(int c=0;c<32;c+=4){\n   float32x4_t t[2][4];'),
        ('  for(int r=0;r<2&&oy+r<h;++r)for(int s=0;s<2&&ox+s<h;++s)finish((oy+r)*h+ox+s);',
         '  tune_probe::add(5,tic);tic=tune_probe::now();\n  for(int r=0;r<2&&oy+r<h;++r)for(int s=0;s<2&&ox+s<h;++s)finish((oy+r)*h+ox+s);tune_probe::add(6,tic);'),
    ])
    edit(root/"native/bindings.cpp",[
        ('const S* ptr=in+', 'tune_probe::clear();const S* ptr=in+'),
        ('   });\n  }return result;', '   tune_probe::commit();});\n  }return result;'),
        (' m.attr("api_version")=2;', ''' m.attr("api_version")=2;
 m.def("profile_reset",&tune_probe::reset);
 m.def("profile_stats",[](){std::vector<unsigned long long> ns,calls;
 for(int k=0;k<tune_probe::N;++k){ns.push_back(tune_probe::sums[k].load());calls.push_back(tune_probe::counts[k].load());}
 py::dict out;out["ns"]=ns;out["calls"]=calls;return out;});'''),
    ])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--data",type=Path)
    parser.add_argument("--instrument",action="store_true")
    args=parser.parse_args()
    if args.instrument:
        instrument(args.base,args.output);return
    sys.path.insert(0,str(args.base))
    import numpy as np
    from cpu_src import Model,_native
    q=np.load(args.data/"qstamps.npz")["q"]
    cores=sorted(os.sched_getaffinity(0))
    result={}
    for name in ("s4","attnw"):
        model=Model.load(args.data/f"{name}.npz")
        rows=[]
        for workers,count in ((1,16),(len(cores),512)):
            os.sched_setaffinity(0,set(cores[:workers]))
            model.measure_qstamps(q[:min(count,workers)],2,workers)
            _native.profile_reset()
            for _ in range(3):model.measure_qstamps(q[:count],2,workers)
            values=_native.profile_stats()
            row={"threads":workers,"count":count,"ms_per_source":{k:n/(3*count)/1e6 for k,n in zip(NAMES,values['ns'])}}
            rows.append(row);print(json.dumps({"model":name,**row}),flush=True)
        result[name]=rows
        os.sched_setaffinity(0,set(cores))
    args.output.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":main()
