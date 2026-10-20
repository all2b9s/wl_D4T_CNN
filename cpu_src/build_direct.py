"""Direct GCC/Clang build alternative for slow CMake probes on cluster filesystems."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import subprocess
import sysconfig
import pybind11

def main():
    root=Path(__file__).resolve().parent
    build=root/"build"/"direct"
    build.mkdir(parents=True,exist_ok=True)
    temp=root/".build-tmp"
    temp.mkdir(exist_ok=True)
    env={**os.environ,"TMPDIR":str(temp)}
    compiler=shlex.split(os.environ.get("CPU_CXX","g++"))
    flags=["-std=c++17","-O3","-fPIC","-fvisibility=hidden","-pthread","-pipe",
           "-I"+str(root/"include"),"-I"+pybind11.get_include(),"-I"+sysconfig.get_paths()["include"]]
    sources=[root/"native"/x for x in ("bindings.cpp","convolution.cpp","kernels.cpp","x86.cpp")]
    headers=list((root/"include").rglob("*.h"))+list((root/"native").glob("*.inc"))
    anacal=os.environ.get("ANACAL_SOURCE_DIR")
    if anacal:
        ana=Path(anacal).resolve()
        if not (ana/"include"/"anacal"/"image.h").is_file():
            raise ValueError("ANACAL_SOURCE_DIR must be an AnaCal source checkout")
        flags += ["-I"+str(ana/"include"),"-DD4CNN_WITH_ANACAL=1"]
        sources += [root/"native"/"anacal_adapter.cpp",ana/"src"/"image.cpp"]
        headers += list((ana/"include").rglob("*.h"))
    state_file=build/"compiler.json"
    state={"compiler":compiler,"flags":flags,"sources":[str(p) for p in sources]}
    previous=json.loads(state_file.read_text()) if state_file.exists() else None
    newest_header=max((p.stat().st_mtime for p in headers),default=0)
    def compile_one(source):
        obj=build/(source.stem+".o")
        if previous!=state or not obj.exists() or obj.stat().st_mtime<max(source.stat().st_mtime,newest_header):
            print("Compiling",source,flush=True)
            subprocess.run(compiler+flags+["-c",str(source),"-o",str(obj)],env=env,check=True)
        return obj
    jobs=int(os.environ.get("BUILD_JOBS","2"))
    if jobs<1:
        raise ValueError("BUILD_JOBS must be positive")
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        objects=list(pool.map(compile_one,sources))
    output=root/("_native"+sysconfig.get_config_var("EXT_SUFFIX"))
    staging=build/output.name
    subprocess.run(compiler+["-shared","-pthread"]+[str(p) for p in objects]+["-o",str(staging)],env=env,check=True)
    # Never truncate a shared object that another interpreter may have mapped.
    staging.replace(output)
    state_file.write_text(json.dumps(state,indent=2))
    print("Built",output,flush=True)

if __name__=="__main__":
    main()
