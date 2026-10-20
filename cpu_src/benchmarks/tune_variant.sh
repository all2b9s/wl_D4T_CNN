#!/usr/bin/env bash
# A source snapshot/build/acceptance/timing step in an existing allocation.
set -euo pipefail
: "${SLURM_JOB_ID:?Use an interactive allocation}"
tag=${1:?variant name}
algorithm=${2:-auto}
[[ $tag =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
work=/tmp/d4cnn_tune2_${SLURM_JOB_ID}
output="$repo/cpu_src/runs/tune2_${SLURM_JOB_ID}/$tag"
mkdir -p "$work/$tag" "$work/temp" "$output"
tar -C "$repo" --exclude=cpu_src/build --exclude=cpu_src/runs --exclude=cpu_src/.build-tmp --exclude='*.so' --exclude=__pycache__ -cf "$output/source.tar" cpu_src
tar -C "$work/$tag" -xf "$output/source.tar"
export PATH=/opt/cray/pe/gcc-native/14/bin:$PATH
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TMPDIR="$work/temp" PYTHONPATH="$work/$tag:$repo" ANACAL_SOURCE_DIR=/projects/bdsp/wenyinli/packages/AnaCal
py=/u/wenyinli/.conda/envs/anacal_clone/bin/python
cd "$work/$tag"
"$py" -B cpu_src/build_direct.py
g++ -std=c++17 -O3 -pthread -I cpu_src/include cpu_src/tests/test_kernels.cpp cpu_src/native/kernels_simd.cpp cpu_src/native/kernels.cpp cpu_src/native/convolution.cpp -o "$work/$tag/test_math"
"$work/$tag/test_math"
"$py" -B -m unittest discover -s cpu_src/tests -v
D4CNN_CONV="$algorithm" "$py" -B -m cpu_src.benchmarks.profile_native --worker --variant "$tag" --base "$work/$tag" --work "$work/work" --data "$work/data" --output "$output"
D4CNN_CONV="$algorithm" "$py" -B cpu_src/benchmarks/check_profile_t4.py --base "$work/$tag" --data "$work/data" --output "$output"
cp cpu_src/_native*.so "$output/"
cp cpu_src/build/direct/compiler.json "$output/compiler.json"
