#!/usr/bin/env bash
# Execute with srun in an existing interactive allocation. Never allocates jobs.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside an interactive allocation}"
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
tag=${1:-final}
[[ $tag =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
scratch=/tmp/d4cnn_finalize_${SLURM_JOB_ID}_${tag}
output="$repo/cpu_src/runs/finalize_${SLURM_JOB_ID}/$tag"
mkdir -p "$scratch/code" "$scratch/data" "$scratch/temp" "$output"
tar -C "$repo" --exclude=cpu_src/runs --exclude=cpu_src/build --exclude=cpu_src/.build-tmp \
  --exclude='*.so' --exclude=__pycache__ -cf "$output/source.tar" cpu_src
tar -C "$scratch/code" -xf "$output/source.tar"
cp "$repo/cpu_src/runs/paired_3417691/"{qstamps,s4,attnw,s4_paired,attnw_paired}.npz "$scratch/data/"
export PATH=/opt/cray/pe/gcc-native/14/bin:$PATH
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$scratch/code:$repo" TMPDIR="$scratch/temp" PYTHONUNBUFFERED=1
export ANACAL_SOURCE_DIR=/projects/bdsp/wenyinli/packages/AnaCal
python_bin=/u/wenyinli/.conda/envs/anacal_clone/bin/python
cd "$scratch/code"
hostname
lscpu > "$output/lscpu.txt"
"$python_bin" -B cpu_src/build_direct.py
g++ -std=c++17 -O3 -pthread -I cpu_src/include cpu_src/tests/test_simd_math.cpp -o "$scratch/test_math"
"$scratch/test_math"
"$python_bin" -B -m unittest discover -s cpu_src/tests -v
"$python_bin" -B -m cpu_src validate --model "$repo/cpu_src/runs/attnw.npz" \
  --golden "$repo/cpu_src/runs/golden64.npz" --precision float64 --nthreads 4 --output "$output/golden64.json"
"$python_bin" -B -m cpu_src.benchmarks.finalize --data "$scratch/data" --output "$output"
cp cpu_src/_native*.so "$output/"
cp cpu_src/build/direct/compiler.json "$output/compiler.json"
sha256sum cpu_src/_native*.so > "$output/native.sha256"
