#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Use the compute allocation}"
repo=/projects/bdsp/wenyinli/codes/single_galaxies
scratch=/tmp/d4cnn_optimize_${SLURM_JOB_ID}
output="$repo/cpu_src/runs/optimize_${SLURM_JOB_ID}"
mkdir -p "$scratch/current" "$scratch/data" "$output"
tar -C "$repo" --exclude=cpu_src/runs --exclude=cpu_src/build --exclude=cpu_src/.build-tmp --exclude=__pycache__ -cf - cpu_src | tar -C "$scratch/current" -xf -
cp "$repo/cpu_src/runs/paired_3417691/"{qstamps,s4,attnw,s4_paired,attnw_paired}.npz "$scratch/data/"
export PATH=/opt/cray/pe/gcc-native/14/bin:$PATH
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$scratch/current:$repo"
export ANACAL_SOURCE_DIR=/projects/bdsp/wenyinli/packages/AnaCal
python_bin=/u/wenyinli/.conda/envs/anacal_clone/bin/python
cd "$scratch/current"
"$python_bin" -B cpu_src/build_direct.py
g++ -std=c++17 -O3 -pthread -I cpu_src/include cpu_src/tests/test_simd_math.cpp -o "$scratch/test_math"
"$scratch/test_math"
"$python_bin" -B -m unittest discover -s cpu_src/tests -v
"$python_bin" -B cpu_src/benchmarks/profile_native.py --worker --variant optimized --base "$scratch/current" --work "$scratch/work" --data "$scratch/data" --output "$output"
D4CNN_CONV=direct "$python_bin" -B cpu_src/benchmarks/profile_native.py --worker --variant optimized_direct --base "$scratch/current" --work "$scratch/work" --data "$scratch/data" --output "$output"
"$python_bin" -B cpu_src/benchmarks/profile_native.py --worker --balanced --variant optimized_balanced --base "$scratch/current" --work "$scratch/work" --data "$scratch/data" --output "$output"
