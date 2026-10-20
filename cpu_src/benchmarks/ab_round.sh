#!/usr/bin/env bash
# Build/test/time the working tree against a baseline (git rev or source tar +
# prebuilt extension) on the same node. Run with srun inside an allocation.
#   ab_round.sh TAG [BASELINE_TAR BASELINE_SO]
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside an interactive allocation}"
tag=${1:?round tag}
[[ $tag =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
work=/tmp/d4cnn_ab_${SLURM_JOB_ID}_${tag}
output="$repo/cpu_src/runs/ab_${SLURM_JOB_ID}/$tag"
mkdir -p "$work/new" "$work/data" "$work/temp" "$output"
tar -C "$repo" --exclude=cpu_src/runs --exclude=cpu_src/build --exclude=cpu_src/.build-tmp \
  --exclude='*.so' --exclude=__pycache__ -cf "$output/source.tar" cpu_src
tar -C "$work/new" -xf "$output/source.tar"
cp "$repo/cpu_src/runs/paired_3417691/"{qstamps,s4,attnw,s4_paired,attnw_paired}.npz "$work/data/"
export PATH=/opt/cray/pe/gcc-native/14/bin:$PATH
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TMPDIR="$work/temp" ANACAL_SOURCE_DIR=/projects/bdsp/wenyinli/packages/AnaCal BUILD_JOBS=8
py=/u/wenyinli/.conda/envs/anacal_clone/bin/python
hostname | tee "$output/host.txt"
lscpu > "$output/lscpu.txt"
cd "$work/new"
"$py" -B cpu_src/build_direct.py
g++ -std=c++17 -O3 -pthread -I cpu_src/include cpu_src/tests/test_kernels.cpp cpu_src/native/kernels_simd.cpp \
  cpu_src/native/kernels.cpp cpu_src/native/convolution.cpp -o "$work/test_kernels"
"$work/test_kernels" | tee "$output/test_kernels.log"
PYTHONPATH="$work/new:$repo" "$py" -B -m unittest discover -s cpu_src/tests -v 2>&1 | tail -n 25 | tee "$output/unittest.log"
PYTHONPATH="$work/new:$repo" "$py" -B -m cpu_src validate --model "$repo/cpu_src/runs/attnw.npz" \
  --golden "$repo/cpu_src/runs/golden64.npz" --precision float64 --nthreads 4 --output "$output/golden64.json" | tail -n 3
PYTHONPATH="$work/new:$repo" "$py" -B -m cpu_src.benchmarks.finalize --data "$work/data" --output "$output/new"
cp cpu_src/_native*.so "$output/new/"
if [[ $# -ge 3 ]]; then
  mkdir -p "$work/base"
  tar -C "$work/base" -xf "$2"
  cp "$3" "$work/base/cpu_src/"
  cd "$work/base"
  PYTHONPATH="$work/base:$repo" "$py" -B -m cpu_src.benchmarks.finalize --data "$work/data" --output "$output/baseline" || true
fi
echo "done $tag"
