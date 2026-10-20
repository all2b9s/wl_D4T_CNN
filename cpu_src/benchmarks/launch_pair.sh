#!/usr/bin/env bash
# Run after salloc --no-shell; this script does not allocate or release resources.
set -euo pipefail
job_id=${1:?Usage: bash cpu_src/benchmarks/launch_pair.sh JOB_ID [OUTPUT_DIRECTORY]}
[[ $job_id =~ ^[0-9]+$ ]] || { echo 'JOB_ID must be numeric' >&2; exit 2; }
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
output=${2:-"$repo/cpu_src/runs/paired_$job_id"}
mkdir -p "$output"
output=$(cd "$output" && pwd)
cd "$repo"
tar -cf "$output/source.tar" --exclude='cpu_src/runs' --exclude='cpu_src/build' \
    --exclude='cpu_src/.build-tmp' --exclude='__pycache__' --exclude='.pytest_cache' src cpu_src
srun --jobid="$job_id" --ntasks=1 --cpus-per-task=72 --gpus-per-task=1 \
    --cpu-bind=cores --gpu-bind=closest --output="$output/run.log" --error="$output/run.err" \
    bash -c 'set -euo pipefail
        repo=$1; output=$2; job_id=$3
        scratch=/tmp/d4cnn_pair_$job_id
        mkdir -p "$scratch/code" "$scratch/work" "$scratch/mpl"
        tar -xf "$output/source.tar" -C "$scratch/code"
        export TMPDIR="$scratch/work" MPLCONFIGDIR="$scratch/mpl"
        export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
        export PYTHONUNBUFFERED=1 PYTHONPATH="$scratch/code"
        cd "$scratch/code"
        exec /u/wenyinli/.conda/envs/anacal_clone/bin/python -B -u -m cpu_src.benchmarks.cluster_pair \
            --repo "$repo" --output "$output" --work "$scratch/work" --samples 512 --repeats 3
    ' bash "$repo" "$output" "$job_id"
