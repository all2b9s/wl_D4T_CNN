#!/usr/bin/env bash
# Final counter/compatibility checks for the isolated 3418136 experiments.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside the existing compute allocation}"
repo=/projects/bdsp/wenyinli/codes/single_galaxies
scratch=/tmp/d4cnn_profile_${SLURM_JOB_ID}
output="$repo/cpu_src/runs/profile_${SLURM_JOB_ID}"
python_bin=/u/wenyinli/.conda/envs/anacal_clone/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$scratch"
export MPLCONFIGDIR="$scratch/matplotlib"
cp "$repo/cpu_src/benchmarks/perf_native.py" "$scratch/perf_native.py"
cp "$repo/cpu_src/benchmarks/check_profile_t4.py" "$scratch/check_profile_t4.py"
cp "$repo/cpu_src/benchmarks/summarize_profile.py" "$scratch/summarize_profile.py"
for variant in baseline combined; do
    base="$scratch/base"
    if [[ "$variant" == combined ]]; then base="$scratch/work/combined"; fi
    for model in s4 attnw; do
        perf stat -x , -o "$output/perf72_${variant}_${model}.csv" \
            -e cycles:u,instructions:u,l1d_cache_refill:u,l2d_cache_refill:u,stall_backend:u,stall_backend_mem:u \
            "$python_bin" -B "$scratch/perf_native.py" --base "$base" --data "$scratch/data" \
            --model "$model" --count 1152 --threads 72
    done
done
"$python_bin" -B "$scratch/check_profile_t4.py" --base "$scratch/work/combined" \
    --data "$scratch/data" --output "$output"
# Keep the exact experimental source/binary snapshots and compiler metadata.
tar -C "$scratch" -czf "$output/experimental_builds.tar.gz" base/cpu_src work \
    profile_native.py profile_extra.py perf_native.py check_profile_t4.py summarize_profile.py
"$python_bin" -B "$scratch/summarize_profile.py" "$output"
sha256sum "$output/experimental_builds.tar.gz" > "$output/experimental_builds.sha256"
scontrol show job "$SLURM_JOB_ID" > "$output/allocation.txt"
