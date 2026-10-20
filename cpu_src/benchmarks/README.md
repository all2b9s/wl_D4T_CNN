# Benchmark and diagnostic tools

For general CPU throughput, use the public CLI from the repository root:

```bash
python -m cpu_src benchmark --model /path/to/model.npz \
  --qstamps /path/to/qstamps.npz --count 512 --threads 1 8 32 \
  --repeats 3 --output cpu_src/runs/benchmark.json
```

This works without a scheduler. Choose available thread counts, compile the
extension for the target CPU, and use allocated compute resources where required
by your site. See the [package README](../README.md) for build instructions,
ISA/convolution overrides and numerical acceptance criteria.

## Research utilities

These tools retain the input conventions of the paired-model experiments.
They are useful with matching exported models and saved references, but are
not general dataset loaders. Except `run.py`, `reassess_pair.py` and the standalone
C++ microbenchmark, the Python runners currently require an active Slurm job
(`SLURM_JOB_ID`) and Linux CPU affinity support. They do not allocate resources.

| Tool | Purpose and inputs |
|---|---|
| `run.py` | Implementation of the public `benchmark` CLI; arbitrary exported model and optional q-stamps. |
| `finalize.py` | Paired s4/AttnW numerical checks and CPU scaling/batch sweeps; `--data`, `--output`, optional `--repeats`. |
| `finalize_sources.py` | Synthetic source-API/q-stamp agreement and batching; `--data`, `--output`; requires AnaCal support. |
| `finalize_reference.py` | Independent Torch CPU and strict-FP32 CUDA GPU reference; `--data`, `--output`; consumes the saved `*_T2.npz` in the output directory. |
| `check_profile_t4.py` | T=4 checks against saved references; `--base` points to the directory containing a built `cpu_src`, plus `--data`, `--output`. |
| `profile_phases.py` | Phase timings from a build compiled with `-DD4CNN_PROFILE`; `--base`, `--data`, `--output`. Nested phase sums are not additive. |
| `perf_native.py` | Workload for external `perf stat`/`perf record`; `--base`, `--data`, optional `--model`, `--count`, `--threads`. |
| `micro_epilogue.cpp` | Standalone epilogue/LayerNorm/GELU microbenchmark using the current shared kernels. |
| `cluster_pair.py` | Original paired CPU/GPU experiment. `--repo`, `--work`, `--output` identify source/data and scratch/output roots; model paths and exposure selection follow the historical experiment layout. |
| `diagnose_pair.py` | FP64 outlier diagnostics on a saved paired experiment; `--output` identifies that experiment directory. |
| `reassess_pair.py` | Recompute the component-mean acceptance of saved paired outputs; takes an experiment directory. |

The `--data` directory used by the paired s4/AttnW runners contains:

- `qstamps.npz`: array `q[N,5,H,H]` in primal, g1, g2, x1, x2 order.
- `s4.npz`, `attnw.npz`: exported models.
- `s4_paired.npz`, `attnw_paired.npz`: references in the same source order,
  including `strict_fp32_e`, `strict_fp32_R` and `gpu_T4_dE_dx` for checks that
  use them. See the corresponding runner for additional diagnostic fields.

Run CPU timing, compilation and GPU reference generation sequentially to avoid
resource contention. Repeated-source batch sweeps measure utilization; they do
not increase the number of independent validation sources. Synthetic source-API
checks establish interface consistency, not science accuracy.

## Historical experiments

[The original paired report](GH200_VALIDATION_20261019.md),
[optimization analysis](../OPTIMIZATION_ANALYSIS.md) and
[validation history](../VALIDATION.md) retain measured results and provenance.
Their `runs/` artifacts are local, Git-ignored outputs and are not distributed
with a clone.

One-off launchers and source-rewriting profilers were removed from the current
tree: `ab_round.sh`, `build_optimized.sh`, `finalize_interactive.sh`,
`finish_profile.sh`, `launch_pair.sh`, `tune_variant.sh`, `profile_native.py`,
`tune_profile.py` and `summarize_profile.py`. They hard-coded experiment paths
or depended on superseded kernel source layouts. Recover a historical script
alongside its matching source snapshot from Git commit `cbf2634`, for example:

```bash
git show cbf2634:cpu_src/benchmarks/profile_native.py > /tmp/profile_native_historical.py
```

Current builds use `build_local.sh` or `build_direct.py`; current kernel
profiling uses the shared implementation's `D4CNN_PROFILE` instrumentation.
