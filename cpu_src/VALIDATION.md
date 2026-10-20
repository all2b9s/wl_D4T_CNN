# x86 port status — 2026-10-20

AVX2/AVX512 now have F4 (default) and F2 transformed convolution, fused C32
LayerNorm/GELU/residual propagation, and first-layer C1 skip fusion. ISA dispatch
remains runtime checked; the generic build requires no global AVX flags.
ARM and x86 share weight transformations. Direct, scalar, FP64 and trace paths
remain available. Build entry points include the new x86 translation unit and
the direct builder tracks its shared `.inc` dependency.

**This port has not been compiled, tested or benchmarked.** Per the user's
instruction, validation is delegated to the user on Delta. The temporary ARM
allocation 3419652 was released; it ran only environment discovery. No numerical
or performance claims for x86 follow from the Grace results below. The installed
ARM binary has not been replaced for this source change.

Prepared coverage: `tests/test_x86.cpp` (standalone/CTest `x86_math`) compares
all supported x86 kernels against independent double references; the Python C32
model regression now iterates over all supported ISAs and direct/F2/F4/default
modes. Commands and build prerequisites are in [README.md](README.md).

---

# CPU optimization round 2 — 2026-10-19

This section supersedes the historical validation below. All builds and numerical/
performance runs used interactive Slurm allocation **3419259**, **gh151**, one
72-core Grace domain (CPU affinity 0–71) and one GH200. CPU timings and compilation/
reference runs were sequential. No heavy tests ran on the login node.
The allocation was explicitly released after verification. Its CANCELLED state
records this release; final build, reference and installed-check steps 7–9 all
completed with exit code 0. Step 2 is the documented initial F4 operator-bound
failure. Full accounting is retained in `runs/tune2_3419259/accounting.txt`.

The installed API-v2 extension now defaults to **F(4x4,3x3)** for NEON FP32 C32
hidden convolutions. `D4CNN_CONV=winograd` retains F2, `direct` retains direct
convolution, and `winograd4` explicitly requests F4. Unsupported forced modes
raise errors. Scalar, other channel counts and FP64 keep their existing paths.
F4 weights are transformed once with double accumulation and stored as FP32;
convolution, activations and tangent propagation remain FP32.

LayerNorm statistics for all dual directions are computed before overwriting any
input. Normalization, GELU and residual arithmetic then stay in registers until
the final store. The first-layer C1 skip projection is included, eliminating its
allocated tensor. This fusion produces bit-identical outputs to the unfused F4
variant on all 512 validation sources for both models.

## Same-node speed measurements

Median of three runs after warmup, T=2, production models and identical q-stamps.
Includes the Python API/input validation/output assembly; excludes loading,
detection and q-stamp preparation. 1 core uses 32 sources; 72 cores use 512.
These are CPU measurements, not GPU measurements.

| Model | Previous 1 core | New 1 core | Previous 72 cores | New 72 cores | 72-core gain |
|---|---:|---:|---:|---:|---:|
| s4 | 10.82 | 15.99 | 642.05 | 947.20 | 47.5% |
| attnw | 9.92 | 14.17 | 577.08 | 801.73 | 38.9% |

All rates are galaxies/s. Final 8/36-core rates are 126.13/536.29 for s4 and
111.89/463.51 for AttnW. At 2304 repeated sources, rates reach 1028.67/874.20;
this is a batch utilization experiment, not an increase in independent samples.
All final scaling and batch timing sets meet the existing <20% spread criterion.
The synthetic source API including q preparation reaches 933.52/791.92 with auto
batching. Its T=4 e/R/position derivatives are bit-identical to explicit q-stamps.

Ablation (72 cores, 512 sources): baseline 642.05/577.08 → F4 865.75/744.34 →
first-layer skip fusion 889.35/759.10 → full epilogue fusion 948.85/798.39.
Final default-path rerun is shown in the table above. Instrumented baseline
profiling identified ~52 ms/source in transformed convolution multiplication,
~20 ms in the epilogue, and ~6.6 ms AttnW attention on one core. Nested timers
are not additive and instrumentation timings are not throughput benchmarks.

## Numerical acceptance and regression coverage

The scientific criterion is unchanged: each e/R component must satisfy
`abs(mean(CPU - strict_FP32_GPU)) < 1e-4`, with finite outputs. The 512 held-out
real sources and model exports are unchanged. T=4 also passes on 64 real sources;
T=2 and T=4 e/R are bit-identical. Maximum absolute component mean biases against
the saved strict GPU reference are:

| Model | e mean bias | R mean bias | Largest individual R error |
|---|---:|---:|---:|
| s4 | 6.766e-09 | 1.802e-07 | 6.199e-06 |
| attnw | 1.144e-08 | 9.770e-07 | 1.359e-05 |

Mean acceptance does **not** imply every source passes the older pointwise
relative/absolute tolerance. Both models have individual R failures under that
diagnostic; the s4 position-response diagnostic also has pointwise failures.
Full signed component means, maxima, RMS and p99 errors remain in the JSON files.
These kernel checks do not replace exposure-level m/c science calibration.

- 14 Python tests pass, covering independent autograd, D3/D5, trace/fallbacks,
  concurrent callers, reusable workspaces, source API and pipeline integration.
- C++ tests pass for executor reuse/exception recovery, GELU, fused epilogues
  with identity/projected skips, and F2/F4 convolution against double direct sums
  across dimensions 1,2,3,4,5,8,9,16 and both dual widths.
- The F4 transform identity is checked exactly using rational coefficients.
- F4's less well-conditioned transform exceeded the inherited F2 operator-only
  bound in the first experiment (2.226e-6 vs 2e-6). Its N(0,.1) fixture bound is
  explicitly 1e-5; F2 stays at 2e-6. No model e/R mean threshold was relaxed.
- FP64 golden outputs and primal traces pass the unchanged 1e-12 tolerance:
  maximum e error 7.49e-16 and R error 7.11e-15.

The fresh strict-FP32 GPU rerun (TF32 disabled for matmul and cuDNN) also
passes every e/R component. Maximum mean e/R biases are 6.62e-9/1.83e-7 for
s4 and 1.19e-8/9.87e-7 for AttnW. Maximum individual R errors are 6.68e-6
and 1.41e-5. See `reference.json` for signed means and diagnostics.
The installed repository binary passed a fresh-process empty-batch check and
reproduced all 512 saved e/R outputs bit for bit, with F4 metadata verified.
See `installed.json` for pointwise pass counts.

## Reproduction and artifacts

- [Final build, raw timings and numerical outputs](runs/finalize_3419259/final/)
- [Same-node baseline, ablations and profiler](runs/tune2_3419259/)
- [Experiment notes, including failed/duplicate attempts](runs/tune2_3419259/experiment_notes.json)
- Final source archive, compiler metadata and source/binary hashes are retained.
  The old installed extension is saved as `previous_native.so`.
- `benchmarks/tune_variant.sh TAG winograd4` builds an isolated variant inside an
  existing allocation; `benchmarks/finalize_interactive.sh final` performs final
  testing. `finalize_sources.py` and `finalize_reference.py` run in subsequent
  sequential steps. `tune_profile.py` instruments the saved **F2 baseline**;
  its exact-pattern assertions intentionally reject changed implementations.

Installed binary SHA256: `7c0285350592eb5efb254ee7f6629a4e6202807433ac2bfe529bf2b6dab841a4`.

---

# Historical round 1 validation (superseded above)

# Current CPU implementation and validation: 2026-10-19

## Installed, tested build

The repository extension has been atomically replaced with the validated native
API-v2 build. The former Python/native `convolution_algorithm` mismatch is fixed;
loading an outdated extension now produces an actionable rebuild error at model
construction. A fresh process importing the installed repository package passed
the empty-batch check and reproduced **all 512 saved e/R outputs bit for bit** for
both production models. The previous binary is retained in the run directory.

All compilation and numerical/performance work ran in interactive allocation
**3418647**, `ghx4-interactive`, node **gh132**, 72 allocated Grace cores and one
GH200 GPU. This was one CPU domain, not an exclusive 288-core node. Substantial
work was not run on the login node. CPU, reference and compilation steps were
sequential so compilation/GPU-reference workloads did not overlap CPU timing.
All eight job steps finished with exit code 0. The allocation was explicitly
released after artifact collection; its final CANCELLED allocation state records
that release, not a failed test. See `allocation.txt` and `accounting.txt`.

Implemented and exercised:

- NEON vector erf-GELU and C32 LayerNorm, scalar/FP64 fallbacks, no global
  fast-math or external vector-math-library dependency.
- C32 FP32 Winograd F(2x2,3x3), tile-output LN/GELU/residual fusion, and an explicit
  direct-convolution override. Scalar ISA now also selects scalar nonlinear math.
- Persistent executors per calling thread, worker affinity within the caller's
  allowed CPU mask, worker-local reusable trunk/canonical buffers, and reuse of
  AnaCal ImageQLease caches across source batches. Concurrent callers have separate
  executors. Changing thread count or caller affinity recreates an executor.
- fixW direct orbit pooling; AttnW query tiling, NEON QK/AV and direct scalar-token
  accumulation for divisible grids. Nondivisible grids, trace and FP64 preserve
  the full-resolution path. Attention and pooling propagate every tangent.
- Automatic source batch size `max(64,8*nthreads)`; explicit sizes remain valid.
- C32 full-model tests, worker/workspace reuse and exception recovery, native API
  version checks, plus a registered CTest math/executor target.

## Fresh strict-FP32 GPU comparison

Same 512 held-out real q-stamps from exposures 940–943 as the earlier study;
same trained `s4` and `attnw` model exports. The GPU reference was rerun in this
allocation with both matmul and cuDNN TF32 disabled. Differences below are
**signed means over sources**, accumulated in float64. Each component must have
absolute bias strictly below `1e-4`.

| Component | s4 CPU − GPU mean | AttnW CPU − GPU mean |
|---|---:|---:|
| e1 | -5.2061e-9 | +1.0795e-8 |
| e2 | -5.0236e-9 | +4.0208e-9 |
| R11 | +1.6074e-7 | +9.9341e-7 |
| R12 | +1.4685e-8 | -6.2545e-8 |
| R21 | -1.3596e-8 | -1.3114e-8 |
| R22 | +7.7409e-8 | +9.0605e-7 |

**Both models pass every component.** All outputs are finite. The retained
pointwise diagnostics pass 1024/1024 e and 2048/2048 R entries for s4; AttnW
passes 1024/1024 e and 2045/2048 R entries. Thus mean acceptance does not imply
every individual response meets the older pointwise tolerance. Maximum absolute
R errors are 6.91e-6 and 1.50e-5, respectively.

T=4 was checked on 64 real sources against the saved independent reference;
e/R means pass and T=2/T=4 e/R are bit-identical. Position-response diagnostics
also pass. The production AttnW FP64 golden check passes both outputs and all
recorded primal traces: maximum e error 7.49e-16, R error 7.11e-15.

## CPU throughput

Galaxies/s for e and all four shear responses, FP32, T=2, 64x64, 5 layers,
32 channels. Warm-up precedes three repetitions; rates use median wall time.
The Python API/input validation/output handling are included. Model loading,
q preparation, detection and catalog/calibration I/O are excluded.
Single-core tests use 32 sources, 8-core tests 128, and 36/72-core tests 512.

| Backend / cores | s4 | AttnW |
|---|---:|---:|
| Current C++ / 1 | 10.81 | 9.93 |
| Current C++ / 8 | 85.44 | 78.39 |
| Current C++ / 36 | 367.09 | 332.71 |
| Current C++ / 72 | **642.42** | **579.25** |
| Current C++ direct conv / 72 | 517.69 | 479.46 |
| Fresh Torch CPU / 1 process | 3.79 | 3.68 |
| Fresh Torch CPU / 72 processes x 1 thread | 120.35 | 121.10 |

Current C++ is **5.34x / 4.78x** faster than the same-node Torch 72-process
baseline. Parallel efficiency is 82.5% / 81.0%, above the 70% scaling gate.
Winograd improves full-model 72-core throughput by about 24.1% / 20.8% relative
to this implementation's direct override. All final timing series met the
max/min <1.2 stability criterion. These are full-model measurements, not evidence
for the original plan's separately defined 20 gal/s single-core trunk gate.

| 72-core batch size | s4 | AttnW |
|---|---:|---:|
| 576 | 685.06 | 615.03 |
| 1152 | 687.12 | 616.90 |
| 2304 | **695.93** | **626.49** |

Larger batches cycle the original 512 sources; they do not add independent
validation samples. The 512-source measurement is the primary comparable rate.
The original 500–580 gal/s forecast is reached for both models at 512 sources;
no extrapolation to 288 cores is claimed.

The pre-fix Winograd source, rebuilt at the start of this session, measured
632.75 / 556.18 gal/s at 512 sources. The workspace/token/pool stage measured
646.11 / 578.38; final affinity/attention changes measured 642.42 / 579.25.
These small successive differences include timing drift and do not isolate
each individual change's contribution. Historical 321–345 gal/s results refer
to the older direct/scalar-math implementation, not this session's starting
Winograd source.

## Source API and tests

13 Python unittest cases passed in the isolated build and again after installing
the binary in the repository. Coverage includes C32 direct/Winograd, T=2/4,
shared/separate heads, odd and nondivisible AttnW sizes, cross-model workspace
reuse, concurrent calls, Gaussian maps, native q-stamps and the torch-free
synthetic FITS detection/measurement/calibration flow. C++ GELU/Winograd tests and
executor exception/recovery checks passed; CTest reports 1/1 passing target.

A separate synthetic 384x384 canvas with 512 non-edge sources checks the source
adapter. T=4 `measure_sources` e/R/dE_dx equal prepared-q inference bit for bit.
Including FFT/q preparation, automatic batch=576 measured 627.49 / 558.86 gal/s,
versus 612.64 / 534.15 for batch=64. These synthetic rates are separate from the
real-stamp table and do not include detection or calibration.

Row-streaming remains an optional fallback: measured scaling already exceeds
the plan's trigger threshold. x86-specific vector math/performance, full-node
NUMA scaling, real-exposure m/c and E2 reference-point science, and AnaCal Task
migration remain separate work. This validation establishes numerical mean
agreement and CPU speed, not a science multiplicative-bias bound.

## Reproducibility

Artifacts: [`runs/finalize_3418647/`](runs/finalize_3418647/).

- `final/results.json`: all repeated CPU timings, e/R means, T4 checks, source,
  model, q-data and native-binary hashes.
- `final/reference.json`: fresh Torch CPU timings and strict GPU mean checks;
  `*_fresh_gpu.npz`, `*_T2.npz`, `*_T4.npz`: raw paired outputs.
- `final/golden64.json`, `final/source_api.json`, `final/final_direct.json`.
- `final/source.tar`, `final/compiler.json`, native binary and previous binary;
  `stage1/`: preceding optimization stage.
- `initial.log`, `final.log/.err`, `reference.log/.err`, `source_checks.log/.err`,
  `installed.log/.err`: complete build, regression, CTest and installed checks.

Use `benchmarks/finalize_interactive.sh TAG` under `srun` in an allocated
interactive job. `finalize_reference.py` runs independent CPU/GPU references;
`finalize_sources.py` checks source batching. Results are Git-ignored and must
be retained separately when sharing a build.

---

## Historical local validation (before the current optimization/install)

Current float32 acceptance, as requested by the user: each e/R component must
satisfy `abs(mean(CPU - reference)) < 1e-4` over matched sources. Individual
outliers and position derivatives are diagnostics. Both trained models pass
against strict-FP32 GPU and Torch CPU on the 512 real stamps. Maximum absolute
R-component mean biases against GPU are **1.46e-7 (s4)** and **1.01e-6 (AttnW)**.
The default TF32 s4 comparison still fails: R11 mean bias is **1.34e-4**.
AttnW versus default TF32 passes (maximum R mean bias **3.55e-5**).
Full reassessment is in `runs/paired_3417691/mean_acceptance.json`. Three tests
cover the strict threshold, source versus component averaging and invalid data.

Environment: aarch64, GCC 14.2, Python 3.12.11, pybind11 3.0.1, AnaCal 0.8.2.
The complete native extension compiled successfully both directly with GCC and
through CMake. Existing `src/` and AnaCal files were not modified.

Eight unittest cases passed, including subcases covering:

- AttnW/fixW, shared/separate heads, fp32/fp64, T=2/T=4, scalar/NEON,
  odd AttnW sizes, nondivisible adaptive pooling and channel tails.
- Nonzero attention-kernel weights, four-direction finite differences,
  Gaussian maps at sizes 64/62/48/63 and sigmas 4/8/16, taper on/off.
- Concurrent model calls, invalid inputs, model metadata and weight rejection.
- Native AnaCal q-stamps against Python AnaCal, with and without noise.
- A subprocess blocking all torch/torchvision imports while executing a complete
  synthetic FITS measurement and g1 calibration pipeline; actual detection was
  also exercised, not only truth-position measurement.
- Reference-shift round trips and identical-run paired science comparison.

Production checkpoint used for network comparison:
`models/blended_rdsm_runs/renoised_q_n0594_330k_3m/attnw/last.pth`, with its actual
adjacent configuration. Four deterministic asymmetric **synthetic** q-stamps,
64 x 64, were compared against independent PyTorch reverse-mode autograd.

| Precision | Maximum absolute e error | Maximum absolute R error | Maximum absolute position-response error |
|---|---:|---:|---:|
| float32 | 1.02e-6 | 1.05e-5 | 1.42e-6 |
| float64 | 7.50e-16 | 7.11e-15 | 1.48e-15 |

Both precision modes pass their documented combined absolute/relative output
tolerances. Intermediate primal traces pass their separate activation tolerance.
Reports and exported/golden artifacts are under `cpu_src/runs/` and are ignored
by Git. The example pipeline and detailed commands are in `README.md`.

These results establish numerical and functional behavior, not scientific
accuracy on real exposures. Exclusive-node throughput/scaling, x86 ISA execution
and paired real-exposure m/c/reference-point experiments remain external
validation work. The 512-real-stamp and one-CPU-domain scaling checks are now
reported below. Local timing is explicitly labeled a smoke
benchmark and includes stability checks; it is not a node-level speedup claim.

The actual legacy `F8_gauss_truth_final.pth` checkpoint was also exported with
explicit `nl_sigma=gw_sigma=16` and compared in both precisions against autograd
on one 64 x 64 stamp (all four directions). All checks passed:

| Precision | Maximum absolute e error | Maximum absolute R error |
|---|---:|---:|
| float32 | 8.29e-08 | 1.04e-06 |
| float64 | 3.33e-16 | 1.28e-15 |

A persistent end-to-end example using the exported production AttnW checkpoint
is available at `cpu_src/runs/demo/config.json`. Its `demo` command was executed
with torch/torchvision imports actively blocked, including real detection and
calibration of two generated synthetic FITS exposures.

The CMake-installed package was imported in a fresh interpreter outside the
repository. Network inference matched the saved fp32 reference, and native
source measurement ran with Python imports of torch, torchvision and anacal
blocked. The full pipeline separately requires the Python AnaCal detection API.

## Paired real-stamp GH200 / Grace validation

Interactive Slurm job 3417691 on gh009 tested the trained **s4** and **AttnW**
`last.pth` checkpoints on 512 common held-out real stamps, with exactly matched
q0 preprocessing. All network inference used float32; strict GPU runs disabled
TF32. Three repeated timings were stable for every setting.

| Model | C++ 72-core gal/s | Torch CPU 72x1 gal/s | GPU strict FP32 gal/s | C++ parallel efficiency |
|---|---:|---:|---:|---:|
| fixW s4 | 344.56 | 126.61 | 915.17 | 83.25% |
| AttnW | 321.42 | 123.93 | 898.40 | 83.04% |

The 70% parallel-efficiency target passes. The original absolute CPU estimate
of 500–580 gal/s is **not reached**. C++ is 2.72x / 2.59x faster than the same-node
Torch CPU baseline. These are inference timings, excluding detection and q
preparation; this allocation owned one CPU/GPU domain, not the entire node.

s4 passes every e/R output tolerance against strict-FP32 GPU and Torch CPU.
AttnW passes every e component but has **2/2048 R components** outside tolerance
against each reference. Its separate T=4 check also has **1/256 position-response
components** outside the original pointwise diagnostic tolerance. Under the
revised component-mean acceptance above, both models pass e/R. Float64 CPU/GPU reruns of
the eight most sensitive stamps per model agree to about 1e-14, supporting a
float32 accumulation discrepancy. Enabling the original cuDNN TF32 flag makes
CPU/GPU differences substantially larger for both models.

See the [full measured report](benchmarks/GH200_VALIDATION_20261019.md) for
outlier sample IDs, all scaling rates, TF32 comparisons, methodology and artifact
paths. Raw results and PNG/PDF one-to-one/residual figures are in
`cpu_src/runs/paired_3417691/`.

## Performance diagnosis and isolated optimization experiments

Interactive job **3418136**, gh051, used one 72-core CPU domain and the same
512 real q-stamps. The allocation has been released. Detailed evidence and
implementation priorities are in [OPTIMIZATION_ANALYSIS.md](OPTIMIZATION_ANALYSIS.md).

| Model | Baseline 1 core | Prototype 1 core | Baseline 72 cores / 512 | Prototype 72 cores / 512 | Prototype 72 cores / 2304 |
|---|---:|---:|---:|---:|---:|
| fixW s4 | 5.545 | 8.523 | 327.307 | 469.926 | 514.737 |
| AttnW | 5.169 | 7.747 | 306.266 | 433.728 | 477.532 |

Rates are gal/s, median of three repeats. Single-core timing used 32 sources.
The 2304-source batch cycles the original 512 inputs; it does not add independent
precision-validation data. Use same-node baselines: this node's original binary
was about 5% slower than the earlier gh009 result.

The prototype combines vector erf-GELU, C32 vector LayerNorm and removal of a
hidden-layer skip copy. All 11 throughput variants passed the component-mean
e/R limit. For the combined prototype, maximum absolute component-mean e/R
biases were **4.33e-9 / 1.22e-7 (s4)** and **1.85e-8 / 9.70e-7 (AttnW)**.
Additional T=4 validation on 64 sources passed; T=2/T=4 e/R were bit-identical.

This is an isolated network prototype, without an AnaCal-linked rebuild or the
full generic-configuration test matrix. It has not replaced the installed
extension. The existing generic/double paths are retained in the experiment,
but production integration still needs its platform/math-edge-case checks.
No new end-to-end m/c validation is claimed.

Saved artifacts under `runs/profile_3418136/` include all repeated timings,
predictions, stage profiles, PMU counters, assembly, PNG/PDF figures, and exact
experimental source/binary snapshots in `experimental_builds.tar.gz`.
