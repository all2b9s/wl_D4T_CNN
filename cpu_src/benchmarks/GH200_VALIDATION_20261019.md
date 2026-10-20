# Real-stamp CPU/GPU validation, 2026-10-19

The C++ implementation is 2.59–2.72 times faster than the measured PyTorch
72-process CPU baseline. Its 72-core throughput is **321–345 gal/s**, below the
original **500–580 gal/s** forecast. Parallel efficiency reaches about **83%**,
exceeding the 70% scaling target. Under the user's revised criterion,
**both models pass**: every e/R component has absolute mean bias below 1e-4
against strict-FP32 GPU and Torch CPU. AttnW's individual response outliers
remain documented below as diagnostics.

The interactive allocation was job **3417691**, node **gh009**, account
`bdsp-dtai-gh`, partition `ghx4-interactive`: one GH200 GPU, 72 Grace CPU cores
and 100 GB RAM. The CPU is Neoverse-V2; inference was pinned to cores 144–215
within one CPU domain. Other quarters of the physical node could be shared.
PyTorch was 2.6.0+cu126; native inference used float32 and NEON. Both networks
use 64 x 64 inputs, five trunk layers and 32 channels.
The allocation was explicitly released after 33m19s. GPU rerun, float64
diagnosis and final plot steps completed with exit code 0; accounting is retained.

## Models and inputs

Both checkpoints are `last.pth` from
`models/blended_rdsm_runs/renoised_q_n0594_330k_3m/`:

- `s4`: `Forward8_fixW_CNN`, `nl_sigma=gw_sigma=4`, separate heads.
- `attnw`: `Forward8_AttnW_CNN`, `nl_sigma=16`, shared head, 16 x 16 attention grid.

These are the trained s4/AttnW runs, rather than the legacy sigma=16 fixW file.
Adjacent configurations, resolved constructor defaults and checkpoint/source
hashes are saved with the exported models.

The input contains 512 real blended stamps: 128 randomly selected from each of
the common held-out exposures 940–943, sampling seed 20261019. Training ended
at exposure 934 and validation used 935–939. All backends received the identical
five-plane q-stamps, including the exact training renoise convention
(`noise_std=0.594`, seed 20020620 combined with each sample ID). Reconstructed
q0 matches the cached training preprocessing **exactly**, max absolute error 0.

## Throughput

Rates are galaxies/s for **e plus all four shear responses**, T=2. Each setting
has warmups and three measured repetitions; numbers below are median rates.
Every throughput setting was stable under the runner's max/min <= 1.2 check.

| Backend / cores | s4 | AttnW |
|---|---:|---:|
| C++ / 1 | 5.75 | 5.38 |
| C++ / 8 | 46.23 | 42.92 |
| C++ / 36 | 195.35 | 182.15 |
| C++ / 72 | **344.56** | **321.42** |
| PyTorch CPU / 1 process | 3.75 | 3.67 |
| PyTorch CPU / 8 processes | 28.02 | 27.51 |
| PyTorch CPU / 72 processes | 126.61 | 123.93 |
| PyTorch GPU / strict FP32 | **915.17** | **898.40** |
| PyTorch GPU / original TF32 flags | 1117.27 | 1087.39 |

GPU batch sizes 16/64/128/256 were tested; 256 was fastest in every case.
Strict mode disables both matmul and cuDNN TF32. Original flags allow cuDNN
TF32 and disable matmul TF32. AMP is off; storage dtype is float32 in both modes.
CPU Torch workers are persistent processes pinned to individual cores, each
using one intra-op thread. C++ uses source-level threads and forward duals;
Torch CPU/GPU uses two reverse-autograd contractions. These are measured
implementations of the same requested outputs.

Timings exclude export, loading, q generation, pool startup, file I/O and plots;
they include input/output handling and GPU synchronization. Device-resident GPU
rates are similar: strict s4 914.83, AttnW 894.78 gal/s. These are inference
rates, not full-exposure detection/calibration rates. One-core measurements
use 128 stamps; parallel and GPU measurements use all 512.

72-core C++ speedups over Torch CPU are **2.72x / 2.59x** (s4 / AttnW).
Parallel efficiencies are **83.25% / 83.04%**. Absolute CPU throughput misses
the forecast's lower bound by **31.1% / 35.7%**; it reaches **37.7% / 35.8%** of
the strict-FP32 GPU rate. Single-core throughput also misses the original
7–8 gal/s estimate. The observed gap therefore includes a single-core gap;
it cannot be attributed solely to parallel scaling. No kernel profiler was
run, so a specific compute/cache bottleneck has not been established.

## One-to-one and precision checks

The revised production criterion is `abs(mean(CPU - reference)) < 1e-4` for
each e/R component, averaging over sources only. Maximum component-mean biases
against strict-FP32 GPU are e=4.27e-9/R=1.46e-7 for s4 and
e=1.57e-8/R=1.01e-6 for AttnW: both pass. See `mean_acceptance.json`.

The following **pointwise diagnostic** tolerances retain the original values:
e `atol=1e-6, rtol=1e-5`; R and position responses `atol=1e-6, rtol=1e-4`.
Fractions in the next table count individual scalar outputs, not mean-bias acceptance.

| C++ vs strict-FP32 GPU, 512 stamps | s4 | AttnW |
|---|---:|---:|
| e maximum absolute difference | 5.22e-7 | 1.97e-6 |
| R maximum absolute difference | 7.15e-6 | 1.50e-5 |
| e components passing | 1024/1024 | 1024/1024 |
| R components passing | 2048/2048 | **2046/2048** |

Both one-to-one plots visually follow the identity line. Normalized residuals
are needed to see AttnW's two R outliers; ordinary axis scales obscure them:

| AttnW sample ID | Component | CPU - GPU | Allowed absolute difference |
|---|---|---:|---:|
| 9410000597 | R11 | +7.80e-6 | 3.86e-6 |
| 9430000018 | R21 | -3.05e-6 | 2.20e-6 |

Against Torch CPU, s4 also passes all e/R components. AttnW again has two R
components outside tolerance (maximum absolute difference 1.36e-5). A separate
T=4 check on the first 64 stamps passes all s4 outputs; AttnW passes e/R but has
**1/256 position-response components** outside tolerance (maximum position
difference 5.96e-6). These individual outliers do not fail the revised e/R
component-mean criterion.

The eight most sensitive stamps from each model were rerun in float64 on both
CPU and GPU, including T=4. Maximum R discrepancies drop to **3.22e-15 (s4)**
and **1.35e-14 (AttnW)**; e and position derivatives also pass `atol=rtol=1e-12`.
On the selected AttnW stamps the native float32 result differs from float64 by
up to 8.02e-6 in R, whereas strict GPU float32 differs by at most 7.93e-7.
This supports accumulated float32 numerical error as the cause of these
outliers; identifying the responsible reduction requires further tracing.
Tolerances were not relaxed and production inference was not switched to double.

With original cuDNN TF32 enabled, CPU/GPU R differences rise to **3.86e-3 (s4)**
and **1.43e-3 (AttnW)**. Only 8.20% / 10.69% of R components pass the strict
pointwise diagnostic tolerance. Under the component-mean criterion, s4 versus
TF32 still fails because R11 mean bias is 1.34e-4; AttnW versus TF32 passes,
with maximum R component-mean bias 3.55e-5. AttnW's TF32 GPU R also varies across repeated GPU
calls (max observed repeat difference 6.97e-5); strict-mode repeat differences
remain within tolerance. Storing float32 tensors alone therefore does not
establish equivalent FP32 arithmetic. These comparisons do not establish
whether TF32 or the remaining native roundoff meets a science m/c requirement.

## Artifacts and reproduction

All artifacts are in [`../runs/paired_3417691/`](../runs/paired_3417691/):
`results.json`, `diagnosis.json`, `data.json`, model exports, q-stamps, raw paired
predictions, float64 subsets, hardware information and PNG/PDF figures.
The `figures/` directory includes one-to-one, residual, normalized-residual and
throughput figures. Out-of-tolerance points are marked red.

The initial run stopped on an auxiliary assertion comparing two GPU input
paths under TF32, after CPU timing had completed. Its logs remain in
`initial.log`/`initial.err`. The runner now records this comparison as a
diagnostic; it preserves the original output tolerances. `--resume-cpu` reused
saved CPU results and inputs within the same allocation, and repeated the full
GPU phase successfully (`gpu.log`/`gpu.err`). No failed measurement was silently
counted as a numerical pass.

The one-off launcher used for this historical run is retained in Git commit
`cbf2634`. Current runner conventions are in [README.md](README.md). Run
`python -m cpu_src.benchmarks.diagnose_pair --output OUTPUT` in the allocation
for float64 outlier checks. `cluster_pair --plots-only` redraws saved predictions
without repeating inference. No existing `src/` code was modified.
