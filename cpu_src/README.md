# D4CNN CPU demo

Standalone CPU inference and blended-image calibration, isolated under `cpu_src/`.
The default is **float32 throughout** (normalization, CNN, attention, pooling and
heads), matching the current PyTorch precision. Float64 is a validation mode.
Both `Forward8_AttnW_CNN` and `Forward8_fixW_CNN` are supported. Existing `src/`
and the AnaCal checkout are read/reused, never patched by this package.

The current implementation includes NEON vector GELU/LayerNorm, C32 Winograd,
fused Winograd epilogues, persistent workers and reusable per-worker buffers.
fixW pools each orbit directly. AttnW uses query tiles, vector QK/AV and directly
accumulates low-resolution scalar tokens when the grid divides the image size.
See [current validation](VALIDATION.md) for measured acceptance and throughput.
The [earlier performance audit](OPTIMIZATION_ANALYSIS.md) retains historical
baseline/ablation results; those numbers are not measurements of the current code.

## Build

Use the same Python interpreter and AnaCal source revision as your detection
installation. NumPy and pybind11 must be available. Run from `single_galaxies/`:

```bash
PYTHON_BIN=/u/wenyinli/.conda/envs/anacal_clone/bin/python \
ANACAL_SOURCE_DIR=/projects/bdsp/wenyinli/packages/AnaCal \
bash cpu_src/build_local.sh
```

`CPU_CXX` selects a direct C++17 GCC/Clang compiler (default `g++`), and
`BUILD_JOBS` defaults to 2. The helper uses `cpu_src/.build-tmp/`, so a full `/tmp`
does not prevent compilation. Do not use the Cray `CC` wrapper unless required
by your deployment. A separate CMake build directory is needed when changing
compilers after initial configuration.

On filesystems where CMake probe creation is very slow, add
`D4CNN_DIRECT_BUILD=1` to the same command. This uses the same compiler flags,
parallel compilation and cached object files, without CMake probes. Both build
paths install the extension atomically.
The Python API checks native API version 2 at model construction and reports a
rebuild instruction if an old extension is present. Always restart interpreters
that loaded an older extension after rebuilding it.

Without `ANACAL_SOURCE_DIR`, the q-stamp model API still builds; native image
measurement raises a clear error. With it, the extension compiles AnaCal's
`src/image.cpp` and reuses its ImageQ/ImageQLease/pocketfft implementation. The
extension does not register AnaCal classes again or link to its Python binary.

For wheel installation, `pyproject.toml` also supports scikit-build-core:

```bash
TMPDIR="$PWD/cpu_src/.build-tmp" python -m pip install ./cpu_src \
  -Ccmake.define.ANACAL_SOURCE_DIR=/path/to/AnaCal
```

The full pipeline additionally uses the existing repository's `src` package,
AnaCal, astropy, fitsio, pandas, scipy, tqdm and galsim. Run from this repository
root or add it to `PYTHONPATH`. PyTorch/torchvision are needed only for export,
golden data, reference comparisons and reference tests.

## Export a model

```bash
python -m cpu_src export \
  --checkpoint models/blended_rdsm_runs/renoised_q_n0594_330k_3m/attnw/last.pth \
  --output cpu_src/runs/attnw.npz
```

The exporter reads the adjacent `config.json`, resolves constructor defaults,
checks weight keys/shapes and records checkpoint/source hashes and preprocessing.
Conflicting explicit parameters are rejected. For old fixW checkpoints, supply
architecture, explicit sigmas and preprocessing, rather than guessing defaults:

```bash
python -m cpu_src export \
  --checkpoint models/F8_gauss_truth_final.pth \
  --architecture Forward8_fixW_CNN \
  --model-kwargs '{"nl_sigma":16,"gw_sigma":16}' \
  --pixel-scale 0.2 --sigma-arcsec 0.3609341825902335 --flim 10 \
  --output cpu_src/runs/fixw.npz
```

Other architecture parameters must match the actual trained checkpoint. Export
preserves each tensor's stored precision, including `kernel_blur`; runtime casts
to the selected model precision. NPZ loading is pickle-free. Do not pass
untrusted pickle-based PyTorch checkpoints to export.

## Python API

```python
from cpu_src import Model

model = Model.load("cpu_src/runs/attnw.npz", precision="float32", isa="auto")
result = model.measure_qstamps(q, n_tangent=2, nthreads=1)
# q: [N,5,H,H] in v,g1,g2,x1,x2 order
# result.e: [N,2], result.R: [N,2,2], R[n,i,j] = de_i/dg_j

result4 = model.measure_sources(
    image, psf, positions_xy, noise=noise_canvas,
    n_tangent=4, nthreads=8, batch_size=64,
)
# positions_xy: [N,2] in x,y order, pixels
# result4.dE_dx: [N,2,2], position derivatives per arcsec
# result4.centers_xy: actual rounded integer stamp centers
```

Outputs use the selected precision. Catalog serialization preserves the existing
float64 catalog columns. T=2 computes two shear directions; T=4 additionally
computes two position directions. Their e/R at the same reference point agree.
The default reference is the **rounded stamp center**, as in the existing ML
pipeline. Position derivatives are not silently used to recenter shear responses.

For an explicit AnaCal reference shift, use:

```python
shifted = result4.shift_reference([dx_arcsec, dy_arcsec])
```

This applies `g1 += dx*x1 - dy*x2`, `g2 += dx*x2 + dy*x1` independently to e1/e2.
A positive shift follows AnaCal's signed `shift_reference` convention. To combine
with selection qnumbers, first establish the same reference and coordinate units.
T=2 cannot perform this shift. This helper is experimental science functionality;
the default pipeline deliberately retains the current T=2 convention.

Images/noise are float32 at the AnaCal boundary; PSF and q-stamp generation are
float64, matching current AnaCal. Network input is then cast to model precision.
Only non-edge sources are accepted. Centers use `floor(position+0.5)`. Models fix
one square stamp size at construction. The AttnW core supports odd sizes, whereas
legacy fixW and native FFT source measurement require even sizes.

A model owns immutable native weights and is safe for concurrent calls. Each
calling thread has an independent persistent executor; workers retain private
scratch and AnaCal leases across batches. Linux workers request affinity to
individual cores within the caller's allowed mask. Changing the thread count or
affinity recreates the executor. `nthreads` parallelizes sources, not a single
source. Avoid nesting another CPU pool around a heavily threaded call; use one
process per NUMA domain. Omitted/JSON-null `batch_size` resolves to
`max(64, 8*nthreads)` (576 at 72 threads). Explicit batch sizes remain supported.
The source adapter batches q preparation and inference instead of allocating
all q-images for an exposure.

## Complete blended pipeline

For a runnable example without external FITS data (after model export):

```bash
python -m cpu_src demo --model cpu_src/runs/attnw.npz --output cpu_src/runs/demo
```

This creates two small synthetic shear exposures, a delta PSF, truth/reference
FITS tables and a configuration, then runs actual detection, C++ measurement
(T=4) and g1 calibration. Setup and execution are torch-free. Choose an empty
output directory. The tiny sample demonstrates the pipeline, not science bias.


Copy `cpu_src/configs/blended.example.json`, fill in dataset paths and choose a
new output directory. Relative paths resolve against the config file. Then:

```bash
python -m cpu_src measure --config /path/to/demo.json
python -m cpu_src calibrate --config /path/to/demo.json
# Or both sequentially:
python -m cpu_src run --config /path/to/demo.json
```

The pipeline reads FITS science/variance/truth, creates a reproducible renoise
canvas, runs existing AnaCal detection/FPFS, applies weight/edge cuts, measures
with native q-stamps and C++ inference, and writes compatible catalogs. It uses
the same noise canvas for detection and ML, with seed `19491001 + exposure_id`.
Catalogs keep the existing `g1_0`, `g1_1`, `g1_2` directory conventions; position
responses and model metadata go into separate `_response.npz` files. Timing and
configuration go into `<fname>_manifest.json`.

The FITS/truth layout follows the existing DC1 pipeline, including its +1500-pixel
truth-position offset and PSF border trimming. This is a dataset-specific demo,
not an automatic FITS-schema detector. Existing files at a chosen output prefix
are replaced on rerun; choose separate output roots for paired backends.

`sim_mode0` and `sim_mode40` are required for g1 calibration; `sim_mode80` alone
supports measurement only. g2 calibration is unavailable in these exposures and
is JSON `null`. `mag_cut` and `include_detection_response` use the existing
selection and detection calibration definitions. The pipeline does not replace
hard-cut selection response with raw flux derivatives.

Use `backend: "torch"` and a separate output root/prefix for a paired reference
run; `reference_device` can be `cpu` or `cuda`. This optional path loads torch,
while detection/noise/q-stamps remain shared implementations.

## Numerical and science validation

```bash
python -m unittest discover -s cpu_src/tests -v

# Small deterministic synthetic check; use --qstamps for real blended samples.
python -m cpu_src golden --model cpu_src/runs/attnw.npz \
  --count 4 --precision float64 --output cpu_src/runs/golden64.npz
python -m cpu_src validate --model cpu_src/runs/attnw.npz \
  --golden cpu_src/runs/golden64.npz --precision float64
python -m cpu_src golden --model cpu_src/runs/attnw.npz \
  --count 4 --precision float32 --output cpu_src/runs/golden32.npz
python -m cpu_src validate --model cpu_src/runs/attnw.npz \
  --golden cpu_src/runs/golden32.npz --precision float32
```

Golden data uses independent reverse-mode autograd contraction for all four
input directions. Four samples also include primal intermediate dumps for
normalization, trunk blocks, D4 fields and attention kernels. Native `trace(q)`
returns primal and tangent traces as `[1+T,pixels,channels]` for diagnosis.
Finite differences test derivatives independently. Attention fixtures have
nonzero kernel-head weights so a detached attention branch cannot pass unnoticed.

Production float32 acceptance uses the user-specified criterion:
`abs(mean(CPU - reference, axis=0)) < 1e-4` for **each** e and R component.
Sources are averaged; e1/e2 and the four R components are never averaged together.
The difference is accumulated in float64. Nonfinite or empty comparisons fail.
Individual-output checks (e `atol=1e-6, rtol=1e-5`; responses
`atol=1e-6, rtol=1e-4`), position derivatives and intermediate activations remain
diagnostics in float32. Float64 validation still uses strict pointwise
`atol=rtol=1e-12` and checks intermediate traces. The CLI exits nonzero when its
applicable acceptance criterion fails. Golden files are tied to the exported
model hash. Synthetic fixtures validate numerics, not science accuracy.

Compare paired catalogs and full science runs:

```bash
python -m cpu_src compare /left/catalog_0.npz /right/catalog_0.npz
python -m cpu_src compare-science /left/cpu_manifest.json /right/torch_manifest.json \
  --repeats 200 --output cpu_src/runs/science_comparison.json
```

The science comparison checks identical detections/selection inputs, recomputes
m/c with the current selection convention and bootstraps paired exposures using
a shared seed. It reports delta m/c, intervals and the `abs(delta_m)<1e-4` target.
Small samples explicitly report insufficient science evidence; numerical tests
alone do not establish a multiplicative-bias bound. E2 can use `dE_dx` and the
explicit shift helper to study reference conventions separately.

## Performance

```bash
python -m cpu_src benchmark --model cpu_src/runs/attnw.npz \
  --count 16 --threads 1 2 4 8 --repeats 3 \
  --output cpu_src/runs/benchmark.json
```

Use `--backend torch --threads 1` for the same-checkpoint CPU autograd baseline.
It reuses the loaded model and processes one source per forward/backward call;
its thread count means PyTorch intra-op threads, whereas C++ threads distribute
sources. No checkpoint loading or export time is included in either path.

The report includes warm-up, repeated full-network timings, ISA, affinity, peak
RSS and parallel efficiency. It includes Python input validation and excludes
q-stamp creation; pipeline manifests report full exposure timing. Set `--isa
scalar`, `neon`, `avx2` or `avx512` to compare supported kernels; unavailable ISAs
are rejected. `D4CNN_ISA` overrides `auto`. Float64 currently uses scalar kernels.
NEON and x86 implementations share each loaded weight across primal/tangents.
The x86 kernels are runtime-dispatched with CPU feature detection. On NEON, AVX2 and AVX512,
FP32 C32 hidden layers default to Winograd F(4x4,3x3); `D4CNN_CONV=direct`
selects direct convolution and `D4CNN_CONV=winograd` retains F(2x2,3x3).
`D4CNN_CONV=winograd4` explicitly selects F4. Both forced Winograd modes reject unsupported precision,
ISA or channel counts. C32 blocks fuse LayerNorm, GELU and residual arithmetic
and the first-layer C1 skip projection on all three SIMD backends.
FP64 and `isa="scalar"` use scalar math. SIMD GELU
approximates erf and exp locally; there is no global fast-math, FTZ/DAZ, external
vector-math ABI, BLAS, OpenMP, row-streaming or AnaCal Task modification.
The result metadata records the selected convolution algorithm.

The x86 F4/F2 and fused epilogues were ported on 2026-10-20. They have **not
been compiled, numerically validated or benchmarked in this change**; validation
will be performed by the user on Delta. The Grace numbers in VALIDATION.md do
not establish x86 performance. Rebuild the extension on the target x86 machine;
the checked-out aarch64 extension cannot be used there.

Inside a Delta interactive allocation, from the repository root (with an x86
Python environment and GCC/Clang):

```bash
python -m cpu_src.build_direct
# Standalone tests need no Python/NumPy/Torch or AnaCal at execution time.
g++ -std=c++17 -O3 -pthread -I cpu_src/include \
  cpu_src/tests/test_x86.cpp cpu_src/native/x86.cpp \
  cpu_src/native/convolution.cpp cpu_src/native/kernels.cpp -o /tmp/d4cnn_test_x86
/tmp/d4cnn_test_x86
python -m unittest discover -s cpu_src/tests -v
```

The standalone test exercises every supported x86 ISA, F2/F4, T=2/T=4,
boundaries and projected/identity residuals against independent scalar double
calculations. CTest registers it as `x86_math` and marks it skipped on ARM.
The full Python C32 test explicitly covers every available ISA and each
convolution override, including default F4. For AnaCal-dependent tests/builds,
set `ANACAL_SOURCE_DIR` to the AnaCal checkout before building.

Use `D4CNN_ISA=avx2` or `avx512` to force an available backend. For matched
throughput runs, compare `D4CNN_CONV=direct`, `winograd` (F2) and `winograd4`
(F4) with the same model/q-stamps/thread count; all now use fused C32 epilogues.
For example, after replacing model/input paths:

```bash
D4CNN_ISA=avx2 D4CNN_CONV=winograd4 python -m cpu_src benchmark \
  --model MODEL.npz --qstamps QSTAMPS.npz --count 512 \
  --threads 1 8 32 --repeats 3 --output x86_f4.json
```

For the current interactive validation, use `benchmarks/finalize_interactive.sh`
inside `srun`; it snapshots source, rebuilds on node-local storage, runs the C++
math/executor tests, Python tests, FP64 golden checks and real-stamp CPU timing.
`benchmarks/finalize_reference.py` separately reruns Torch CPU and strict-FP32 GPU
references after the CPU timing step. Both require an existing allocation.
The C++ test is also registered with CTest (`ctest --test-dir BUILD_DIR`).

Performance must be measured on the intended Grace/x86 hardware; peak-FLOP
estimates are not delivery claims. SIMD backends need validation on each actual
ISA, and dedicated-node scaling/science runs should be scheduled separately
from the small local checks.

### Historical paired GH200 / Grace run

The current optimized build and fresh comparisons are recorded in
[VALIDATION.md](VALIDATION.md). The following report is retained as a baseline.

The [2026-10-19 measured report](benchmarks/GH200_VALIDATION_20261019.md) records
321–345 gal/s at 72 cores (about 83% parallel efficiency). The 500–580 gal/s
forecast is not reached. Both models pass the revised e/R component-mean
criterion against strict-FP32 GPU. Individual-output diagnostics and the
TF32 s4 R11 mean-bias failure are detailed in that report.

`benchmarks/cluster_pair.py` compares the trained s4 and AttnW `last.pth`
checkpoints on 512 common held-out blended stamps (exposures 940–943). It
reconstructs all five q planes with the training noise seeds and checks q0
against the cached training preprocessing before inference. It records C++
1/8/36/72-core scaling, persistent PyTorch CPU 1/8/72-process baselines,
GPU batch sweeps with strict FP32 and original TF32 settings, T=4 correctness,
one-to-one/residual figures, raw predictions and repeated timings.

Allocate resources using the live site/account requirements, then launch from
this repository root (replace the example job ID with your allocation):

```bash
salloc --account=bdsp-dtai-gh --partition=ghx4-interactive \
  --nodes=1 --ntasks=1 --cpus-per-task=72 --gpus-per-node=1 \
  --mem=100G --time=02:00:00 --constraint='projects&taiga&work' --no-shell
bash cpu_src/benchmarks/launch_pair.sh 3417691
scancel 3417691
```

The launcher stages source, the built native extension and data on compute-node
`/tmp`; it uses the local `anacal_clone` interpreter. Outputs default to
`cpu_src/runs/paired_JOB_ID/`. The launcher does not allocate or release a job.
All CPU fork pools finish before CUDA initialization. Host-to-output inference
timings include input/output handling and exclude q generation, loading and
plotting; GPU device-resident timings are retained separately. This comparison
does not measure full-exposure detection or m/c calibration.

`--resume-cpu` on the Python runner reuses saved CPU results in the same
allocation and repeats GPU measurements. `--plots-only` redraws saved figures.
`python -m cpu_src.benchmarks.diagnose_pair --output OUTPUT` checks the eight
most sensitive stamps per model in float64 and repeats GPU predictions. Run
these commands on compute resources, with this repository on `PYTHONPATH`.
