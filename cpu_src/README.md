# D4CNN CPU inference

Standalone CPU inference and blended-image calibration, isolated under `cpu_src/`.
The default is **float32 throughout** (normalization, CNN, attention, pooling and
heads), matching the current PyTorch precision. Float64 is a validation mode.
Both `Forward8_AttnW_CNN` and `Forward8_fixW_CNN` are supported. Existing `src/`
and the AnaCal checkout are read/reused, never patched by this package.

The current implementation includes NEON/AVX2/AVX-512 vector GELU/LayerNorm, C32 Winograd,
fused Winograd epilogues, persistent workers and reusable per-worker buffers.
fixW pools each orbit directly. AttnW uses query tiles, vector QK/AV and directly
accumulates low-resolution scalar tokens when the grid divides the image size.
See [current validation](VALIDATION.md) for measured acceptance and throughput.

## Build

Use the same Python interpreter and AnaCal source revision as your detection
installation. NumPy and pybind11 must be available. Run from `single_galaxies/`:

```bash
PYTHON_BIN=python3 ANACAL_SOURCE_DIR=/path/to/AnaCal \
bash cpu_src/build_local.sh
```

`CPU_CXX` selects a direct C++17 GCC/Clang compiler (default `g++`), and
`BUILD_JOBS` defaults to 2. The helper uses `cpu_src/.build-tmp/`, so a full `/tmp`
does not prevent compilation. Use a separate CMake build directory when changing
compilers after initial configuration. Build on the target architecture; an ARM
extension cannot be reused on x86, or vice versa.

On filesystems where CMake probe creation is very slow, add
`D4CNN_DIRECT_BUILD=1` to the same command. This uses the same compiler flags,
parallel compilation and cached object files, without CMake probes. Both build
paths install the extension atomically.
The Python API checks native API version 3 at model construction and reports a
rebuild instruction if an old extension is present. Always restart interpreters
that loaded an older extension after rebuilding it.

Without `ANACAL_SOURCE_DIR`, the q-stamp model API still builds; native image
measurement raises a clear error. With it, the extension compiles AnaCal's
`src/image.cpp` and reuses its ImageQ/ImageQLease/pocketfft implementation. The
extension does not register AnaCal classes again or link to its Python binary.

For wheel installation, `pyproject.toml` also supports scikit-build-core:

```bash
python -m pip install ./cpu_src \
  -Ccmake.define.ANACAL_SOURCE_DIR=/path/to/AnaCal
```

The full pipeline additionally uses the existing repository's `src` package,
AnaCal, astropy, fitsio, pandas, scipy, tqdm and galsim. Run from this repository
root or add it to `PYTHONPATH`. PyTorch/torchvision are needed only for export,
golden data, reference comparisons and reference tests.

## Export a model

```bash
python -m cpu_src export \
  --checkpoint /path/to/attnw/last.pth \
  --output cpu_src/runs/attnw.npz
```

The exporter reads the adjacent `config.json`, resolves constructor defaults,
checks weight keys/shapes and records checkpoint/source hashes and preprocessing.
Conflicting explicit parameters are rejected. For old fixW checkpoints, supply
architecture, explicit sigmas and preprocessing, rather than guessing defaults:

```bash
python -m cpu_src export \
  --checkpoint /path/to/fixw/checkpoint.pth \
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

### Measured CPU throughput and numerical validation (2026-10-20)

FP32 inference, T=2, median of three stable runs. EPYC results were measured on
2× AMD EPYC 7763 with GCC 14 and AVX2; ARM results use NVIDIA Grace with NEON.
These timings include network inference and Python input/output handling,
excluding model loading, q-stamp preparation and detection. Hardware, compiler,
batch size and thread placement affect performance. Run provenance and earlier
optimization rounds are retained in [VALIDATION.md](VALIDATION.md).

| CPU / configuration | s4 (gal/s) | AttnW (gal/s) |
|---|---:|---:|
| EPYC 7763, 1 core | 21.7 | 18.8 |
| EPYC 7763, 72 cores, 512 sources | 1175 | 912 |
| EPYC 7763, 128 cores, 512 sources | 1934 | 1474 |
| EPYC 7763, 128 cores, 2304 sources | 1987 | 1563 |
| Grace, 1 core | 17.0 | 15.5 |
| Grace, 72 cores, 512 sources | 1031 | 895 |

At 128 cores and 512 sources, EPYC parallel efficiency relative to one core is
69.6% / 61.3% (s4 / AttnW). s4 rises from 973 at 64 cores to 1934 at 128 cores.

- F4 is the default: s4 reaches 1934 gal/s versus 1532 with F2 at 128 cores.
- Direct convolution is a slow fallback on the measured x86 CPU: s4 gives
  2.2 / 245 gal/s at 1 / 128 cores; scalar gives 1.4 / about 144.
- Use automatic batching or enough sources to occupy all workers. Batches of
  only 64 sources at 128 cores give 1131 / 978 gal/s in the source-API experiment.
  Automatic source batches contain `max(64,8*nthreads)` objects (1024 at 128
  workers). The table above measures q-stamp inference separately.

On x86, C++ kernel tests, 14 Python tests and FP64 golden validation passed.
For 512 real stamps against strict-FP32 GPU, the measured maximum absolute
signed component-mean biases are:

| Model | e mean bias | R mean bias | Largest individual absolute ΔR |
|---|---:|---:|---:|
| s4 | 5.8e-9 | 1.1e-7 | 6.9e-6 |
| AttnW | 1.1e-8 | 1.1e-7 | 4.2e-6 |

All e/R components pass the 1e-4 mean-bias criterion. F2/direct/scalar also pass;
T=2/T=4 e/R are bit-identical and position-derivative checks pass. Some individual
R values exceed the older pointwise diagnostic tolerance. Grace round 3 has
maximum component-mean bias <=2.3e-7 and identical T=2/T=4 e/R.

FP64 golden validation passes the unchanged pointwise `atol=rtol=1e-12` on both
architectures. EPYC 7763 (Milan) has no AVX-512 and runs AVX2.
Numerical agreement does not replace exposure-level m/c science validation.

### Isolated xlens measurement (`isolated_measure_pool`)

Production path for the isolated-galaxy xlens sims (`shape_measure_multi_cpu.sh`).
One batch = 100 `img_` dirs = 10 000 galaxies; output is `<fname>_shapes_<b>.npy`
`[N,2]` and `<fname>_Rana_<b>.npy` `[N,1,2,2]`, compatible with `src/cal_biases.py`.

```bash
python -u -m cpu_src.isolated_measure_pool --model cpu_src/runs/X.npz --noise 0.594 \
  --fname NAME --shear_tasks 0_g1 --range 0 2000 --save_base /path/to/measurement_cpu/
```

**Design.** A single pool of worker processes, one per core (default: allowed cores
minus 2 kept for the parent). Each worker takes one `img_` dir (100 galaxies) and runs
load -> q-image -> 1-thread C++ inference, so no core idles between stages and none is
oversubscribed. The parent draws the per-batch measurement noise from
`RandomState(20020620 + batch)` (same stream and float32 noise level as before) into
shared memory, at most `--max_batches_inflight` (default 4, 328 MB each) batches ahead,
and writes a batch once all 100 dirs are back. Files are written atomically and finished
batches are skipped, so a timed-out job can simply be resubmitted. Output is
bit-identical to the previous prefetch-thread version (`isolated_measure.py`, kept for
reference): max abs difference 0 for e and R over 16 batches. `--nthreads` and
`--io_workers` are accepted but ignored.

**Measured (2026-10-22, 2× EPYC 7763, 128 cores, `0_g1`, noise 0.594):**

| Version | s / batch | gal/s | Note |
|---|---:|---:|---|
| Old production job (`io_workers=64`, 14 h run, 1699 batches) | 29.7 | 337 | avg ~54 of 128 cores busy; timed out |
| Old pipeline re-measured, `io_workers=64` | 17.1 | ~580 | CPU idle 41%, sys 32% (192 threads on 128 cores) |
| Old pipeline re-measured, `io_workers=16` | 10.7 | ~930 | CPU idle 45% |
| `isolated_measure_pool`, 126 workers (16 batches) | 5.5 | 1819 | worker utilization 93%, steady CPU idle 3–9%, sys ~1% |

The pool figure includes startup; its busy core-seconds split into load 2%, q-image 1%,
inference 97%, i.e. throughput is now set by inference (~1950 gal/s ceiling, matching
the table above). A full 4-shear run (16 000 batches) is therefore about 24 node-hours
at 5.5 s/batch (estimate from the 16-batch measurement), versus about 132 at 29.7 s.

Why the old version stalled: its prep (per batch, `io_workers=16`, no inference running)
took 6.5 s serially, namely `build_dataset` 2.7 s, noise draw 0.94 s (single thread),
`prepare_q_images` 2.7 s, `astype` 0.13 s. Most of the reading and q-image time was
shipping ~1 GB of cutouts/PSFs/noise between processes, not disk or arithmetic; prep
was longer than inference and the two competed for the same cores when overlapped.

### Kernel tests and profiling

Run on your workstation or allocated compute resources with a C++17 compiler.
On a managed cluster, follow its scheduling policy for builds and benchmarks.
After the CMake build above:

```bash
ctest --test-dir cpu_src/build/local --output-on-failure
python -m unittest discover -s cpu_src/tests -v
```

For a standalone kernel test without Python or AnaCal at execution time:

```bash
g++ -std=c++17 -O3 -pthread -I cpu_src/include \
  cpu_src/tests/test_kernels.cpp cpu_src/native/kernels_simd.cpp \
  cpu_src/native/convolution.cpp cpu_src/native/kernels.cpp -o /tmp/d4cnn_test_kernels
/tmp/d4cnn_test_kernels
```

The test exercises each available SIMD ISA, F2/F4, T=2/T=4, boundary tiles,
fused epilogues, first-layer skip projections, LayerNorm, GELU, softmax and
attention against independent double calculations. CTest registers it as
`kernels`. Python tests additionally cover the full model and pipeline; install
the reference/pipeline dependencies and build with `ANACAL_SOURCE_DIR` first.

`D4CNN_ISA=avx2|avx512|neon|scalar` selects an available ISA. On x86, `auto`
probes AVX2 against AVX-512 once per process when both are available;
`D4CNN_ISA_PROBE=0` disables that probe. For example:

```bash
D4CNN_ISA=avx2 D4CNN_CONV=winograd4 python -m cpu_src benchmark \
  --model /path/to/model.npz --qstamps /path/to/qstamps.npz --count 512 \
  --threads 1 8 32 --repeats 3 --output cpu_src/runs/f4_benchmark.json
```

Choose thread counts within your available CPU allocation. Compare F4, F2 and
direct with identical model/input/thread settings. The portable benchmark entry
point is `python -m cpu_src benchmark`; dataset-specific research utilities and
their input conventions are listed in [benchmarks/README.md](benchmarks/README.md).
