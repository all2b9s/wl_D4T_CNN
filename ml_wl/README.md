# ml_wl — one config per round: CNN → eDSM → RDSM → CNN

    cd single_galaxies          # activate the anacal_clone conda env first
    python -m ml_wl.run ml_wl/configs/r1_F8gt.yaml --test        # tiny end-to-end check, writes to r1_F8gt_test
    python -m ml_wl.run ml_wl/configs/r1_F8gt.yaml               # submit every unfinished step (afterok chain)
    python -m ml_wl.run ml_wl/configs/r1_F8gt.yaml --status
    python -m ml_wl.run ml_wl/configs/r1_F8gt.yaml --only dataset,train --force train   # rerun part of a round
    python -m ml_wl.run ml_wl/configs/r1_F8gt.yaml --dry-run     # write sbatch files, submit nothing

Outputs and run metadata live under `/work/hdd/bfmo/wenyinli/measurement/ml_wl/<name>/`; Slurm logs live under `output/ml_wl/<name>/` in the single_galaxies repository. `ml_wl/` contains code and configuration only. Override `rounds`, `data_root`, or `log_dir` in YAML when needed.

Steps:
`measure` (teacher CNN shapes + R) → `catalog` → `edsm` (optuna) → `retrain` → `rdsm` (pick checkpoint + RDSM labels)
→ `dataset` (10k subset + renoised images + csv) → `train` (student CNN, optionally exported to `models/`).

* Each job touches `DONE.<i>`; finished jobs are skipped on the next call. `--force STEP` clears them.
* `--adopt STEP=PATH` (catalog | edsm | retrain | rdsm) reuses an existing output. `measure` cannot be adopted,
  so for a round that reuses old outputs run only the later steps: `--only dataset,train` (or `steps:` in the yaml).
* Next round: copy the yaml, point `teacher_cnn.checkpoint` at `/work/hdd/bfmo/wenyinli/measurement/ml_wl/<prev>/10_cnn/best.pth`, change `name`.
* `retrain.epochs` must equal the 20 epochs hard-coded in `opt_e/optuna_eDSM.py` (50 inflated |m| by 3-10x).
* `single_galaxies/src` and `opt_e/src` are untouched. The opt_e steps call its CLI scripts (`paths.opt_e`); the two repos
  stay separate, and `<round_dir>/provenance.json` records both git HEADs per submission.
* The `src` packages of the two repos share a name, so each step is its own job and never imports the other repo.

## Image preparation (teacher and student)
The sim cutouts are raw: PSF-convolved, noiseless, not deconvolved, not renoised. Both the teacher measurement and the
student dataset run them through `src.anacal.batch_calibration.prepare_q_images` (PSF-deconvolving resmoothing to a
Gaussian of FWHM `sim.psf_fwhm`) -- the same call `src/shape_measurement.py` uses. `dataset.noise.mode`:
`adaptive` (noise + renoise map, then prepare), `q_clean` (prepare the noiseless cutout, = what the teacher measured at
noise 0), `none` (raw cutouts, unprepared; does not match measurement-time inputs).

`dataset.image_sim` takes the student images from another sim directory (keys override `sim`; `psf_fwhm` is the resmoothing
target and should stay equal to what the CNN will be applied with). The RDSM labels belong to the galaxies the teacher
measured, so the other directory must hold the same galaxies in the same `img_k` / row layout (e.g. another PSF or shear
rendering); the step compares catsim `indices` row by row and aborts on any mismatch. Different galaxies need their own
teacher round (set `sim.dir` to that directory).


## Detection-weighted AttnW round

`configs/r1_AttnW_last_detw.yaml` fixes the previous AttnW `last.pth` and original
rendering (`tract_variance_calibration/input`, fixed noise 0.594). Run from
`single_galaxies` in `anacal_clone`:

```bash
python -m unittest ml_wl.test_blended -v
python -m ml_wl.run ml_wl/configs/r1_AttnW_last_detw.yaml --dry-run
python -m ml_wl.run ml_wl/configs/r1_AttnW_last_detw.yaml
```

The configured round stops after a single 25-trial study. It selects whole
exposures using the original seed=42 40/20/40 pools and the nearest cumulative
counts in the existing measurement manifest: 1105/551/1103 exposures give
999989/500361/1000133 detections before finite-input filtering. The exposure list
is saved once in `02_catalog/split_exposures.npz`; only missing selected exposures
are measured. New catalogs are checked immediately, so a mismatch stops the GPU
job rather than wasting the rest of its allocation. IDs must match exactly;
compared floating fields use rtol=1e-5, atol=1e-3 for CNN R11/R22 only and
atol=1e-7 for all other fields.

Inputs are `x=w*e`, `Rx=w*R+e outer dw/dg`. One compressed `catalog.npz` contains
float32 weighted inputs/full responses, raw CNN shapes/full responses, detection
and exposure IDs, and uint8 split codes (train=0,val=1,test=2). Other original
catalog fields are not copied. After the verified catalog and audit are saved,
redundant per-exposure measurement copies are removed. Failed preparation keeps
them for resuming. To change the sample after preparation, use a new round.

Optuna and retraining support `--data-split/--eval-split`, so training and
validation read the same NPZ with the training-only standard deviation. Defaults
retain the original unsplit behavior. Detection weighting is already in the
inputs: do not multiply the outputs by w again.

Early stopping uses fixed-noise DSM loss on up to 8192 validation rows, seed 1849.
At least six epochs are run; four epochs without a 1% improvement stop a trial,
with best-validation weights restored before full validation metrics. Once three
completed trials with the same noisy dimensions supply references at an epoch,
a trial with >2x their median loss for two consecutive epochs is pruned. Native
Optuna intermediate reporting is not used because the study is multi-objective.
Nonfinite losses are pruned. Trial time is checked every epoch (300 seconds), and
the study starts no new trials after 4800 seconds. Limits may stop below 25 trials.
No per-trial checkpoints or batch progress plots are saved; policy and loss
histories live in the study DB.

DeltaAI rejects CPU-only Slurm jobs, so catalog preparation runs as a CPU step
at the end of the measurement allocation (`prepare_catalog_in_measure`); no
standalone catalog job is submitted. Configured allocations are measurement plus
preparation 1h15m/one GPU and study 1h30m/one GPU: at most 2.75 GPU-hours of
requested walltime. Measurement uses 48G RAM and the study uses 32G. Broken dependencies are
cancelled automatically. `--force` does not recursively invalidate later steps.

Later RDSM export uses the saved training norm and `--compact`: only physical
RDSM outputs, full responses, IDs and small checkpoint metadata are retained.
The evaluator joins the shared input catalog by row/IDs, uses diagonal responses
for shear/noise, and reports off-diagonal zero diagnostics. It keeps summary CSVs
and reusable exposure statistics, not bootstrap draw arrays. The Optuna m remains
an input-perturbation diagnostic, not an end-to-end image-shear bias measurement.
