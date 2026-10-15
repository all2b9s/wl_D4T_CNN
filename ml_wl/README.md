# ml_wl — one config per round: CNN → eDSM → RDSM → CNN

    cd single_galaxies          # activate the anacal_clone conda env first
    python -m ml_wl.run ml_wl/configs/r1_F8gt.yaml --test        # tiny end-to-end check, writes to r1_F8gt_test
    python -m ml_wl.run ml_wl/configs/r1_F8gt.yaml               # submit every unfinished step (afterok chain)
    python -m ml_wl.run ml_wl/configs/r1_F8gt.yaml --status
    python -m ml_wl.run ml_wl/configs/r1_F8gt.yaml --only dataset,train --force train   # rerun part of a round
    python -m ml_wl.run ml_wl/configs/r1_F8gt.yaml --dry-run     # write sbatch files, submit nothing

Steps (outputs under `rounds/<name>/NN_step/`; 01/02/06 are symlinks into `data_root` on /work):
`measure` (teacher CNN shapes + R) → `catalog` → `edsm` (optuna) → `retrain` → `rdsm` (pick checkpoint + RDSM labels)
→ `dataset` (10k subset + renoised images + csv) → `train` (student CNN, optionally exported to `models/`).

* Each job touches `DONE.<i>`; finished jobs are skipped on the next call. `--force STEP` clears them.
* `--adopt STEP=PATH` (catalog | edsm | retrain | rdsm) reuses an existing output. `measure` cannot be adopted,
  so for a round that reuses old outputs run only the later steps: `--only dataset,train` (or `steps:` in the yaml).
* Next round: copy the yaml, point `teacher_cnn.checkpoint` at `rounds/<prev>/07_cnn/best.pth`, change `name`.
* `retrain.epochs` must equal the 20 epochs hard-coded in `opt_e/optuna_eDSM.py` (50 inflated |m| by 3-10x).
* `single_galaxies/src` and `opt_e/src` are untouched. The opt_e steps call its CLI scripts (`paths.opt_e`); the two repos
  stay separate, and `rounds/<name>/provenance.json` records both git HEADs per submission.
* The `src` packages of the two repos share a name, so each step is its own job and never imports the other repo.
