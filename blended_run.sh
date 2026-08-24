#!/bin/bash
# ===========================================================================
# Blended Image FPFS Detection + ML Shape Measurement — Batch Submission
# ===========================================================================

# ---- Config ----
JOB_NAME="blended_det_fpfs_noiseless_s4"
MODEL_TYPE="cnn"           # "cnn" or "gauss"
GAUSS_SIGMA=4.0                 # only used when MODEL_TYPE=gauss
MODEL_PATH="./models/F8_fpfs_l5c32r01_50ep.pth"   # only used when MODEL_TYPE=cnn
PSF_PATH="./logs/blended_test_psf.npy"
CAT_REF_PATH="/projects/bdsp/wenyinli/codes/data/catsim-v4/OneDegSq.fits"
BLENDED_DIR="/taiga/illinois/las/astro/xinliuxl/DC1_sim_blended"
#BLENDED_DIR="/work/hdd/bfcn/wenyinli/noiseless_sim"
OUTPUT_ROOT="/work/hdd/bfmo/wenyinli/measurement/blended_sims"
BASE_OUT="/projects/bdsp/wenyinli/datasets/xlens_sims/n0594/${JOB_NAME}"

IO_CHUNK=5
ML_BATCH=600
NOISE_LEVEL="0"                 # e.g. 0.06 for 6% noise; empty = from variance map
SNR_CUT=5.0                    # minimum peak SNR for FPFS detection (anacal snr_peak_min)
SIM_MODES=("sim_mode0" "sim_mode40")
# EXP_RANGE: first exposure index, last+1 exposure index
EXP_START=0
EXP_END=10000

# ---- Calibration (standalone job, run after all measurement jobs) ----
RUN_CALIBRATION="true"         # submit a dedicated calibration job
CAL_SHAPE_MODE="ml"            # "fpfs", "ml", or "both"
CAL_MAG_CUT=""                 # e.g. 24.5 for selection response; empty = no cut
CAL_BS_TIMES=100
CAL_WORKERS=64
CAL_OUTPUT_DIR=""              # optional; default under the calibration output dir
CAL_LOG_CSV=""                 # optional CSV log path

# ---- Submit one job per sim_mode (measurement only) ----
JOB_IDS=()
for SIM_MODE in "${SIM_MODES[@]}"; do
  # Map sim_mode to output label
  case "$SIM_MODE" in
    sim_mode0) MODE_LABEL="g1_0" ;;
    sim_mode40) MODE_LABEL="g1_1" ;;
    *) MODE_LABEL="$SIM_MODE" ;;
  esac

  JOB_NAME_I="${JOB_NAME}_${MODE_LABEL}"
  OUTDIR_I="${OUTPUT_ROOT}/${MODE_LABEL}/${JOB_NAME}"
  mkdir -p "$OUTDIR_I"
  mkdir -p "${BASE_OUT}"

  # Build python args conditionally
  PYTHON_ARGS="--blended_dir ${BLENDED_DIR}"
  PYTHON_ARGS+=" --output_root ${OUTPUT_ROOT}"
  PYTHON_ARGS+=" --psf_path ${PSF_PATH}"
  PYTHON_ARGS+=" --cat_ref_path ${CAT_REF_PATH}"
  PYTHON_ARGS+=" --sim_modes ${SIM_MODE}"
  PYTHON_ARGS+=" --range ${EXP_START} ${EXP_END}"
  PYTHON_ARGS+=" --model_type ${MODEL_TYPE}"
  PYTHON_ARGS+=" --gauss_sigma ${GAUSS_SIGMA}"
  if [ "$MODEL_TYPE" = "cnn" ]; then
    PYTHON_ARGS+=" --model ${MODEL_PATH}"
  fi
  if [ -n "$NOISE_LEVEL" ]; then
    PYTHON_ARGS+=" --noise ${NOISE_LEVEL}"
  fi
  PYTHON_ARGS+=" --snr_cut ${SNR_CUT}"
  #PYTHON_ARGS+=" --skip_detection"
  #PYTHON_ARGS+=" --center_on_truth"
  #PYTHON_ARGS+=" --mag_cut 26"
  PYTHON_ARGS+=" --fname ${JOB_NAME}"
  PYTHON_ARGS+=" --io_chunk ${IO_CHUNK}"
  PYTHON_ARGS+=" --ml_batch ${ML_BATCH}"
  PYTHON_ARGS+=" --no_csv"
  PYTHON_ARGS+=" --workers 64"

  SB_FILE="${OUTPUT_ROOT}/job_blended_${MODE_LABEL}.sbatch"

  cat > "$SB_FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME_I}
#SBATCH --partition=ghx4
#SBATCH --mem=200G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bfmo-dtai-gh
#SBATCH --no-requeue
#SBATCH -t 12:00:00
#SBATCH --output=${BASE_OUT}/%x.%j.out
#SBATCH --error=${BASE_OUT}/%x.%j.err

set -euo pipefail

source /projects/bfcn/wenyinli/miniconda3/etc/profile.d/conda.sh
conda activate /u/wenyinli/.conda/envs/anacal_clone

echo "=== Blended Image FPFS Detection + ML Shape Measurement ==="
echo "BLENDED_DIR:${BLENDED_DIR}"
echo "Sim mode: ${SIM_MODE}  →  ${MODE_LABEL}"
echo "Range: ${EXP_START}-${EXP_END}  |  Model type: ${MODEL_TYPE}"
if [ "$MODEL_TYPE" = "cnn" ]; then
  echo "Checkpoint: ${MODEL_PATH}"
else
  echo "Gauss sigma: ${GAUSS_SIGMA}  (no checkpoint)"
fi
echo "IO chunk: ${IO_CHUNK} exposures  |  ML batch: ${ML_BATCH}"

cd /projects/bdsp/wenyinli/codes/single_galaxies

python -u blended_run.py ${PYTHON_ARGS}
EOF

  echo "Submitting: ${JOB_NAME_I}  (${SIM_MODE})"
  JOB_ID=$(sbatch "$SB_FILE" | awk '{print $NF}')
  JOB_IDS+=("$JOB_ID")
  echo "  → job script: ${SB_FILE}  (job ${JOB_ID})"
done

# ===========================================================================
# Calibration job (needs BOTH shear signs → after all measurement jobs)
# ===========================================================================
if [ "$RUN_CALIBRATION" = "true" ]; then
  CAL_JOB_NAME="${JOB_NAME}_cali"
  CAL_SB_FILE="${OUTPUT_ROOT}/job_blended_calibration.sbatch"

  CAL_ARGS="--output_root ${OUTPUT_ROOT}"
  CAL_ARGS+=" --fname ${JOB_NAME}"
  CAL_ARGS+=" --range ${EXP_START} ${EXP_END}"
  CAL_ARGS+=" --shape_mode ${CAL_SHAPE_MODE}"
  CAL_ARGS+=" --bs_times ${CAL_BS_TIMES}"
  CAL_ARGS+=" --workers ${CAL_WORKERS}"
  if [ -n "$CAL_MAG_CUT" ]; then
    CAL_ARGS+=" --mag_cut ${CAL_MAG_CUT}"
  fi
  if [ -n "$CAL_OUTPUT_DIR" ]; then
    CAL_ARGS+=" --output_dir ${CAL_OUTPUT_DIR}"
  fi
  if [ -n "$CAL_LOG_CSV" ]; then
    CAL_ARGS+=" --log_csv ${CAL_LOG_CSV}"
  fi

  # Run only after all measurement jobs finish successfully
  DEP_ARGS=""
  if [ ${#JOB_IDS[@]} -gt 0 ]; then
    DEP_ARGS="--dependency=afterok:$(IFS=:; echo "${JOB_IDS[*]}")"
  fi

  cat > "$CAL_SB_FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=${CAL_JOB_NAME}
#SBATCH --partition=ghx4
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CAL_WORKERS}
#SBATCH --gpus-per-node=1
#SBATCH --account=bfmo-dtai-gh
#SBATCH -t 01:00:00
#SBATCH --output=${BASE_OUT}/%x.%j.out
#SBATCH --error=${BASE_OUT}/%x.%j.err

set -euo pipefail

source /projects/bfcn/wenyinli/miniconda3/etc/profile.d/conda.sh
conda activate /u/wenyinli/.conda/envs/anacal_clone

echo "=== Blended Calibration (single-shear g1) ==="
echo "output_root: ${OUTPUT_ROOT}  |  fname: ${JOB_NAME}"
echo "Range: ${EXP_START}-${EXP_END}  |  shape_mode: ${CAL_SHAPE_MODE}"

echo "note: needs g1_0 & g1_1 measurement outputs (dependency on pipeline jobs)"

cd /projects/bdsp/wenyinli/codes/single_galaxies

python -u blended_calibration.py ${CAL_ARGS}
EOF

  echo "Submitting: ${CAL_JOB_NAME}  (calibration, dep=${DEP_ARGS:-none})"
  sbatch $DEP_ARGS "$CAL_SB_FILE"
  echo "  → job script: ${CAL_SB_FILE}"
fi

echo "All jobs submitted."
