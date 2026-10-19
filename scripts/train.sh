#!/bin/bash
# Current blending workflow. Calling without arguments only displays help.
set -euo pipefail
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CODE_DIR"
case "${1:-help}" in
  train)
    shift
    if (( $# )); then echo 'Use MODELS and DRY_RUN environment variables; see scripts/README.md.' >&2; exit 2; fi
    exec bash scripts/training_size_test/retrain_q_three.sh
    ;;
  evaluate)
    shift
    exec /u/wenyinli/.conda/envs/anacal_clone/bin/python scripts/training_size_test/submit_q_evaluation.py "$@"
    ;;
  status)
    squeue -u "$(id -un)" -o '%.15i %.32j %.10T %.10M %E'
    ;;
  help|--help|-h)
    cat <<'HELP'
Current blending workflow (AdaSigma / AttnW / s4):
  bash scripts/train.sh status
  DRY_RUN=1 bash scripts/train.sh train
  bash scripts/train.sh train
  bash scripts/train.sh evaluate             # generate/validate only
  bash scripts/train.sh evaluate --submit

Training implementation: scripts/training_size_test/train_blended_rdsm_size.py
Parameters/submission:   scripts/training_size_test/retrain_q_three.sh
Evaluation submission:  scripts/training_size_test/submit_q_evaluation.py
Data: renoised q_img[0], noise_std=0.594; 330k objects, 9 epochs.
AdaSigma/AttnW: detection-weighted individual MSE, lambda=0. s4: ordinary MSE.
Existing submissions are guarded against duplicate submission.
Read scripts/README.md for the exact call chain and output paths.
HELP
    ;;
  *) echo "Unknown command: $1; use --help" >&2; exit 2 ;;
esac
