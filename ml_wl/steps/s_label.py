"""Step s_label (only with student_sim): apply the eDSM selected on `sim` to the student galaxies' teacher shapes.

The eDSM (trained on the teacher sim) maps shape -> RDSM label, so the student's labels need no eDSM of their own.
The normalisation (std of the teacher-sim shapes) must be the one the model was trained with, not the student set's.
"""
from ml_wl.config import step_dir
from ml_wl.steps.catalog import artifact
from pathlib import Path

RDSM_PY = Path(__file__).resolve().with_name("rdsm.py")


def jobs(cfg):
    if cfg["student_sim"] is None:
        return []
    a, out = step_dir(cfg, "rdsm"), step_dir(cfg, "s_label")
    lines = [
        f"CKPT=$(python {RDSM_PY} ckpt --selection {a}/selection.json)",
        f"NORM=$(python {RDSM_PY} norm --catalog {a}/rdsm_catalog.npz)",
        'echo "checkpoint: $CKPT  norm (teacher sim): $NORM"',
        f"python -u build_gauss_train_rdsm_catalog.py --checkpoint \"$CKPT\" --data {artifact(cfg, 's_catalog')} "
        f"--out {out}/rdsm_catalog.npz --norm $NORM --device cuda",
    ]
    return [dict(lines=lines, cwd="opt")]
