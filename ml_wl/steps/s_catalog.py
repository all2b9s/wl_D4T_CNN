"""Step s_catalog (only with student_sim): merge the teacher shapes measured on the student's galaxies."""
from ml_wl.steps.catalog import make_job


def jobs(cfg):
    s = cfg["student_sim"]
    if s is None:
        return []
    return make_job(cfg, "s_measure", "s_catalog", s, s["fpfs_dir"], s["fpfs_prefix"])
