"""Step s_measure (only with student_sim): the teacher CNN measures the student's galaxies (same code as measure)."""
from ml_wl.steps import measure
from ml_wl.steps._common import load_job_args, sg_job


def jobs(cfg):
    if cfg["student_sim"] is None:
        return []
    return [dict(lines=sg_job("s_measure", cfg), cwd="sg") for _ in measure.batch_ranges(cfg, cfg["student_sim"])]


def main():
    cfg, job = load_job_args("s_measure")
    measure.run(cfg, job, "s_measure", cfg["student_sim"])


if __name__ == "__main__":
    main()
