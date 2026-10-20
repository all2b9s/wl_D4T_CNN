"""Driver: python -m ml_wl.run <round.yaml> [--test] [--dry-run] [--status] [--only A,B] [--force A,B]
                                           [--adopt STEP=PATH ...] [--local STEP]

Writes one sbatch per job under <round_dir>/sbatch/ and submits them with afterok dependencies. Every job touches
<step_dir>/DONE.<i> on success; a step is finished when all its DONE.<i> exist, and finished jobs are skipped on
the next invocation (resume). Heavy imports (torch, src.*) never happen here: the steps run as separate jobs.
"""
import argparse
import datetime
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

from ml_wl import config as C

SBATCH_TMPL = """#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --mem={mem}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
{gpu_directives}
#SBATCH --no-requeue
#SBATCH -t {time}
#SBATCH --output={logs}/%x.%j.out
#SBATCH --error={logs}/%x.%j.err

"""


def step_module(step):
    return importlib.import_module(f"ml_wl.steps.{'blended' if step == 'evaluate' else step}")


def done_file(cfg, step, i):
    return C.step_dir(cfg, step) / f"DONE.{i}"


def step_jobs(cfg, step):
    return step_module(step).jobs(cfg)


def step_state(cfg, step):
    jobs = step_jobs(cfg, step)
    done = [done_file(cfg, step, i).exists() for i in range(len(jobs))]
    return jobs, done


def prepare_dirs(cfg, steps=None):
    rd = C.round_dir(cfg)
    (rd / "sbatch").mkdir(parents=True, exist_ok=True)
    Path(cfg["log_dir"]).mkdir(parents=True, exist_ok=True)
    steps = list(steps or selected_steps(cfg, None)) if cfg.get("blended") is not None else C.STEPS
    if cfg.get("blended") is not None and "measure" in steps and "catalog" not in steps:
        steps.append("catalog")
    for step in steps:
        name, big = C.STEP_DIRS[step]
        link = rd / name
        if big and C.big_dir(cfg, step) != link:
            real = C.big_dir(cfg, step)
            real.mkdir(parents=True, exist_ok=True)
            if not link.is_symlink() and not link.exists():
                link.symlink_to(real)
        else:
            link.mkdir(parents=True, exist_ok=True)
    C.frozen_config_path(cfg).write_text(json.dumps(cfg, indent=2) + "\n")


def git_info(path):
    def g(*a):
        return subprocess.run(["git", "-C", str(path), *a], capture_output=True, text=True).stdout.strip()
    return {"path": str(path), "head": g("rev-parse", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_files": len([l for l in g("status", "--porcelain").splitlines() if l])}


def write_provenance(cfg):
    ck = Path(cfg["teacher_cnn"]["checkpoint"])
    entry = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "config": cfg["source_config"],
             "single_galaxies": git_info(C.SG_DIR), "opt_e": git_info(cfg["paths"]["opt_e"]),
             "teacher_checkpoint": str(ck),
             "teacher_sha256": hashlib.sha256(ck.read_bytes()).hexdigest() if ck.exists() else None}
    if cfg.get("blended") is not None:
        entry["blended"] = cfg["blended"]
        entry["input_transform"] = "detw_e"
    p = C.round_dir(cfg) / "provenance.json"
    hist = json.loads(p.read_text()) if p.exists() else []
    p.write_text(json.dumps(hist + [entry], indent=2) + "\n")


def build_script(cfg, step, i, job, sbatch=True):
    s = cfg["slurm"]
    resources = {**s, **s.get("resources", {}).get(step, {})}
    gpus = int(resources.get("gpus", 1))
    cwd = C.SG_DIR if job["cwd"] == "sg" else Path(cfg["paths"]["opt_e"])
    head = ""
    if sbatch:
        head = SBATCH_TMPL.format(jobname=f"{cfg['name']}_{step}{i}", partition=s["partition"], account=s["account"],
                                  mem=resources["mem"], cpus=resources["cpus"], time=s["time"][step], logs=cfg["log_dir"],
                                  gpu_directives=f"#SBATCH --gpus-per-node={gpus}\n#SBATCH --gpu-bind=closest" if gpus else "")
    env = (f"source {s['conda_sh']}\nconda activate {s['env']}\n" if sbatch else "")
    runtime_env = "export PYTHONDONTWRITEBYTECODE=1"
    if cfg.get("blended") is not None:
        runtime_env += " OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
    body = [f"set -euo pipefail", env + "unset PYTHONPATH", runtime_env, f"JOB_INDEX={i}", 'echo "host: $(hostname)  start: $(date)"',
            f"cd {cwd}", *job["lines"], f"touch {done_file(cfg, step, i)}", 'echo "end: $(date)"']
    return head + "\n".join(body) + "\n"


def sbatch(path, deps):
    cmd = ["sbatch", "--parsable", "--kill-on-invalid-dep=yes"] + ([f"--dependency=afterok:{':'.join(deps)}"] if deps else []) + [str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(result.stderr.strip())
    return result.stdout.strip().split(";")[0]


def selected_steps(cfg, only):
    steps = only or cfg["steps"] or [s for s in C.STEPS if s != "evaluate"]
    bad = [s for s in steps if s not in C.STEPS]
    if bad:
        raise SystemExit(f"unknown steps {bad}; valid: {C.STEPS}")
    return [s for s in C.STEPS if s in steps]


ADOPT = {  # step -> artifact path inside the step dir
    "catalog": lambda cfg: C.step_dir(cfg, "catalog") / "catalog.npz",
    "edsm": lambda cfg: step_module("edsm").db_path(cfg),
    "retrain": lambda cfg: step_module("retrain").study_dir(cfg),
    "rdsm": lambda cfg: C.step_dir(cfg, "rdsm") / "rdsm_catalog.npz",
}


def adopt(cfg, spec):
    step, path = spec.split("=", 1)
    if step not in ADOPT:
        raise SystemExit(f"--adopt supports {sorted(ADOPT)}")
    src, dst = Path(path).resolve(), ADOPT[step](cfg)
    if not src.exists():
        raise SystemExit(f"{src} does not exist")
    if step == "edsm" and src.name != dst.name:
        raise SystemExit(f"adopted DB must be named {dst.name} (set edsm.study_name to match its internal study)")
    if dst.is_symlink() or dst.exists():
        raise SystemExit(f"{dst} already exists; remove it first")
    dst.symlink_to(src)
    if step == "rdsm" and (src.parent / "selection.json").exists():     # keeps the provenance train.py records
        (dst.parent / "selection.json").symlink_to(src.parent / "selection.json")
    for i in range(len(step_jobs(cfg, step))):
        done_file(cfg, step, i).touch()
    print(f"adopted {step}: {dst} -> {src}")


def status(cfg):
    jp = C.round_dir(cfg) / "jobs.json"
    ids = json.loads(jp.read_text()) if jp.exists() else {}
    for step in C.STEPS:
        jobs, done = step_state(cfg, step)
        line = f"{step:9s} {sum(done)}/{len(jobs)} done"
        if step in ids and not all(done):
            out = subprocess.run(["sacct", "-j", ",".join(ids[step]), "-X", "-n", "-P", "--format=JobID,State"],
                                 capture_output=True, text=True).stdout.split()
            line += "   jobs: " + ", ".join(o.replace("|", ":") for o in out)
        print(line)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--test", action="store_true", help="tiny version of every step, writes to <name>_test")
    p.add_argument("--dry-run", action="store_true", help="write sbatch files only, submit nothing")
    p.add_argument("--status", action="store_true")
    p.add_argument("--only", help="comma-separated steps to run (default: yaml `steps`, else all)")
    p.add_argument("--force", help="comma-separated steps whose DONE markers are cleared first")
    p.add_argument("--adopt", action="append", default=[], metavar="STEP=PATH",
                   help="reuse an existing output (catalog|edsm|retrain|rdsm) and mark the step done")
    p.add_argument("--local", metavar="STEP", help="run one step in the current shell instead of sbatch")
    a = p.parse_args()

    cfg = C.load(a.config, test=a.test)
    if a.status:
        return status(cfg)
    requested = [a.local] if a.local else selected_steps(cfg, a.only.split(",") if a.only else None)
    prepare_dirs(cfg, requested)
    for spec in a.adopt:
        adopt(cfg, spec)
    for step in (a.force.split(",") if a.force else []):
        for f in C.step_dir(cfg, step).glob("DONE.*"):
            f.unlink()
        print(f"cleared DONE markers of {step} (later steps are NOT invalidated)")

    if a.local:
        jobs, done = step_state(cfg, a.local)
        for i, job in enumerate(jobs):
            if not done[i]:
                subprocess.run(["bash", "-c", build_script(cfg, a.local, i, job, sbatch=False)],
                               check=True, env={**os.environ, "PYTHONPATH": ""})
        return

    write_provenance(cfg)
    jp = C.round_dir(cfg) / "jobs.json"
    ids = json.loads(jp.read_text()) if jp.exists() else {}
    submitted = []
    print(f"round {cfg['name']}  ->  {C.round_dir(cfg)}")
    # 1) write every sbatch first, so a config problem fails before anything is submitted
    plan = []
    for step in selected_steps(cfg, a.only.split(",") if a.only else None):
        jobs, done = step_state(cfg, step)
        todo = [i for i, d in enumerate(done) if not d]
        if not jobs:
            print(f"  {step:9s} n/a (no standalone job)")
            continue
        if not todo:
            print(f"  {step:9s} done, skipped")
            continue
        files = []
        for i in todo:
            sb = C.round_dir(cfg) / "sbatch" / f"{list(C.STEPS).index(step) + 1}_{step}_{i}.sbatch"
            sb.write_text(build_script(cfg, step, i, jobs[i]))
            files.append(sb)
        plan.append((step, files))
    # 2) submit, each step after everything submitted before it
    for step, files in plan:
        new = ["DRYRUN" if a.dry_run else sbatch(sb, submitted) for sb in files]
        print(f"  {step:9s} jobs {new}" + (f"  (after {submitted})" if submitted else ""))
        if not a.dry_run:
            ids[step] = new
            jp.write_text(json.dumps(ids, indent=2) + "\n")
            submitted += new
    if not a.dry_run:
        jp.write_text(json.dumps(ids, indent=2) + "\n")
    print("logs:", C.round_dir(cfg) / "logs")


if __name__ == "__main__":
    main()
