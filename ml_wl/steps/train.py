"""Step 7: train the student CNN with src.training.train_model on the step-6 images/labels."""
import json
import shutil
from pathlib import Path

from ml_wl.config import SG_DIR, step_dir
from ml_wl.steps._common import d, load_job_args, sg_job


def jobs(cfg):
    return [dict(lines=sg_job("train", cfg), cwd="sg")]


def main():
    import torch
    from src.training import train_model
    from ml_wl.models import build_model

    cfg, _ = load_job_args("train")
    s = cfg["student_cnn"]
    out = d(cfg, "train")
    ds_dir = step_dir(cfg, "dataset")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device, flush=True)
    model = build_model(s["arch"], s["kwargs"]).to(device)
    best, final = out / "best.pth", out / "final.pth"
    model = train_model(model, images_path=str(ds_dir / "images.npy"), csv_path=str(ds_dir / "catalog.csv"),
                        target="e", epochs=s["epochs"], batch_size=s["batch_size"],
                        num_workers=s["num_workers"], lr=s["lr"], wd=s["wd"], pt_path=str(best))
    torch.save(model.state_dict(), final)
    sel = step_dir(cfg, "rdsm") / "selection.json"
    (out / "config.json").write_text(json.dumps({
        "round": cfg["name"], "student_cnn": s, "dataset": cfg["dataset"],
        "teacher_checkpoint": cfg["teacher_cnn"]["checkpoint"],
        "rdsm_selection": json.loads(sel.read_text()) if sel.exists() else None,
        "images": str(ds_dir / "images.npy"), "csv": str(ds_dir / "catalog.csv"),
    }, indent=2) + "\n")
    if s["export_as"]:
        for src, name in ((best, f"{s['export_as']}.pth"), (final, f"{s['export_as']}_final.pth")):
            dst = Path(SG_DIR) / "models" / name
            if dst.exists():
                print(f"NOT overwriting existing {dst}", flush=True)
            else:
                shutil.copy2(src, dst)
                print("exported", dst, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
