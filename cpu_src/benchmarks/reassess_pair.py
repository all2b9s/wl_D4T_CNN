"""Reassess saved CPU/GPU outputs with the user-specified component mean criterion."""
import argparse
import json
from pathlib import Path
import numpy as np
from cpu_src.validation import error_stats


def reassess(directory):
    report = {"criterion": "abs(mean(CPU - reference, over sources)) < 1e-4 for each e/R component",
              "limit": 1e-4, "models": {}}
    for name in ("s4", "attnw"):
        model = {}
        with np.load(directory / f"{name}_paired.npz", allow_pickle=False) as data:
            for reference in ("torch_cpu", "strict_fp32", "production_default"):
                stats = {key: error_stats(data[f"cpp_{key}"], data[f"{reference}_{key}"],
                                         1e-6, 1e-5 if key == "e" else 1e-4, 1e-4)
                         for key in ("e", "R")}
                model[reference] = {"passed": all(s["passed"] for s in stats.values()), **stats}
        report["models"][name] = model
    (directory / "mean_acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(reassess(args.directory), indent=2))
