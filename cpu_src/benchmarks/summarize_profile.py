"""Summarize saved ablations and generate standalone performance figures."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path)
args = parser.parse_args()
root = args.directory
names = ["baseline", "rebuild", "nocopy", "tuned", "fastmath", "tile16", "spatial2",
         "vector_gelu", "vector_ln", "combined", "tuned_nosched"]
reports = {name: json.loads((root / f"{name}.json").read_text()) for name in names}
summary = {"variants": {}, "batch_sizes": {}, "stage_ms": {}, "pmu": {}}
lines = ["# Measured ablations: gh051 / job 3418136", "",
         "Median of three timed runs. Single-core: 32 real sources; 72 cores: 512 real sources. Same-node speedups only.", "",
         "| Variant | s4 1 core | s4 72 cores | AttnW 1 core | AttnW 72 cores | e/R means pass |",
         "|---|---:|---:|---:|---:|---|"]
for name, report in reports.items():
    rows = report["models"]
    rates = [rows[model][i]["gal_per_sec"] for model in ("s4", "attnw") for i in (0, 1)]
    passed = all(rows[model][1]["accuracy_vs_strict_gpu"][key]["mean_bias_passed"]
                 for model in ("s4", "attnw") for key in ("e", "R"))
    summary["variants"][name] = {"rates": dict(zip(("s4_1", "s4_72", "attnw_1", "attnw_72"), rates)),
                                  "means_passed": passed}
    lines.append("| " + name + " | " + " | ".join(f"{x:.3f}" for x in rates) + f" | {passed} |")
for name in ("baseline_balanced", "tuned_balanced", "combined_balanced"):
    report = json.loads((root / f"{name}.json").read_text())
    summary["batch_sizes"][name] = {model: {str(row["count"]): row["gal_per_sec"] for row in rows}
                                    for model, rows in report["models"].items()}
for name in ("profile", "combined_profile", "profile_tuned", "vector_gelu_profile"):
    report = json.loads((root / f"{name}.json").read_text())
    summary["stage_ms"][name] = {model: {str(row["threads"]): {stage: v["ms_per_source"]
        for stage, v in row["profile"].items()} for row in rows} for model, rows in report["models"].items()}
for path in root.glob("perf*.csv"):
    counters = {}
    for line in path.read_text().splitlines():
        parts = line.split(",")
        if len(parts) >= 3 and parts[0].isdigit():
            counters[parts[2]] = int(parts[0])
    if "cycles:u" in counters and "instructions:u" in counters:
        counters["IPC"] = counters["instructions:u"] / counters["cycles:u"]
        for key in ("stall_backend:u", "stall_backend_mem:u"):
            if key in counters:
                counters[key + "_per_cycle"] = counters[key] / counters["cycles:u"]
    summary["pmu"][path.stem] = counters
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 10, "savefig.dpi": 180})
fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
stages = [("Convolution", ("conv_first", "conv_hidden")), ("GELU", ("gelu",)),
          ("LayerNorm", ("layernorm",)), ("Residual", ("residual",)),
          ("Attention", ("attention",)), ("Other", ())]
colors = plt.get_cmap("tab10").colors
for ax, model in zip(axes, ("s4", "attnw")):
    bottom = np.zeros(2)
    for i, (label, keys) in enumerate(stages):
        values = []
        for name in ("profile", "combined_profile"):
            data = summary["stage_ms"][name][model]["1"]
            value = sum(data[k] for k in keys) if keys else data["total"] - sum(data[k] for _, group in stages[:-1] for k in group)
            values.append(value)
        ax.bar([0, 1], values, bottom=bottom, label=label, color=colors[i])
        bottom += values
    ax.set_xticks([0, 1], ["Baseline", "Vector GELU + LN\n+ no skip copy"])
    ax.set_ylabel("Single-core time (ms / source)")
    ax.set_title(model)
    ax.set_ylim(0, 215)
    for i, total in enumerate(bottom):
        ax.text(i, total + 2, f"{total:.1f}", ha="center")
axes[1].legend(loc="upper right", fontsize=8)
fig.suptitle("Where the CPU time goes: same trained models, fp32, T=2")
for ext in ("png", "pdf"):
    fig.savefig(root / f"stage_breakdown.{ext}")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
labels = ["Baseline", "Rebuild", "No skip copy", "Neoverse-v2 flags", "+ fast-math", "16-channel tile",
          "2-pixel tile", "Vector GELU", "Vector LayerNorm", "Combined", "V2 / no scheduling"]
for ax, model in zip(axes, ("s4", "attnw")):
    values = [summary["variants"][name]["rates"][model + "_72"] for name in names]
    ax.barh(range(len(names)), values, color=["#d9822b" if name == "combined" else "#387bb2" for name in names])
    ax.set_yticks(range(len(names)), labels)
    ax.invert_yaxis()
    ax.axvspan(500, 580, color="green", alpha=.12, label="Original forecast")
    ax.set_xlim(0, 610)
    ax.set_xlabel("Sources / second (72 cores, batch 512)")
    ax.set_title(model)
    for i, value in enumerate(values):
        ax.text(value + 4, i, f"{value:.1f}", va="center", fontsize=8)
axes[1].legend(loc="lower right", fontsize=8)
for ext in ("png", "pdf"):
    fig.savefig(root / f"variant_throughput.{ext}")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
for ax, model in zip(axes, ("s4", "attnw")):
    for name, label in (("baseline_balanced", "Baseline"), ("combined_balanced", "Combined")):
        data = summary["batch_sizes"][name][model]
        ax.plot(range(len(data)), list(data.values()), "o-", label=label)
        ax.set_xticks(range(len(data)), list(data))
    ax.axhspan(500, 580, color="green", alpha=.12)
    ax.set_title(model)
    ax.set_xlabel("Batch size (cycling the same 512 real sources)")
    ax.set_ylabel("Sources / second, 72 cores")
    ax.legend()
for ext in ("png", "pdf"):
    fig.savefig(root / f"batch_throughput.{ext}")
plt.close(fig)
