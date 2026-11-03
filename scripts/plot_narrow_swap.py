"""Plot rho by layer for each narrow-swap scope, and write the numbers behind the plot as a table."""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, CONTROL = "#2a78d6", "#eb6834", "#8a8983"
SCOPES = {
    "last": "Last prompt position only",
    "word": 'Destination word only (position 144)',
    "rest": "All positions after the word",
}
DIRECTIONS = {("plate", "stove"): (BLUE, "o", "stove swapped into plate run"), ("stove", "plate"): (ORANGE, "s", "plate swapped into stove run")}

narrow_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "evidence/preview_narrow_swap")
full_path = Path(sys.argv[2] if len(sys.argv) > 2 else "evidence/preview_full_swap/rows.csv")
rows = list(csv.DictReader(open(narrow_dir / "rows.csv")))
frames = sorted({r["episode_index"] for r in rows})
layers = sorted({int(r["layer"]) for r in rows if r["scope"] != "full"})
index = {layer: i for i, layer in enumerate(layers)}

# Reference: full swap at each layer, from this run or the earlier preview, on the same frames.
full = defaultdict(list)
for r in rows:
    if r["scope"] == "full":
        full[int(r["layer"])].append(float(r["rho"]))
for r in csv.DictReader(open(full_path)):
    if r["condition"] == "full_swap" and r["episode_index"] in frames and int(r["layer"]) in index:
        full[int(r["layer"])].append(float(r["rho"]))

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "text.color": TEXT, "axes.labelcolor": MUTED})
figure, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True, facecolor=SURFACE)
for axis, (scope, title) in zip(axes, SCOPES.items()):
    axis.set_facecolor(SURFACE)
    axis.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
    axis.axhline(1, color=GRID, linewidth=1, zorder=1)
    x = [index[layer] for layer in layers if full[layer]]
    y = [statistics.mean(full[layer]) for layer in layers if full[layer]]
    axis.plot(x, y, color=TEXT, linewidth=1, linestyle=(0, (4, 3)), zorder=2, label="full swap (mean)")
    for (receiver, donor), (colour, marker, label) in DIRECTIONS.items():
        chosen = [r for r in rows if r["scope"] == scope and r["condition"] == "swap" and r["receiver"] == receiver]
        offset = -0.09 if receiver == "plate" else 0.09
        axis.scatter(
            [index[int(r["layer"])] + offset for r in chosen], [float(r["rho"]) for r in chosen],
            s=46, color=colour, marker=marker, edgecolors=SURFACE, linewidths=1, zorder=4, label=label,
        )
    controls = [r for r in rows if r["scope"] == scope and r["condition"] == "random_same_norm"]
    axis.scatter(
        [index[int(r["layer"])] + 0.26 for r in controls], [float(r["rho"]) for r in controls],
        s=32, color=CONTROL, marker="x", linewidths=1.2, zorder=3, label="random edit, same size",
    )
    axis.set_title(title, fontsize=11, loc="left", color=TEXT)
    axis.set_xticks(range(len(layers)), [str(layer) for layer in layers])
    axis.set_xlim(-0.5, len(layers) - 0.4)
    axis.set_xlabel("patched layer")
    axis.tick_params(colors=MUTED, length=0)
    axis.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
axes[0].set_ylabel("ρ: 0 = unmoved, 1 = other instruction")
handles, labels = axes[0].get_legend_handles_labels()
figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False, labelcolor=MUTED)
invalid = sum(r["valid"] != "True" for r in rows)
figure.suptitle(
    f"Where does the destination live? Exploratory: {len(frames)} recorded training frames, written actions only, "
    f"{invalid} invalid outputs of {len(rows)}",
    x=0.01, ha="left", fontsize=12, color=TEXT,
)
figure.tight_layout(rect=(0, 0.08, 1, 0.96))
figure.savefig(narrow_dir / "rho_by_layer_and_scope.png", dpi=170, facecolor=SURFACE)

with open(narrow_dir / "summary_by_cell.csv", "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["scope", "condition", "layer", "n", "mean_rho", "min_rho", "max_rho", "invalid"])
    keys = sorted({(r["scope"], r["condition"], int(r["layer"])) for r in rows})
    for scope, condition, layer in keys:
        chosen = [r for r in rows if (r["scope"], r["condition"], int(r["layer"])) == (scope, condition, layer)]
        values = [float(r["rho"]) for r in chosen]
        writer.writerow(
            [scope, condition, layer, len(values), round(statistics.mean(values), 3), min(values), max(values),
             sum(r["valid"] != "True" for r in chosen)]
        )
print("wrote", narrow_dir / "rho_by_layer_and_scope.png")
