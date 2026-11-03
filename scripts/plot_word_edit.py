"""Plot and summarize the word-only edit on saved simulator states. Exploratory, tuning episodes only."""
import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONDITIONS = ["full_after_word", "word_natural", "complement", "word_equal_size", "random_word_size", "random_full_size"]
LABELS = ["full swap\nafter word", "word part\nnatural size", "everything\nelse", "word part\nscaled to full size", "random,\nword-part size", "random,\nfull size"]
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
GAP_FLOOR = 0.5  # exclude states whose two baselines are nearly identical

folder = Path(sys.argv[1])
rows = list(csv.DictReader(open(folder / "rows.csv")))
lens = list(csv.DictReader(open(folder / "lens_reads.csv")))
states = sorted({r["snapshot"] for r in rows})
kept = [r for r in rows if float(r["base_gap_l2"]) >= GAP_FLOOR]
excluded = sorted({r["snapshot"] for r in rows if float(r["base_gap_l2"]) < GAP_FLOOR})
kept_states = {r["snapshot"] for r in kept}

figure, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=SURFACE)
# A move is a ratio, so it is meaningless when the two runs read almost the same. Those are dropped here.
READ_GAP_FLOOR = 1.0
last_19 = [
    m for m in lens
    if m["read_layer"] == "19" and m["scope"] == "last" and m["snapshot"] in kept_states
    and abs(float(m["read_donor"]) - float(m["read_base"])) >= READ_GAP_FLOOR
]
panels = [
    (axes[0], "Did the actions move?", "ρ: 0 = unmoved, 1 = other instruction", kept, "rho"),
    (axes[1], "Did the reading move? (layer 19, last prompt position)", "0 = unmoved, 1 = other run's reading", last_19, "move_toward_donor"),
]
for axis, title, ylabel, data, key in panels:
    axis.set_facecolor(SURFACE)
    axis.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
    axis.axhline(1, color=GRID, linewidth=1, zorder=1)
    for receiver, colour, marker, offset, label in (("plate", BLUE, "o", -0.1, "edit toward stove in plate run"), ("stove", ORANGE, "s", 0.1, "edit toward plate in stove run")):
        chosen = [d for d in data if d["receiver"] == receiver]
        axis.scatter(
            [CONDITIONS.index(d["condition"]) + offset for d in chosen], [float(d[key]) for d in chosen],
            s=44, color=colour, marker=marker, edgecolors=SURFACE, linewidths=1, alpha=0.9, zorder=3, label=label,
        )
    axis.set_xticks(range(len(CONDITIONS)), LABELS, fontsize=9, color=MUTED)
    axis.set_title(title, loc="left", fontsize=12, color=TEXT)
    axis.set_ylabel(ylabel, color=MUTED)
    axis.tick_params(length=0, colors=MUTED)
    axis.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
axes[0].legend(frameon=False, labelcolor=MUTED)
axes[1].set_ylim(-1.5, 3.5)
axes[1].annotate("off scale above", (CONDITIONS.index("word_equal_size"), 3.3), ha="center", fontsize=9, color=MUTED)
figure.suptitle(
    f"Word-only edit on saved simulator states. Exploratory: {len(kept_states)} tuning states, written actions only, "
    f"{sum(r['valid'] != 'True' for r in kept)} invalid of {len(kept)}",
    x=0.01, ha="left", fontsize=12, color=TEXT,
)
figure.tight_layout(rect=(0, 0, 1, 0.95))
figure.savefig(folder / "word_edit_on_snapshots.png", dpi=170, facecolor=SURFACE)


def stats(values):
    values = [v for v in values if v == v]
    if not values:
        return None
    return {"n": len(values), "mean": round(statistics.mean(values), 3), "median": round(statistics.median(values), 3), "min": min(values), "max": max(values)}


by_state = {}
for r in kept:
    by_state.setdefault((r["snapshot"], r["receiver"]), {})[r["condition"]] = float(r["rho"])
summary = {
    "status": "exploratory, tuning episodes only, written actions only; not a gate or Stage 5 result",
    "states_run": len(states),
    "states_kept": len(kept_states),
    "excluded_nearly_identical_baselines": excluded,
    "gap_floor_l2": GAP_FLOOR,
    "base_gap_l2": stats([float(r["base_gap_l2"]) for r in rows if r["condition"] == "full_after_word"]),
    "word_fraction_of_norm": stats([float(r["word_fraction_of_norm"]) for r in kept if r["condition"] == "full_after_word"]),
    "invalid_outputs": sum(r["valid"] != "True" for r in kept),
    "rho": {c: stats([float(r["rho"]) for r in kept if r["condition"] == c]) for c in CONDITIONS},
    "rho_first_action": {c: stats([float(r["rho_first_action"]) for r in kept if r["condition"] == c]) for c in CONDITIONS},
    "interaction_full_minus_word_minus_complement": stats(
        [v["full_after_word"] - v["word_natural"] - v["complement"] for v in by_state.values() if len(v) == len(CONDITIONS)]
    ),
    "reading_move_median": {
        f"{c}_layer{layer}_{scope}": (lambda s: s and s["median"])(
            stats([float(m["move_toward_donor"]) for m in lens if m["condition"] == c and m["read_layer"] == layer and m["scope"] == scope and m["snapshot"] in kept_states])
        )
        for c in CONDITIONS
        for layer, scope in (("17", "last"), ("19", "last"), ("22", "last"), ("31", "last"))
    },
}
(folder / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
