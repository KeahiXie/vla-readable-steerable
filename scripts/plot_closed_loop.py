"""Plot where the bowl landed in closed-loop runs, one cell per state and condition, and write the counts.

Arguments are one or more closed_loop folders. Rollouts are combined, and the outputs go in the last folder.
Conditions with no rollouts are left off.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

LABELS = {
    "none": "no edit",
    "word_natural": "word part\nnatural size",
    "word_equal_size": "word part\nscaled to full size",
    "random_full_size": "random,\nfull size",
    "complement": "everything\nelse",
    "full_after_word": "full swap",
    "other_instruction": "other instruction,\nno edit",
}
SURFACE, TEXT, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
COLOURS = {"own": "#eb6834", "other": "#2a78d6", "none": "#e4e3df", "missing": SURFACE}
INK = {"own": "#0b0b0b", "other": "#ffffff", "none": "#52514e", "missing": "#52514e"}
WORDS = {"own": "own target", "other": "other target", "none": "no landing", "missing": ""}

roots = [Path(folder) for folder in sys.argv[1:]]
root = roots[-1]
rows = [json.loads(line) for folder in roots for task in ("plate", "stove") for line in open(folder / task / "rollouts.jsonl")]
CONDITIONS = [c for c in LABELS if any(r["condition"] == c for r in rows)]
result = {(r["snapshot"], r["condition"]): "other" if r["landed_other"] else "own" if r["landed_own"] else "none" for r in rows}
states = sorted({r["snapshot"] for r in rows})
kept = [s for s in states if result.get((s, "none")) == "own"]
excluded = [s for s in states if s not in kept]

kinds = list(COLOURS)
grid = [[kinds.index(result.get((s, c), "missing")) for c in CONDITIONS] for s in kept]
figure, axis = plt.subplots(figsize=(1.9 * len(CONDITIONS), 0.42 * len(kept) + 2.2), facecolor=SURFACE)
axis.imshow(grid, cmap=ListedColormap([COLOURS[k] for k in kinds]), vmin=0, vmax=len(kinds) - 1, aspect="auto")
for y, state in enumerate(kept):
    for x, condition in enumerate(CONDITIONS):
        kind = result.get((state, condition), "missing")
        axis.text(x, y, WORDS[kind], ha="center", va="center", fontsize=9, color=INK[kind])
axis.set_xticks(range(len(CONDITIONS)), [LABELS[c] for c in CONDITIONS], fontsize=10, color=TEXT)
axis.xaxis.tick_top()
axis.set_yticks(range(len(kept)), [s.replace(".npy", "").replace("_seed", " ") for s in kept], fontsize=9, color=MUTED)
axis.tick_params(length=0)
axis.set_xticks([x - 0.5 for x in range(1, len(CONDITIONS))], minor=True)
axis.set_yticks([y - 0.5 for y in range(1, len(kept))], minor=True)
axis.grid(which="minor", color=SURFACE, linewidth=2)
axis.tick_params(which="minor", length=0)
for side in axis.spines.values():
    side.set_visible(False)
figure.suptitle(
    f"Where the bowl landed with the edit repeated at every generation. Exploratory: {len(kept)} tuning states",
    x=0.01, ha="left", fontsize=12, color=TEXT,
)
figure.tight_layout(rect=(0, 0, 1, 0.96))
figure.savefig(root / "landing_by_condition.png", dpi=170, facecolor=SURFACE)


def counts(chosen):
    table = {}
    for condition in CONDITIONS:
        found = [result[(s, condition)] for s in chosen if (s, condition) in result]
        table[condition] = {"n": len(found), **{k: found.count(k) for k in ("own", "other", "none")}}
    return table


used = [r for r in rows if r["snapshot"] in kept]
summary = {
    "status": "exploratory, tuning episodes only; not a gate or Stage 5 result",
    "states": len(kept),
    "excluded_no_edit_did_not_land_on_own_target": excluded,
    "max_chunks": max(r["generations"] for r in rows),
    "invalid_generations": sum(r["invalid_generations"] for r in used),
    "generations": sum(r["generations"] for r in used),
    "counts": counts(kept),
    "counts_plate_states": counts([s for s in kept if s.startswith("plate")]),
    "counts_stove_states": counts([s for s in kept if s.startswith("stove")]),
    "mean_word_fraction": {c: round(sum(r["mean_word_fraction"] for r in used if r["condition"] == c) / max(1, sum(r["condition"] == c for r in used)), 4) for c in ("full_after_word", "word_natural")},
}
(root / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary["counts"], indent=1))
print("excluded", excluded, "invalid", summary["invalid_generations"], "of", summary["generations"])
