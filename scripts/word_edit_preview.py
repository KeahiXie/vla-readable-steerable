"""Exploratory word-only edit: split the swap at one layer into its word-readable part and the rest.

At each position after the word, delta = h_other - h. W is the span of norm.weight * unembedding row
for " plate" and " stove", the directions the lens reads in h. Each edit is applied separately:
  full_after_word   h + delta
  word_natural      h + P_W delta
  complement        h + (I - P_W) delta
  word_equal_size   the word part scaled up to the size of delta
  random_word_size, random_full_size   controls matched to those two sizes
Also records whether each edit changed the lens reading, because if it didn't, rho near 0 means
nothing. Recorded training frames with no simulator, so this is not a Stage 5 result.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hooks import capture_output, replace_output, replace_prompt_output
from lens import logit_lens
from load_policy import load_config, load_policy
from snapshots import load_frames, make_batch, prompt_inputs

WORD_POSITION = 144
READ_LAYERS = [17, 19, 22, 31]
CONDITIONS = ["full_after_word", "word_natural", "complement", "word_equal_size", "random_word_size", "random_full_size"]
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config/cpu.yaml")
parser.add_argument("--frames", type=int, default=3)
parser.add_argument("--layer", type=int, default=16)
parser.add_argument("--out", required=True)
args = parser.parse_args()

cfg = load_config(args.config)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
policy, pre, post, _ = load_policy(cfg["pins"], device="cpu")
model = policy.model
vlm = model.vlm
text_model = vlm.model.text_model
layers = text_model.layers
tokenizer = model.processor.tokenizer
frames = load_frames(cfg["pins"], cfg["snapshots"])[: args.frames]
generator = torch.Generator().manual_seed(cfg["seed"])
plate_id = tokenizer(" plate", add_special_tokens=False).input_ids[0]
stove_id = tokenizer(" stove", add_special_tokens=False).input_ids[0]

# The lens computes w . (norm.weight * h) / rms(h), so its direction in h is norm.weight * w.
with torch.no_grad():
    directions = text_model.norm.weight * vlm.lm_head.weight[[plate_id, stove_id]]
    basis, _ = torch.linalg.qr(directions.T)

raw_text = []
original_decode = model.processor.batch_decode


def recording_decode(*a, **k):
    decoded = original_decode(*a, **k)
    raw_text.append(decoded[0])
    return decoded


model.processor.batch_decode = recording_decode


def generate(batch):
    raw_text.clear()
    with torch.no_grad():
        actions = model.generate_actions(batch)[0].cpu()
    numbers = raw_text[-1].strip().split()
    valid = len(numbers) == actions.numel() and all(n.isdigit() and 0 <= int(n) < 512 for n in numbers)
    return actions, valid, raw_text[-1].strip()


def rho(edit, base, counterfactual):
    direction = (counterfactual - base).flatten()
    return (((edit - base).flatten() @ direction) / (direction @ direction)).item()


def read(h, positions):
    """Mean lens " stove" minus " plate" over the given positions."""
    with torch.no_grad():
        logits = logit_lens(h[:, positions, :], text_model.norm, vlm.lm_head)[0]
    return (logits[:, stove_id] - logits[:, plate_id]).mean().item()


def capture(inputs, edited=None):
    """Prompt-only pass, optionally with the edit applied, returning the layers the lens reads."""
    store = {}
    handles = [capture_output(layers[i], store, i) for i in READ_LAYERS]
    if edited is not None:
        handles.append(replace_output(layers[args.layer], edited))
    else:
        handles.append(capture_output(layers[args.layer], store, args.layer))
    with torch.no_grad():
        vlm(**inputs, use_cache=False)
    for handle in handles:
        handle.remove()
    if edited is not None:
        store[args.layer] = edited
    return store


rows_file = open(out / "rows.csv", "w", newline="")
rows_writer = csv.DictWriter(
    rows_file,
    ["episode_index", "receiver", "donor", "condition", "layer", "rho", "rho_first_action", "l2_from_base", "base_gap_l2", "edit_norm", "valid", "raw"],
)
rows_writer.writeheader()
lens_file = open(out / "lens_reads.csv", "w", newline="")
lens_writer = csv.writer(lens_file)
lens_writer.writerow(["episode_index", "receiver", "condition", "read_layer", "scope", "read_edit", "read_base", "read_donor", "move_toward_donor"])
rows, moves, sizes, identity, started = [], [], [], [], time.perf_counter()

for frame_number, frame in enumerate(frames):
    episode = frame["episode_index"]
    batches = {name: make_batch(frame, text, pre) for name, text in cfg["instructions"].items()}
    inputs = {name: prompt_inputs(policy, batch) for name, batch in batches.items()}
    length = inputs["plate"]["input_ids"].shape[1]
    after_word = list(range(WORD_POSITION + 1, length))
    stores = {name: capture(value) for name, value in inputs.items()}
    bases = {name: generate(batch)[0] for name, batch in batches.items()}

    for receiver, donor in (("plate", "stove"), ("stove", "plate")):
        batch, base, counterfactual = batches[receiver], bases[receiver], bases[donor]
        h = stores[receiver][args.layer]

        if frame_number == 0:
            handle = replace_prompt_output(layers[args.layer], h)
            same = torch.equal(generate(batch)[0], base)
            handle.remove()
            identity.append({"receiver": receiver, "self_patch_identical": bool(same)})
            print(identity[-1], flush=True)
            assert same, "self-patch changed the output"

        delta = torch.zeros_like(h)
        delta[:, after_word] = stores[donor][args.layer][:, after_word] - h[:, after_word]
        word_part = delta @ basis @ basis.T
        complement = delta - word_part
        assert (word_part + complement - delta).abs().max() < 1e-4
        sizes.append(
            {
                "episode_index": episode,
                "receiver": receiver,
                "delta_norm": round(delta.norm().item(), 3),
                "word_part_norm": round(word_part.norm().item(), 3),
                "word_fraction_of_norm": round((word_part.norm() / delta.norm()).item(), 4),
            }
        )
        print(sizes[-1], flush=True)

        noise = torch.zeros_like(h)
        noise[:, after_word] = torch.randn(h[:, after_word].shape, generator=generator)
        noise = noise / noise.norm()
        edits = {
            "full_after_word": delta,
            "word_natural": word_part,
            "complement": complement,
            "word_equal_size": word_part * delta.norm() / word_part.norm(),
            "random_word_size": noise * word_part.norm(),
            "random_full_size": noise * delta.norm(),
        }

        for condition in CONDITIONS:
            edited = h + edits[condition]
            handle = replace_prompt_output(layers[args.layer], edited)
            actions, valid, raw = generate(batch)
            handle.remove()
            row = {
                "episode_index": episode,
                "receiver": receiver,
                "donor": donor,
                "condition": condition,
                "layer": args.layer,
                "rho": round(rho(actions, base, counterfactual), 4),
                "rho_first_action": round(rho(actions[0], base[0], counterfactual[0]), 4),
                "l2_from_base": round((actions - base).norm().item(), 4),
                "base_gap_l2": round((counterfactual - base).norm().item(), 4),
                "edit_norm": round(edits[condition].norm().item(), 3),
                "valid": valid,
                "raw": raw,
            }
            rows.append(row)
            rows_writer.writerow(row)
            rows_file.flush()
            print({k: v for k, v in row.items() if k != "raw"}, f"{time.perf_counter() - started:.0f}s", flush=True)

            edited_store = capture(inputs[receiver], edited)
            for read_layer in [args.layer, *READ_LAYERS]:
                for scope, positions in (("after_word", after_word), ("last", [length - 1])):
                    if read_layer == args.layer and scope == "last":
                        continue
                    value = read(edited_store[read_layer], positions)
                    start = read(stores[receiver][read_layer], positions)
                    goal = read(stores[donor][read_layer], positions)
                    move = (value - start) / (goal - start) if abs(goal - start) > 1e-6 else float("nan")
                    lens_writer.writerow([episode, receiver, condition, read_layer, scope, round(value, 4), round(start, 4), round(goal, 4), round(move, 4)])
                    moves.append({"condition": condition, "receiver": receiver, "read_layer": read_layer, "scope": scope, "move": move})
            lens_file.flush()

rows_file.close()
lens_file.close()

labels = {
    "full_after_word": "full swap\nafter word",
    "word_natural": "word part\nnatural size",
    "complement": "everything\nelse",
    "word_equal_size": "word part\nscaled to full size",
    "random_word_size": "random,\nword-part size",
    "random_full_size": "random,\nfull size",
}
figure, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=SURFACE)
panels = [
    (axes[0], "Did the actions move?", "ρ: 0 = unmoved, 1 = other instruction", rows, "rho"),
    (axes[1], f"Did the lens reading move? (layer {args.layer}, edited positions)", "0 = unmoved, 1 = other run's reading",
     [m for m in moves if m["read_layer"] == args.layer and m["scope"] == "after_word"], "move"),
]
for axis, title, ylabel, data, key in panels:
    axis.set_facecolor(SURFACE)
    axis.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
    axis.axhline(1, color=GRID, linewidth=1, zorder=1)
    for receiver, colour, marker, offset, label in (("plate", BLUE, "o", -0.1, "edit toward stove in plate run"), ("stove", ORANGE, "s", 0.1, "edit toward plate in stove run")):
        chosen = [d for d in data if d["receiver"] == receiver]
        axis.scatter(
            [CONDITIONS.index(d["condition"]) + offset for d in chosen], [float(d[key]) for d in chosen],
            s=52, color=colour, marker=marker, edgecolors=SURFACE, linewidths=1, zorder=3, label=label,
        )
    axis.set_xticks(range(len(CONDITIONS)), [labels[c] for c in CONDITIONS], fontsize=9, color=MUTED)
    axis.set_title(title, loc="left", fontsize=12, color=TEXT)
    axis.set_ylabel(ylabel, color=MUTED)
    axis.tick_params(length=0, colors=MUTED)
    axis.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
axes[0].legend(frameon=False, labelcolor=MUTED, loc="upper right")
figure.suptitle(
    f"Word-only edit at layer {args.layer}. Exploratory: {len(frames)} recorded training frames, written actions only, "
    f"{sum(not r['valid'] for r in rows)} invalid of {len(rows)}",
    x=0.01, ha="left", fontsize=12, color=TEXT,
)
figure.tight_layout(rect=(0, 0, 1, 0.95))
figure.savefig(out / "word_edit.png", dpi=170, facecolor=SURFACE)


def mean(values):
    return round(sum(values) / len(values), 3) if values else None


by_frame = {}
for row in rows:
    by_frame.setdefault((row["episode_index"], row["receiver"]), {})[row["condition"]] = row["rho"]
summary = {
    "status": "exploratory, not a Stage 5 result",
    "frames": [frame["episode_index"] for frame in frames],
    "layer": args.layer,
    "positions": "after the word through the prompt end",
    "patch": "prompt pass only; generated tokens run unedited",
    "word_subspace": 'span of norm.weight * unembedding row for " plate" and " stove"',
    "self_patch_during_generation": identity,
    "invalid_outputs": sum(not row["valid"] for row in rows),
    "sizes": sizes,
    "mean_rho": {c: mean([r["rho"] for r in rows if r["condition"] == c]) for c in CONDITIONS},
    "mean_lens_move": {
        f"{c}_layer{layer}_{scope}": mean([m["move"] for m in moves if m["condition"] == c and m["read_layer"] == layer and m["scope"] == scope and m["move"] == m["move"]])
        for c in CONDITIONS
        for layer, scope in ((args.layer, "after_word"), (19, "after_word"), (19, "last"), (31, "last"))
    },
    "interaction_full_minus_word_minus_complement": {
        f"{e}_{r}": round(v["full_after_word"] - v["word_natural"] - v["complement"], 4) for (e, r), v in by_frame.items()
    },
    "total_seconds": round(time.perf_counter() - started, 1),
}
(out / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
