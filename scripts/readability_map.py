"""Exploratory: where can a logit lens read the destination? Every layer, position 144 to the prompt end.

Value is lens logit " stove" minus " plate". Prompt lengths differ between frames, so the mean figure
aligns by token: the fixed text from the word, the state numbers averaged into one column, then the
fixed tail. The per-frame figure uses real positions. Recorded training frames, not Gate 1 evidence.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hooks import capture_output
from lens import logit_lens
from load_policy import load_config, load_policy
from snapshots import load_frames, make_batch, prompt_inputs

WORD_POSITION = 144
HEAD = 5  # the word, ",", " State", ":", " "
TAIL = 9  # ",", " Actions", ":", " ", end of turn, newline, "Ass", "istant", ":"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#1c5cab", "#f0efec", "#c23a35"])
SURFACE, TEXT, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config/cpu.yaml")
parser.add_argument("--frames", type=int, default=3)
parser.add_argument("--out", required=True)
args = parser.parse_args()

cfg = load_config(args.config)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
policy, pre, post, _ = load_policy(cfg["pins"], device="cpu")
vlm = policy.model.vlm
text_model = vlm.model.text_model
layers = text_model.layers
tokenizer = policy.model.processor.tokenizer
plate_id = tokenizer(" plate", add_special_tokens=False).input_ids[0]
stove_id = tokenizer(" stove", add_special_tokens=False).input_ids[0]
frames = load_frames(cfg["pins"], cfg["snapshots"])[: args.frames]

maps, labels = {}, {}
with open(out / "readout_by_position.csv", "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["episode_index", "instruction", "layer", "position", "token", "stove_minus_plate"])
    for frame in frames:
        episode = frame["episode_index"]
        for instruction, text in cfg["instructions"].items():
            inputs = prompt_inputs(policy, make_batch(frame, text, pre))
            ids = inputs["input_ids"][0].tolist()
            store = {}
            handles = [capture_output(layer, store, i) for i, layer in enumerate(layers)]
            with torch.no_grad():
                vlm(**inputs, use_cache=False)
            for hook in handles:
                hook.remove()

            values = np.zeros((len(layers), len(ids) - WORD_POSITION))
            with torch.no_grad():
                for i in range(len(layers)):
                    logits = logit_lens(store[i][:, WORD_POSITION:, :], text_model.norm, vlm.lm_head)[0]
                    values[i] = (logits[:, stove_id] - logits[:, plate_id]).numpy()
            maps[(episode, instruction)] = values
            labels[episode] = [tokenizer.decode([token]) for token in ids[WORD_POSITION:]]
            for i in range(len(layers)):
                for j, value in enumerate(values[i]):
                    writer.writerow([episode, instruction, i, WORD_POSITION + j, labels[episode][j], round(float(value), 4)])
            print(episode, instruction, "prompt length", len(ids), flush=True)


def aligned(values):
    """Head columns, state numbers averaged into one column, tail columns."""
    return np.concatenate([values[:, :HEAD], values[:, HEAD:-TAIL].mean(axis=1, keepdims=True), values[:, -TAIL:]], axis=1)


episodes = [frame["episode_index"] for frame in frames]
plate = np.mean([aligned(maps[(e, "plate")]) for e in episodes], axis=0)
stove = np.mean([aligned(maps[(e, "stove")]) for e in episodes], axis=0)
difference = stove - plate
tick_labels = ['word', ",", "State", ":", "␣", "state numbers (mean)", ",", "Actions", ":", "␣", "end of turn", "newline", "Ass", "istant", ":"]
np.savez(out / "aligned_means.npz", plate=plate, stove=stove, difference=difference)

figure, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True, facecolor=SURFACE)
panels = (("Plate instruction", plate), ("Stove instruction", stove), ("Stove run minus plate run", difference))
limit = max(np.abs(plate).max(), np.abs(stove).max())
for axis, (title, values) in zip(axes, panels):
    scale = limit if values is not difference else np.abs(difference).max()
    image = axis.imshow(values, origin="lower", aspect="auto", cmap=DIVERGING, vmin=-scale, vmax=scale, interpolation="nearest")
    axis.set_title(title, loc="left", fontsize=12, color=TEXT)
    axis.set_xticks(range(len(tick_labels)), tick_labels, rotation=60, ha="right", fontsize=9, color=MUTED)
    axis.set_yticks(range(0, len(layers), 4))
    axis.tick_params(length=0, colors=MUTED)
    for side in axis.spines.values():
        side.set_visible(False)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02, label='lens " stove" minus " plate"')
axes[0].set_ylabel("layer")
figure.suptitle(
    f"Where is the destination readable? Exploratory: mean of {len(episodes)} recorded training frames, logit lens only",
    x=0.01, ha="left", fontsize=13, color=TEXT,
)
figure.tight_layout(rect=(0, 0, 1, 0.96))
figure.savefig(out / "readability_map.png", dpi=150, facecolor=SURFACE)

figure, axes = plt.subplots(1, len(episodes), figsize=(5.5 * len(episodes), 6), sharey=True, facecolor=SURFACE)
scale = max(np.abs(maps[(e, "stove")] - maps[(e, "plate")]).max() for e in episodes)
for axis, episode in zip(np.atleast_1d(axes), episodes):
    values = maps[(episode, "stove")] - maps[(episode, "plate")]
    image = axis.imshow(values, origin="lower", aspect="auto", cmap=DIVERGING, vmin=-scale, vmax=scale, interpolation="nearest")
    axis.set_title(f"Episode {episode}: {values.shape[1]} positions", loc="left", color=TEXT)
    axis.set_xlabel("position, from the word onward", color=MUTED)
    axis.tick_params(length=0, colors=MUTED)
np.atleast_1d(axes)[0].set_ylabel("layer", color=MUTED)
figure.colorbar(image, ax=axes, fraction=0.02, pad=0.01, label="stove run minus plate run")
figure.savefig(out / "readability_map_per_frame.png", dpi=150, facecolor=SURFACE)

summary = {
    "status": "exploratory, not Gate evidence",
    "frames": episodes,
    "snapshot_source": cfg["snapshots"]["source"],
    "columns": tick_labels,
    "difference_by_column_layers_16_to_19_mean": dict(zip(tick_labels, difference[16:20].mean(axis=0).round(3).tolist())),
    "difference_last_position_by_layer": dict(zip(range(len(layers)), difference[:, -1].round(3).tolist())),
    "difference_word_by_layer": dict(zip(range(len(layers)), difference[:, 0].round(3).tolist())),
}
(out / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary["difference_by_column_layers_16_to_19_mean"], indent=2, ensure_ascii=False))
