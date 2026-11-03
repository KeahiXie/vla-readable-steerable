"""Exploratory: does swapping a mid-layer's prompt activations move the generated actions?

Recorded training frames with no simulator, so this is a first look and not Gate 2 evidence.
rho = (a_edit - a) . (a_cf - a) / |a_cf - a|^2, where 0 means unmoved and 1 means fully at the
other instruction's action. Rows are written as they finish.
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
from hooks import capture_output, replace_prompt_output
from load_policy import load_config, load_policy
from snapshots import load_frames, make_batch, prompt_inputs

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config/cpu.yaml")
parser.add_argument("--frames", type=int, default=5)
parser.add_argument("--layers", default="8,16,19,24,31")
parser.add_argument("--control-layer", type=int, default=16)
parser.add_argument("--out", required=True)
args = parser.parse_args()

cfg = load_config(args.config)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
swap_layers = [int(x) for x in args.layers.split(",")]

policy, pre, post, _ = load_policy(cfg["pins"], device="cpu")
model = policy.model
layers = model.vlm.model.text_model.layers
frames = load_frames(cfg["pins"], cfg["snapshots"])[: args.frames]
generator = torch.Generator().manual_seed(cfg["seed"])

# Record the raw generated text, because invalid text silently becomes bin 0.
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
    return actions, valid


def capture_all(inputs):
    store = {}
    handles = [capture_output(layers[i], store, i) for i in {*swap_layers, args.control_layer}]
    with torch.no_grad():
        model.vlm(**inputs, use_cache=False)
    for handle in handles:
        handle.remove()
    return store


def patched(batch, layer, replacement):
    handle = replace_prompt_output(layers[layer], replacement)
    actions, valid = generate(batch)
    handle.remove()
    return actions, valid


def rho(edit, base, counterfactual):
    direction = (counterfactual - base).flatten()
    return (((edit - base).flatten() @ direction) / (direction @ direction)).item()


file = open(out / "rows.csv", "w", newline="")
writer = csv.DictWriter(
    file,
    ["episode_index", "receiver", "donor", "condition", "layer", "rho", "rho_first_action", "l2_from_base", "base_gap_l2", "valid"],
)
writer.writeheader()
rows, identity, started = [], [], time.perf_counter()


def record(episode, receiver, donor, condition, layer, edit, base, counterfactual, valid):
    row = {
        "episode_index": episode,
        "receiver": receiver,
        "donor": donor,
        "condition": condition,
        "layer": layer,
        "rho": round(rho(edit, base, counterfactual), 4),
        "rho_first_action": round(rho(edit[0], base[0], counterfactual[0]), 4),
        "l2_from_base": round((edit - base).norm().item(), 4),
        "base_gap_l2": round((counterfactual - base).norm().item(), 4),
        "valid": valid,
    }
    rows.append(row)
    writer.writerow(row)
    file.flush()
    print(row, f"{time.perf_counter() - started:.0f}s", flush=True)


for frame_number, frame in enumerate(frames):
    episode = frame["episode_index"]
    batches = {name: make_batch(frame, text, pre) for name, text in cfg["instructions"].items()}
    inputs = {name: prompt_inputs(policy, batch) for name, batch in batches.items()}
    stores = {name: capture_all(value) for name, value in inputs.items()}
    bases = {name: generate(batch)[0] for name, batch in batches.items()}

    for receiver, donor in (("plate", "stove"), ("stove", "plate")):
        batch, base, counterfactual = batches[receiver], bases[receiver], bases[donor]

        if frame_number == 0:
            actions, _ = patched(batch, args.control_layer, stores[receiver][args.control_layer])
            identity.append({"receiver": receiver, "self_patch_identical": bool(torch.equal(actions, base))})
            print(identity[-1], flush=True)

        for layer in swap_layers:
            actions, valid = patched(batch, layer, stores[donor][layer])
            record(episode, receiver, donor, "full_swap", layer, actions, base, counterfactual, valid)

        # Control: a random edit of the same size, on the positions that differ between instructions.
        h, delta = stores[receiver][args.control_layer], stores[donor][args.control_layer] - stores[receiver][args.control_layer]
        noise = torch.randn(h.shape, generator=generator) * (delta != 0).any(dim=-1, keepdim=True)
        noise = noise * delta.norm() / noise.norm()
        actions, valid = patched(batch, args.control_layer, h + noise)
        record(episode, receiver, donor, "random_same_norm", args.control_layer, actions, base, counterfactual, valid)

file.close()
summary = {
    "frames": [frame["episode_index"] for frame in frames],
    "snapshot_source": cfg["snapshots"]["source"],
    "patch_scope": "prompt pass only, all prompt positions (only positions 144 and later differ)",
    "self_patch_during_generation": identity,
    "invalid_outputs": sum(not row["valid"] for row in rows),
    "n_rows": len(rows),
    "total_seconds": round(time.perf_counter() - started, 1),
}
(out / "summary.json").write_text(json.dumps(summary, indent=2))

figure, axis = plt.subplots(figsize=(7, 4))
for receiver, colour in (("plate", "#1f77b4"), ("stove", "#d95f02")):
    chosen = [r for r in rows if r["receiver"] == receiver and r["condition"] == "full_swap"]
    axis.scatter(
        [r["layer"] for r in chosen], [r["rho"] for r in chosen], color=colour, alpha=0.7, label=f"full swap into {receiver} run"
    )
controls = [r for r in rows if r["condition"] == "random_same_norm"]
axis.scatter([r["layer"] + 0.6 for r in controls], [r["rho"] for r in controls], color="#777777", marker="x", label="random, same size")
axis.axhline(0, color="#888888", linewidth=0.8)
axis.axhline(1, color="#888888", linewidth=0.8, linestyle="--")
axis.set_xlabel("patched layer")
axis.set_ylabel("rho (0 = unmoved, 1 = other instruction's actions)")
axis.set_title(f"Full-swap preview on {len(frames)} recorded frames (exploratory)")
axis.legend(frameon=False)
figure.tight_layout()
figure.savefig(out / "rho_by_layer.png", dpi=150)
print(json.dumps(summary, indent=2))
