"""Exploratory: which prompt positions carry the destination at each layer?

Replaces only some positions of a layer's prompt-pass output with the other instruction's values:
  last   the last prompt position
  word   position 144, the destination word itself
  rest   positions 145 to the prompt end, everything after the word
A matched control adds random noise to the same positions with the same total size. The reference
scope, full, swaps every differing position (144 to the prompt end). Recorded training frames with
no simulator, so this is not Gate 2 evidence.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hooks import capture_output, replace_prompt_output
from load_policy import load_config, load_policy
from snapshots import load_frames, make_batch, prompt_inputs

WORD_POSITION = 144

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config/cpu.yaml")
parser.add_argument("--frames", type=int, default=3)
parser.add_argument("--skip-frames", type=int, default=0, help="continue a run: skip these frames and append")
parser.add_argument("--layers", default="8,16,19,22")
parser.add_argument("--full-layers", default="22", help="layers lacking a full-swap reference from the earlier preview")
parser.add_argument("--out", required=True)
args = parser.parse_args()

cfg = load_config(args.config)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
swap_layers = [int(x) for x in args.layers.split(",")]
full_layers = [int(x) for x in args.full_layers.split(",") if x]

policy, pre, post, _ = load_policy(cfg["pins"], device="cpu")
model = policy.model
layers = model.vlm.model.text_model.layers
frames = load_frames(cfg["pins"], cfg["snapshots"])[args.skip_frames : args.frames]
generator = torch.Generator().manual_seed(cfg["seed"] + args.skip_frames)

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


def patched(batch, layer, replacement):
    handle = replace_prompt_output(layers[layer], replacement)
    result = generate(batch)
    handle.remove()
    return result


def rho(edit, base, counterfactual):
    direction = (counterfactual - base).flatten()
    return (((edit - base).flatten() @ direction) / (direction @ direction)).item()


file = open(out / "rows.csv", "a" if args.skip_frames else "w", newline="")
writer = csv.DictWriter(
    file,
    ["episode_index", "receiver", "donor", "scope", "condition", "layer", "rho", "rho_first_action", "l2_from_base", "base_gap_l2", "valid", "raw"],
)
if not args.skip_frames:
    writer.writeheader()
rows, identity, started = [], [], time.perf_counter()


def record(episode, receiver, donor, scope, condition, layer, result, base, counterfactual):
    actions, valid, raw = result
    row = {
        "episode_index": episode,
        "receiver": receiver,
        "donor": donor,
        "scope": scope,
        "condition": condition,
        "layer": layer,
        "rho": round(rho(actions, base, counterfactual), 4),
        "rho_first_action": round(rho(actions[0], base[0], counterfactual[0]), 4),
        "l2_from_base": round((actions - base).norm().item(), 4),
        "base_gap_l2": round((counterfactual - base).norm().item(), 4),
        "valid": valid,
        "raw": raw,
    }
    rows.append(row)
    writer.writerow(row)
    file.flush()
    print({k: v for k, v in row.items() if k != "raw"}, f"{time.perf_counter() - started:.0f}s", flush=True)


for frame_number, frame in enumerate(frames):
    episode = frame["episode_index"]
    batches = {name: make_batch(frame, text, pre) for name, text in cfg["instructions"].items()}
    inputs = {name: prompt_inputs(policy, batch) for name, batch in batches.items()}

    differing = (inputs["plate"]["input_ids"][0] != inputs["stove"]["input_ids"][0]).nonzero().flatten().tolist()
    assert differing == [WORD_POSITION], differing
    # State numbers are written digit by digit, so prompt length changes from frame to frame.
    length = inputs["plate"]["input_ids"].shape[1]
    scopes = {"last": [length - 1], "word": [WORD_POSITION], "rest": list(range(WORD_POSITION + 1, length))}
    print("frame", episode, "prompt length", length, flush=True)

    stores = {}
    for name, value in inputs.items():
        store = {}
        handles = [capture_output(layers[i], store, i) for i in {*swap_layers, *full_layers}]
        with torch.no_grad():
            model.vlm(**value, use_cache=False)
        for handle in handles:
            handle.remove()
        stores[name] = store
    bases = {name: generate(batch)[0] for name, batch in batches.items()}

    for receiver, donor in (("plate", "stove"), ("stove", "plate")):
        batch, base, counterfactual = batches[receiver], bases[receiver], bases[donor]

        if frame_number == 0:
            same = torch.equal(patched(batch, swap_layers[0], stores[receiver][swap_layers[0]])[0], base)
            identity.append({"receiver": receiver, "self_patch_identical": bool(same)})
            print(identity[-1], flush=True)
            assert same, "self-patch changed the output"

        for layer in swap_layers:
            h, other = stores[receiver][layer], stores[donor][layer]
            for scope, positions in scopes.items():
                edited = h.clone()
                edited[:, positions] = other[:, positions]
                record(episode, receiver, donor, scope, "swap", layer, patched(batch, layer, edited), base, counterfactual)

                noise = torch.zeros_like(h)
                noise[:, positions] = torch.randn(h[:, positions].shape, generator=generator)
                size = (other[:, positions] - h[:, positions]).norm()
                control = h + noise * size / noise.norm()
                record(episode, receiver, donor, scope, "random_same_norm", layer, patched(batch, layer, control), base, counterfactual)

        for layer in full_layers:
            record(episode, receiver, donor, "full", "swap", layer, patched(batch, layer, stores[donor][layer]), base, counterfactual)

file.close()
summary = {
    "status": "exploratory, not Gate evidence",
    "frames_in_this_invocation": [frame["episode_index"] for frame in frames],
    "skip_frames": args.skip_frames,
    "snapshot_source": cfg["snapshots"]["source"],
    "patch": "prompt pass only; generated tokens run unedited",
    "self_patch_during_generation": identity,
    "invalid_outputs_in_this_invocation": sum(not row["valid"] for row in rows),
    "n_rows_in_this_invocation": len(rows),
    "total_seconds": round(time.perf_counter() - started, 1),
}
(out / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
