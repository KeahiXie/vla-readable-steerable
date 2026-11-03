"""Exploratory: roll out from saved states with an edit repeated at every generation, and score the landing.

The instruction is the state's own task. At each generation the donor is recomputed from the current
observation with the other instruction, and the edit applies at one layer, at the positions after the
word, on the prompt pass only. Conditions:
  none               own instruction, no edit
  full_after_word    h + delta
  word_natural       h + P_W delta
  complement         h + (I - P_W) delta
  word_equal_size    the word part scaled up to the size of delta
  random_full_size   fresh noise at each generation, the size of delta
  other_instruction  the other instruction, no edit, as a reference for what the state allows
A rollout ends when the bowl is on either target, or at the chunk limit. Whole rollouts are written, and
finished ones are skipped, so a run can stop and continue. Tuning episodes only.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hooks import capture_output, replace_prompt_output
from load_policy import load_config, load_policy
from sim import BOWL, PLATE, STOVE, TASK_TEXT, Scene, env_action
from snapshots import prompt_inputs

WORD_POSITION = 144
CONDITIONS = ["none", "full_after_word", "word_natural", "complement", "word_equal_size", "random_full_size", "other_instruction"]

parser = argparse.ArgumentParser()
parser.add_argument("--snapshots", required=True)
parser.add_argument("--pattern", required=True)
parser.add_argument("--conditions", default=",".join(CONDITIONS))
parser.add_argument("--max-chunks", type=int, default=16)
parser.add_argument("--layer", type=int, default=16)
parser.add_argument("--out", required=True)
args = parser.parse_args()

cfg = load_config("config/smoke.yaml")
policy, pre, post, _ = load_policy(cfg["pins"], device="cpu")
model = policy.model
vlm = model.vlm
text_model = vlm.model.text_model
layer = text_model.layers[args.layer]
tokenizer = model.processor.tokenizer
word_ids = {w: tokenizer(" " + w, add_special_tokens=False).input_ids[0] for w in TASK_TEXT}
out = Path(args.out)
(out / "videos").mkdir(parents=True, exist_ok=True)

with torch.no_grad():
    directions = text_model.norm.weight * vlm.lm_head.weight[[word_ids["plate"], word_ids["stove"]]]
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
        actions = model.generate_actions(batch)
    numbers = raw_text[-1].strip().split()
    valid = len(numbers) == actions.numel() and all(n.isdigit() and 0 <= int(n) < 512 for n in numbers)
    return actions, valid


def layer_output(inputs):
    store = {}
    handle = capture_output(layer, store, 0)
    with torch.no_grad():
        vlm(**inputs, use_cache=False)
    handle.remove()
    return store[0]


def edited_chunk(condition, batches, receiver, donor, generator):
    """One generation under a condition. Returns the chunk, its validity and the size of the edit."""
    if condition == "none":
        return (*generate(batches[receiver]), {})
    if condition == "other_instruction":
        return (*generate(batches[donor]), {})

    inputs = {name: prompt_inputs(policy, batch) for name, batch in batches.items()}
    differing = (inputs["plate"]["input_ids"][0] != inputs["stove"]["input_ids"][0]).nonzero().flatten().tolist()
    assert differing == [WORD_POSITION], differing
    after_word = list(range(WORD_POSITION + 1, inputs[receiver]["input_ids"].shape[1]))
    h = layer_output(inputs[receiver])
    delta = torch.zeros_like(h)
    delta[:, after_word] = layer_output(inputs[donor])[:, after_word] - h[:, after_word]
    word_part = delta @ basis @ basis.T
    if condition == "full_after_word":
        edit = delta
    elif condition == "word_natural":
        edit = word_part
    elif condition == "complement":
        edit = delta - word_part
    elif condition == "word_equal_size":
        edit = word_part * delta.norm() / word_part.norm()
    else:
        noise = torch.zeros_like(h)
        noise[:, after_word] = torch.randn(h[:, after_word].shape, generator=generator)
        edit = noise / noise.norm() * delta.norm()
    handle = replace_prompt_output(layer, h + edit)
    actions, valid = generate(batches[receiver])
    handle.remove()
    return actions, valid, {"edit_norm": edit.norm().item(), "word_fraction": (word_part.norm() / delta.norm()).item()}


def horizontal(a, b):
    return round(float(np.linalg.norm((a - b)[:2])), 4)


path_rows = out / "rollouts.jsonl"
finished = set()
if path_rows.exists():
    finished = {(r["snapshot"], r["condition"]) for r in map(json.loads, open(path_rows))}
log = open(path_rows, "a")
paths = sorted(Path(args.snapshots).glob(args.pattern))
scene = Scene(paths[0].name.split("_")[0], policy.config)

for path in paths:
    receiver = path.name.split("_")[0]
    donor = "stove" if receiver == "plate" else "plate"
    state = np.load(path)
    for condition in args.conditions.split(","):
        if (path.name, condition) in finished:
            continue
        started = time.perf_counter()
        generator = torch.Generator().manual_seed(int(path.stem.split("seed")[1]) + cfg["seed"])
        observation = scene.restore(state)
        targets = {"plate": scene.position(PLATE), "stove": scene.position(STOVE)}
        start = scene.position(BOWL)
        frames = [observation["pixels"]["image"][0][::-1, ::-1].copy()]
        landed, steps, invalid, sizes, trace = [], 0, 0, [], []
        for chunk_index in range(args.max_chunks):
            batches = {name: scene.batch(observation, name, pre) for name in TASK_TEXT}
            chunk, valid, size = edited_chunk(condition, batches, receiver, donor, generator)
            invalid += not valid
            if size:
                sizes.append(size)
            for i in range(chunk.shape[1]):
                observation, _ = scene.step(env_action(chunk[:, i], post, scene))
                steps += 1
                frames.append(observation["pixels"]["image"][0][::-1, ::-1].copy())
                landed = scene.landed_on()
                if landed:
                    break
            trace.append(scene.position(BOWL).round(4).tolist())
            if landed:
                break
        end = scene.position(BOWL)
        record = {
            "snapshot": path.name,
            "receiver": receiver,
            "donor": donor,
            "condition": condition,
            "landed_on": landed,
            "landed_own": receiver in landed,
            "landed_other": donor in landed,
            "steps": steps,
            "generations": chunk_index + 1,
            "invalid_generations": invalid,
            "bowl_start": start.round(4).tolist(),
            "bowl_end": end.round(4).tolist(),
            "end_distance_plate": horizontal(end, targets["plate"]),
            "end_distance_stove": horizontal(end, targets["stove"]),
            "mean_edit_norm": round(float(np.mean([s["edit_norm"] for s in sizes])), 3) if sizes else None,
            "mean_word_fraction": round(float(np.mean([s["word_fraction"] for s in sizes])), 4) if sizes else None,
            "bowl_by_chunk": trace,
            "seconds": round(time.perf_counter() - started, 1),
        }
        imageio.mimsave(out / "videos" / f"{path.stem}_{condition}.mp4", frames, fps=20, macro_block_size=None)
        log.write(json.dumps(record) + "\n")
        log.flush()
        print({k: record[k] for k in ("snapshot", "condition", "landed_on", "steps", "generations", "invalid_generations", "bowl_end", "seconds")}, flush=True)
scene.close()
print("done")
