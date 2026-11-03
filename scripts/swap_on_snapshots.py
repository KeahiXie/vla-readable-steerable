"""Exploratory, on saved simulator states: lens readout at every layer, and the word-only edit.

Each state supplies one image and robot state, used with both instructions. At one layer and the
positions after the word, delta = h_other - h. W is the span of norm.weight * unembedding row for
" plate" and " stove". Each edit applies on the prompt pass only:
  full_after_word   h + delta
  word_natural      h + P_W delta
  complement        h + (I - P_W) delta
  word_equal_size   the word part scaled up to the size of delta
  random_word_size, random_full_size   controls matched to those two sizes
Rows are written as each state finishes, and states already written are skipped, so a run can stop
and continue. Tuning episodes only until the pre-registered values are fixed.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hooks import capture_output, replace_output, replace_prompt_output
from lens import logit_lens
from load_policy import load_config, load_policy
from sim import TASK_TEXT, Scene
from snapshots import prompt_inputs

WORD_POSITION = 144
READ_LAYERS = [17, 19, 22, 31]
CONDITIONS = ["full_after_word", "word_natural", "complement", "word_equal_size", "random_word_size", "random_full_size"]
ROW_FIELDS = ["snapshot", "receiver", "donor", "condition", "layer", "rho", "rho_first_action", "l2_from_base", "base_gap_l2", "edit_norm", "word_fraction_of_norm", "valid", "raw"]

parser = argparse.ArgumentParser()
parser.add_argument("--snapshots", required=True)
parser.add_argument("--pattern", default="*_seed*.npy")
parser.add_argument("--limit", type=int, default=0, help="stop after this many states in this run; 0 means all")
parser.add_argument("--layer", type=int, default=16)
parser.add_argument("--out", required=True)
args = parser.parse_args()

cfg = load_config("config/smoke.yaml")
policy, pre, post, _ = load_policy(cfg["pins"], device="cpu")
model = policy.model
vlm = model.vlm
text_model = vlm.model.text_model
layers = text_model.layers
tokenizer = model.processor.tokenizer
word_ids = {w: tokenizer(" " + w, add_special_tokens=False).input_ids[0] for w in TASK_TEXT}
scene = Scene("plate", policy.config)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

# The lens computes w . (norm.weight * h) / rms(h), so its direction in h is norm.weight * w.
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
        actions = model.generate_actions(batch)[0].cpu()
    numbers = raw_text[-1].strip().split()
    valid = len(numbers) == actions.numel() and all(n.isdigit() and 0 <= int(n) < 512 for n in numbers)
    return actions, valid, raw_text[-1].strip()


def rho(edit, base, counterfactual):
    direction = (counterfactual - base).flatten()
    return (((edit - base).flatten() @ direction) / (direction @ direction)).item()


def read(h, positions):
    with torch.no_grad():
        logits = logit_lens(h[:, positions, :], text_model.norm, vlm.lm_head)[0]
    return (logits[:, word_ids["stove"]] - logits[:, word_ids["plate"]]).mean().item()


def capture_all(inputs):
    store = {}
    handles = [capture_output(layer, store, i) for i, layer in enumerate(layers)]
    with torch.no_grad():
        vlm(**inputs, use_cache=False)
    for handle in handles:
        handle.remove()
    return store


def capture_edited(inputs, edited):
    store = {}
    handles = [capture_output(layers[i], store, i) for i in READ_LAYERS]
    handles.append(replace_output(layers[args.layer], edited))
    with torch.no_grad():
        vlm(**inputs, use_cache=False)
    for handle in handles:
        handle.remove()
    store[args.layer] = edited
    return store


def opened(name, header):
    path = out / name
    new = not path.exists()
    handle = open(path, "a", newline="")
    if new:
        csv.writer(handle).writerow(header)
    return handle


rows_path = out / "rows.csv"
finished = set()
if rows_path.exists():
    counts = {}
    for row in csv.DictReader(open(rows_path)):
        counts[row["snapshot"]] = counts.get(row["snapshot"], 0) + 1
    finished = {name for name, count in counts.items() if count == 2 * len(CONDITIONS)}

rows_file = opened("rows.csv", ROW_FIELDS)
rows_writer = csv.DictWriter(rows_file, ROW_FIELDS)
readout_file = opened("readout.csv", ["snapshot", "instruction", "layer", "scope", "stove_minus_plate"])
readout_writer = csv.writer(readout_file)
lens_file = opened("lens_reads.csv", ["snapshot", "receiver", "condition", "read_layer", "scope", "read_edit", "read_base", "read_donor", "move_toward_donor"])
lens_writer = csv.writer(lens_file)
log = opened("states.jsonl", [])

paths = [path for path in sorted(Path(args.snapshots).glob(args.pattern)) if path.name not in finished]
if args.limit:
    paths = paths[: args.limit]
print("states to run:", [path.name for path in paths], flush=True)

for path in paths:
    started = time.perf_counter()
    generator = torch.Generator().manual_seed(int(path.stem.split("seed")[1]) + cfg["seed"])
    observation = scene.restore(np.load(path))
    batches = {name: scene.batch(observation, name, pre) for name in TASK_TEXT}
    inputs = {name: prompt_inputs(policy, batch) for name, batch in batches.items()}

    differing = (inputs["plate"]["input_ids"][0] != inputs["stove"]["input_ids"][0]).nonzero().flatten().tolist()
    assert differing == [WORD_POSITION], differing
    length = inputs["plate"]["input_ids"].shape[1]
    after_word = list(range(WORD_POSITION + 1, length))

    stores = {name: capture_all(value) for name, value in inputs.items()}
    for name, store in stores.items():
        for i in range(len(layers)):
            for scope, positions in (("word", [WORD_POSITION]), ("after_word", after_word), ("last", [length - 1])):
                readout_writer.writerow([path.name, name, i, scope, round(read(store[i], positions), 4)])
    readout_file.flush()

    results = {name: generate(batch) for name, batch in batches.items()}
    bases = {name: result[0] for name, result in results.items()}
    state_row = rows = None
    identical, rows_out, lens_out = {}, [], []

    for receiver, donor in (("plate", "stove"), ("stove", "plate")):
        batch, base, counterfactual = batches[receiver], bases[receiver], bases[donor]
        h = stores[receiver][args.layer]

        handle = replace_prompt_output(layers[args.layer], h)
        identical[receiver] = bool(torch.equal(generate(batch)[0], base))
        handle.remove()
        assert identical[receiver], "self-patch changed the output"

        delta = torch.zeros_like(h)
        delta[:, after_word] = stores[donor][args.layer][:, after_word] - h[:, after_word]
        word_part = delta @ basis @ basis.T
        complement = delta - word_part
        fraction = (word_part.norm() / delta.norm()).item()
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
            rows_out.append(
                {
                    "snapshot": path.name,
                    "receiver": receiver,
                    "donor": donor,
                    "condition": condition,
                    "layer": args.layer,
                    "rho": round(rho(actions, base, counterfactual), 4),
                    "rho_first_action": round(rho(actions[0], base[0], counterfactual[0]), 4),
                    "l2_from_base": round((actions - base).norm().item(), 4),
                    "base_gap_l2": round((counterfactual - base).norm().item(), 4),
                    "edit_norm": round(edits[condition].norm().item(), 3),
                    "word_fraction_of_norm": round(fraction, 4),
                    "valid": valid,
                    "raw": raw,
                }
            )
            edited_store = capture_edited(inputs[receiver], edited)
            for read_layer in [args.layer, *READ_LAYERS]:
                for scope, positions in (("after_word", after_word), ("last", [length - 1])):
                    if read_layer == args.layer and scope == "last":
                        continue
                    value = read(edited_store[read_layer], positions)
                    start = read(stores[receiver][read_layer], positions)
                    goal = read(stores[donor][read_layer], positions)
                    move = (value - start) / (goal - start) if abs(goal - start) > 1e-6 else float("nan")
                    lens_out.append([path.name, receiver, condition, read_layer, scope, round(value, 4), round(start, 4), round(goal, 4), round(move, 4)])

    # Whole states only, so a pause or stop can't leave a half-written state counted as finished.
    rows_writer.writerows(rows_out)
    rows_file.flush()
    lens_writer.writerows(lens_out)
    lens_file.flush()
    gap = rows_out[0]["base_gap_l2"]
    record = {
        "snapshot": path.name,
        "prompt_length": length,
        "base_gap_l2": gap,
        "baselines_valid": {name: result[1] for name, result in results.items()},
        "self_patch_identical": identical,
        "seconds": round(time.perf_counter() - started, 1),
    }
    log.write(json.dumps(record) + "\n")
    log.flush()
    print(record, {r["receiver"][0] + ":" + r["condition"]: r["rho"] for r in rows_out}, flush=True)
print("done")
