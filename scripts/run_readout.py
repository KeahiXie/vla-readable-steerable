"""Stage 0 run: logit-lens readout at every layer plus hook checks. No generation.

The research logic lives in src/hooks.py and src/lens.py. This file is the loop around them.
"""
import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hooks import capture_output, replace_output
from lens import logit_lens
from load_policy import load_config, load_policy
from snapshots import load_frames, make_batch, prompt_inputs

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()
cfg = load_config(args.config)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
started = time.perf_counter()

policy, pre, post, _ = load_policy(cfg["pins"], device=cfg["device"])
vlm = policy.model.vlm
text_model = vlm.model.text_model
layers = text_model.layers  # the language layers only; the vision encoder has its own
tok = policy.model.processor.tokenizer

word_ids = {}
for word in cfg["readout"]["words"]:
    ids = tok(" " + word, add_special_tokens=False).input_ids
    assert len(ids) == 1, f"' {word}' is not one token: {ids}"
    word_ids[word] = ids[0]

patch_layers = cfg["checks"]["self_patch_layers"]
patch_layers = range(len(layers)) if patch_layers == "all" else patch_layers
frames = load_frames(cfg["pins"], cfg["snapshots"])

rows, checks = [], []
for frame in frames:
    for instruction, text in cfg["instructions"].items():
        inputs = prompt_inputs(policy, make_batch(frame, text, pre))

        # Clean run, recording the output of every layer.
        store = {}
        handles = [capture_output(layer, store, i) for i, layer in enumerate(layers)]
        with torch.no_grad():
            clean = vlm(**inputs, use_cache=False).logits
        for handle in handles:
            handle.remove()

        # Readout at the last prompt token (what the model reads just before writing actions).
        with torch.no_grad():
            for i in range(len(layers)):
                logits = logit_lens(store[i][:, -1:, :], text_model.norm, vlm.lm_head)[0, 0]
                plate = logits[word_ids["plate"]].item()
                stove = logits[word_ids["stove"]].item()
                diff = stove - plate
                rows.append(
                    {
                        "episode_index": frame["episode_index"],
                        "frame_index": frame["frame_index"],
                        "instruction": instruction,
                        "layer": i,
                        "position": "prompt_end",
                        "logit_plate": plate,
                        "logit_stove": stove,
                        "stove_minus_plate": diff,
                        "sign_matches_instruction": (diff > 0) == (instruction == "stove"),
                    }
                )

            # The lens on the final layer must reproduce the model's real output.
            final = logit_lens(store[len(layers) - 1], text_model.norm, vlm.lm_head)
            lens_error = (final - clean).abs().max().item()

            # Self-patch: writing h back unchanged must leave the output exactly the same.
            patch_errors = {}
            for i in patch_layers:
                handle = replace_output(layers[i], store[i])
                patched = vlm(**inputs, use_cache=False).logits
                handle.remove()
                patch_errors[i] = (patched - clean).abs().max().item()

            # Without this, a hook that never fires would pass the self-patch check too.
            middle = len(layers) // 2
            handle = replace_output(layers[middle], torch.zeros_like(store[middle]))
            zeroed = vlm(**inputs, use_cache=False).logits
            handle.remove()
            zero_patch_change = (zeroed - clean).abs().max().item()

            # All hooks must be gone, or later runs would silently be patched.
            after = vlm(**inputs, use_cache=False).logits
            after_removal_error = (after - clean).abs().max().item()

        checks.append(
            {
                "episode_index": frame["episode_index"],
                "instruction": instruction,
                "final_layer_lens_vs_logits_max_abs": lens_error,
                "self_patch_max_abs_by_layer": patch_errors,
                "zero_patch_max_abs_change": zero_patch_change,
                "after_hook_removal_max_abs": after_removal_error,
            }
        )

with open(out / "readout.csv", "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# Per-layer summary. "paired" compares the two runs of the same frame, which removes any
# offset shared by both instructions; "sign" only asks whether the value is above zero.
with open(out / "readout_by_layer.csv", "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        ["layer", "n_frames", "sign_correct_of_2n", "paired_stove_above_plate", "median_paired_gap", "mean_plate_run", "mean_stove_run"]
    )
    for i in range(len(layers)):
        values = {
            name: {r["episode_index"]: r for r in rows if r["layer"] == i and r["instruction"] == name}
            for name in cfg["instructions"]
        }
        gaps = [
            values["stove"][e]["stove_minus_plate"] - values["plate"][e]["stove_minus_plate"] for e in values["plate"]
        ]
        writer.writerow(
            [
                i,
                len(gaps),
                sum(r["sign_matches_instruction"] for name in values for r in values[name].values()),
                sum(gap > 0 for gap in gaps),
                round(statistics.median(gaps), 4),
                round(statistics.mean(r["stove_minus_plate"] for r in values["plate"].values()), 4),
                round(statistics.mean(r["stove_minus_plate"] for r in values["stove"].values()), 4),
            ]
        )

worst_patch = max(max(c["self_patch_max_abs_by_layer"].values()) for c in checks)
worst_lens = max(c["final_layer_lens_vs_logits_max_abs"] for c in checks)
smallest_zero = min(c["zero_patch_max_abs_change"] for c in checks)
worst_after = max(c["after_hook_removal_max_abs"] for c in checks)
summary = {
    "config": args.config,
    "device": cfg["device"],
    "snapshot_source": cfg["snapshots"]["source"],
    "n_frames": len(frames),
    "n_layers": len(layers),
    "word_token_ids": word_ids,
    "n_token_positions": int(store[0].shape[1]),
    "hidden_size": int(store[0].shape[2]),
    "worst_self_patch_max_abs": worst_patch,
    "worst_final_lens_max_abs": worst_lens,
    "smallest_zero_patch_change": smallest_zero,
    "worst_after_hook_removal_max_abs": worst_after,
    "checks_passed": worst_patch == 0.0 and worst_after == 0.0 and worst_lens < 1e-3 and smallest_zero > 0.0,
    "total_seconds": round(time.perf_counter() - started, 1),
}
(out / "checks.json").write_text(json.dumps(checks, indent=2))
(out / "summary.json").write_text(json.dumps(summary, indent=2))

figure, axis = plt.subplots(figsize=(7, 4))
for instruction, colour in (("plate", "#1f77b4"), ("stove", "#d95f02")):
    for frame in frames:
        chosen = [r for r in rows if r["instruction"] == instruction and r["episode_index"] == frame["episode_index"]]
        axis.plot(
            [r["layer"] for r in chosen],
            [r["stove_minus_plate"] for r in chosen],
            color=colour,
            alpha=0.5,
            linewidth=1.2,
            label=instruction if frame is frames[0] else None,
        )
axis.axhline(0, color="#888888", linewidth=0.8)
axis.set_xlabel("layer (output of decoder layer)")
axis.set_ylabel('lens logit " stove" minus " plate"')
axis.set_title(f"{cfg['name']}: prompt-end readout, {len(frames)} recorded frame(s)")
axis.legend(title="instruction", frameon=False)
figure.tight_layout()
figure.savefig(out / "readout.png", dpi=150)

print(json.dumps(summary, indent=2))
sys.exit(0 if summary["checks_passed"] else 1)
