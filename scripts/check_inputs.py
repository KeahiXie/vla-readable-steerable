"""Plumbing check: do the two prompts align, and does the unedited model produce sensible actions?"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from load_policy import load_config, load_policy
from snapshots import load_frames, make_batch, prompt_inputs

cfg = load_config("config/smoke.yaml")
out_dir = Path("evidence/setup")
out_dir.mkdir(parents=True, exist_ok=True)
device = sys.argv[1] if len(sys.argv) > 1 else "cpu"

policy, pre, post, _ = load_policy(cfg["pins"], device=device)
tok = policy.model.processor.tokenizer
frame = load_frames(cfg["pins"], cfg["snapshots"])[0]
report = {"device": device, "episode_index": frame["episode_index"], "frame_index": frame["frame_index"]}

batches = {name: make_batch(frame, text, pre) for name, text in cfg["instructions"].items()}
inputs = {name: prompt_inputs(policy, batch) for name, batch in batches.items()}
ids = {name: value["input_ids"][0].cpu() for name, value in inputs.items()}

report["n_tokens"] = {name: len(value) for name, value in ids.items()}
differ = (ids["plate"] != ids["stove"]).nonzero().flatten().tolist()
report["differing_positions"] = differ
report["differing_tokens"] = {name: [tok.decode([value[i]]) for i in differ] for name, value in ids.items()}
report["pixels_equal"] = bool(torch.equal(inputs["plate"]["pixel_values"], inputs["stove"]["pixel_values"]))
report["pixel_values_shape"] = list(inputs["plate"]["pixel_values"].shape)
report["n_image_tokens"] = int((ids["plate"] == policy.model.vlm.config.image_token_id).sum())
report["prompt_tail"] = [tok.decode([i]) for i in ids["plate"][-45:]]

policy.model.vlm.requires_grad_(False)
for name, value in inputs.items():
    start = time.perf_counter()
    with torch.no_grad():
        out = policy.model.vlm(**value, use_cache=False)
    logits = out.logits[0, -1]
    top = logits.topk(5).indices.tolist()
    report[f"forward_{name}"] = {
        "seconds": round(time.perf_counter() - start, 3),
        "logits_shape": list(out.logits.shape),
        "top5_next_token": [tok.decode([i]) for i in top],
    }

for name, batch in batches.items():
    start = time.perf_counter()
    with torch.no_grad():
        actions = policy.model.generate_actions(batch)
    seconds = time.perf_counter() - start
    real = post(actions[:, 0])  # first action of the chunk, in real units
    report[f"generate_{name}"] = {
        "seconds": round(seconds, 1),
        "chunk_shape": list(actions.shape),
        "first_action_normalized": [round(x, 3) for x in actions[0, 0].tolist()],
        "first_action_real": [round(x, 3) for x in real[0].tolist()],
        "all_bin_zero_fallback": bool((actions < -0.99).all()),
    }
report["recorded_action"] = [round(x, 3) for x in frame["recorded_action"]]

path = out_dir / f"check_inputs_{device}.json"
path.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
