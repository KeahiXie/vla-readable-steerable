"""Measure where one action generation spends its time: model forward passes or everything else."""
import argparse
import cProfile
import io
import json
import pstats
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from load_policy import load_config, load_policy
from snapshots import load_frames, make_batch

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu")
parser.add_argument("--out", required=True)
args = parser.parse_args()

cfg = load_config("config/smoke.yaml")
policy, pre, post, _ = load_policy(cfg["pins"], device=args.device)
frame = load_frames(cfg["pins"], cfg["snapshots"])[0]
batch = make_batch(frame, cfg["instructions"]["plate"], pre)

# Time every text-model forward pass with hooks, so the rest of the generation is the remainder.
text_model = policy.model.vlm.model.text_model
vision_model = policy.model.vlm.model.vision_model
spans = {"text_calls": 0, "text_s": 0.0, "vision_calls": 0, "vision_s": 0.0}


def timed(module, label):
    def before(mod, inputs, kwargs):
        mod._started = time.perf_counter()

    def after(mod, inputs, kwargs, output):
        spans[f"{label}_calls"] += 1
        spans[f"{label}_s"] += time.perf_counter() - mod._started

    module.register_forward_pre_hook(before, with_kwargs=True)
    module.register_forward_hook(after, with_kwargs=True)


timed(text_model, "text")
timed(vision_model, "vision")

with torch.no_grad():
    policy.model.generate_actions(batch)  # warm-up
spans.update(text_calls=0, text_s=0.0, vision_calls=0, vision_s=0.0)

profiler = cProfile.Profile()
start = time.perf_counter()
profiler.enable()
with torch.no_grad():
    policy.model.generate_actions(batch)
profiler.disable()
total = time.perf_counter() - start

result = {
    "device": args.device,
    "total_s": round(total, 3),
    "text_forward_calls": spans["text_calls"],
    "text_forward_s": round(spans["text_s"], 3),
    "vision_forward_calls": spans["vision_calls"],
    "vision_forward_s": round(spans["vision_s"], 3),
    "everything_else_s": round(total - spans["text_s"] - spans["vision_s"], 3),
    "torch_threads": torch.get_num_threads(),
}
stream = io.StringIO()
pstats.Stats(profiler, stream=stream).sort_stats("tottime").print_stats(14)

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2))
out.with_suffix(".txt").write_text(stream.getvalue())
print(json.dumps(result, indent=2))
print(stream.getvalue()[:2600])
