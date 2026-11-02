"""Check replace_output in src/hooks.py with the self-patch and zero-patch tests at layer 16."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hooks import capture_output, replace_output
from load_policy import load_config, load_policy
from snapshots import load_frames, make_batch, prompt_inputs

LAYER = 16

cfg = load_config("config/smoke.yaml")
policy, pre, post, _ = load_policy(cfg["pins"], device="cpu")
vlm = policy.model.vlm
layer = vlm.model.text_model.layers[LAYER]
frame = load_frames(cfg["pins"], cfg["snapshots"])[0]
inputs = prompt_inputs(policy, make_batch(frame, cfg["instructions"]["plate"], pre))


def run():
    with torch.no_grad():
        return vlm(**inputs, use_cache=False).logits


store = {}
handle = capture_output(layer, store, LAYER)
clean = run()
handle.remove()
h = store[LAYER]


def patched_run(replacement):
    patch = replace_output(layer, replacement)
    assert hasattr(patch, "remove"), "replace_output must return the hook handle"
    logits = run()
    patch.remove()
    return logits


self_change = (patched_run(h) - clean).abs().max().item()
zero_change = (patched_run(torch.zeros_like(h)) - clean).abs().max().item()
after_removal = (run() - clean).abs().max().item()

print(f"self-patch: largest output change {self_change}")
print(f"zero-patch: largest output change {zero_change}")
print(f"after remove(): largest output change {after_removal}")

assert self_change == 0.0, "writing h back unchanged altered the output"
assert zero_change > 0.0, "writing zeros changed nothing, so the replacement never reached the model"
assert after_removal == 0.0, "the output is still altered after remove()"
print("PASS: self-patch is exact, the patch really fires, and it comes off cleanly")
