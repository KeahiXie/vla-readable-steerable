"""Check capture_output in src/hooks.py and show what it captured."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hooks import capture_output
from load_policy import load_config, load_policy
from snapshots import load_frames, make_batch, prompt_inputs

cfg = load_config("config/smoke.yaml")
policy, pre, post, _ = load_policy(cfg["pins"], device="cpu")
vlm = policy.model.vlm
layers = vlm.model.text_model.layers
frame = load_frames(cfg["pins"], cfg["snapshots"])[0]


def run(instruction):
    inputs = prompt_inputs(policy, make_batch(frame, cfg["instructions"][instruction], pre))
    store = {}
    handles = [capture_output(layer, store, i) for i, layer in enumerate(layers)]
    with torch.no_grad():
        vlm(**inputs, use_cache=False)
    for handle in handles:
        handle.remove()
    return inputs, store


inputs, plate = run("plate")
_, stove = run("stove")

assert len(plate) == len(layers), f"captured {len(plate)} layers, expected {len(layers)}"
assert isinstance(plate[0], torch.Tensor), f"store holds {type(plate[0]).__name__}, expected a Tensor"
assert not torch.equal(plate[0], plate[1]), "layers 0 and 1 captured the same values"

count_before = len(plate)
plate[0] = None
with torch.no_grad():
    vlm(**inputs, use_cache=False)
assert plate[0] is None and len(plate) == count_before, "a hook is still attached after remove()"

print("PASS: one tensor per layer, and the hooks come off cleanly")
print("layers captured:", len(stove))
print("shape of one capture [batch, positions, hidden]:", list(stove[16].shape))
print()
print("Where do the two instructions differ inside the network?")
for i in (1, 16, 31):
    changed = (plate[i][0] != stove[i][0]).any(dim=-1).nonzero().flatten().tolist()
    print(f"  layer {i:2d}: first differing position {changed[0]}, positions differing {len(changed)} of {plate[i].shape[1]}")
