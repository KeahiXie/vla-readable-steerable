"""Compare generation with mixed precision off and on: same frames, both instructions."""
import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from load_policy import load_config, load_policy
from snapshots import load_frames, make_batch

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config/cpu.yaml")
parser.add_argument("--device", default="cpu")
parser.add_argument("--frames", type=int, default=3)
parser.add_argument("--out", required=True)
args = parser.parse_args()

cfg = load_config(args.config)
policy, pre, post, _ = load_policy(cfg["pins"], device=args.device)
frames = load_frames(cfg["pins"], cfg["snapshots"])[: args.frames]


def generate(batch, amp):
    context = torch.autocast(device_type=args.device, dtype=torch.bfloat16) if amp else nullcontext()
    start = time.perf_counter()
    with torch.no_grad(), context:
        actions = policy.model.generate_actions(batch)
    return actions[0].cpu(), time.perf_counter() - start


rows = []
for frame in frames:
    for instruction, text in cfg["instructions"].items():
        batch = make_batch(frame, text, pre)
        off, off_seconds = generate(batch, amp=False)
        on, on_seconds = generate(batch, amp=True)
        rows.append(
            {
                "episode_index": frame["episode_index"],
                "instruction": instruction,
                "off_seconds": round(off_seconds, 2),
                "on_seconds": round(on_seconds, 2),
                "identical_chunk": bool(torch.equal(off, on)),
                "max_abs_diff_normalized": round((off - on).abs().max().item(), 4),
                "mean_abs_diff_normalized": round((off - on).abs().mean().item(), 4),
                "first_action_off": [round(x, 3) for x in off[0].tolist()],
                "first_action_on": [round(x, 3) for x in on[0].tolist()],
            }
        )
        print(json.dumps(rows[-1]))

# How far apart the two instructions are, to judge whether the off/on difference is large.
gaps = {}
for frame in frames:
    pair = {r["instruction"]: r for r in rows if r["episode_index"] == frame["episode_index"]}
    gaps[frame["episode_index"]] = {
        mode: round(
            max(abs(a - b) for a, b in zip(pair["plate"][f"first_action_{mode}"], pair["stove"][f"first_action_{mode}"])),
            3,
        )
        for mode in ("off", "on")
    }
result = {"device": args.device, "amp_dtype": "bf16", "rows": rows, "max_first_action_gap_plate_vs_stove": gaps}
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2))
print(json.dumps(gaps))
