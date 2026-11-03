"""Stage 2 eligibility: from each saved state, does each instruction head to its own destination?

Rolls out both instructions from the same state for a fixed number of 8-step chunks and records how
far the bowl moved toward each destination. Every state is logged, including ones that fail.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from load_policy import load_config, load_policy
from sim import BOWL, PLATE, STOVE, Scene, env_action

parser = argparse.ArgumentParser()
parser.add_argument("--snapshots", required=True)
parser.add_argument("--pattern", default="*_seed*.npy")
parser.add_argument("--chunks", type=int, default=4)
parser.add_argument("--out", required=True)
args = parser.parse_args()

cfg = load_config("config/smoke.yaml")
policy, pre, post, _ = load_policy(cfg["pins"], device="cpu")
scenes = {name: Scene(name, policy.config) for name in ("plate", "stove")}
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
log = open(out, "a")


def horizontal_distance(a, b):
    return float(np.linalg.norm((a - b)[:2]))


for path in sorted(Path(args.snapshots).glob(args.pattern)):
    state = np.load(path)
    result = {"snapshot": path.name}
    for instruction, scene in scenes.items():
        observation = scene.restore(state)
        destinations = {"plate": scene.position(PLATE), "stove": scene.position(STOVE)}
        start = scene.position(BOWL)
        success = False
        for _ in range(args.chunks):
            batch = scene.batch(observation, instruction, pre)
            with torch.no_grad():
                chunk = policy.model.generate_actions(batch)
            for i in range(chunk.shape[1]):
                observation, success = scene.step(env_action(chunk[:, i], post, scene))
                if success:
                    break
            if success:
                break
        end = scene.position(BOWL)
        result[instruction] = {
            "success": success,
            "bowl_height": round(float(end[2]), 4),
            "moved_toward_plate": round(horizontal_distance(start, destinations["plate"]) - horizontal_distance(end, destinations["plate"]), 4),
            "moved_toward_stove": round(horizontal_distance(start, destinations["stove"]) - horizontal_distance(end, destinations["stove"]), 4),
        }
    result["eligible"] = (
        result["plate"]["moved_toward_plate"] > result["stove"]["moved_toward_plate"]
        and result["stove"]["moved_toward_stove"] > result["plate"]["moved_toward_stove"]
    )
    log.write(json.dumps(result) + "\n")
    log.flush()
    print(result, flush=True)
