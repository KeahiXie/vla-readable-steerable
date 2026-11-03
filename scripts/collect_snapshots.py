"""Stage 2: run episodes with 8-step chunks and save the simulator state once the bowl is lifted.

A snapshot is taken at the first model-query boundary after the bowl has risen by LIFT_METERS, so no
queued actions carry over. Episodes are the independent unit. Every episode is logged, including
those that produce no snapshot.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from load_policy import load_config, load_policy
from sim import BOWL, PLATE, STOVE, Scene, env_action

LIFT_METERS = 0.05

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["plate", "stove"], required=True)
parser.add_argument("--seeds", required=True, help="first:count, for example 1000:10")
parser.add_argument("--max-steps", type=int, default=120)
parser.add_argument("--out", required=True)
args = parser.parse_args()

first, count = (int(x) for x in args.seeds.split(":"))
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

cfg = load_config("config/smoke.yaml")
policy, pre, post, _ = load_policy(cfg["pins"], device="cpu")
scene = Scene(args.task, policy.config)

log = open(out / f"{args.task}_episodes.jsonl", "a")
for seed in range(first, first + count):
    started = time.perf_counter()
    observation = scene.reset(seed)
    start_bowl = scene.position(BOWL)
    step, generations, success, snapshot = 0, 0, False, None
    while step < args.max_steps and not success:
        if scene.position(BOWL)[2] - start_bowl[2] >= LIFT_METERS:
            snapshot = scene.save_state()
            break
        batch = scene.batch(observation, args.task, pre)
        with torch.no_grad():
            chunk = policy.model.generate_actions(batch)
        generations += 1
        for i in range(chunk.shape[1]):
            observation, success = scene.step(env_action(chunk[:, i], post, scene))
            step += 1
            if success:
                break

    record = {
        "task": args.task,
        "seed": seed,
        "steps": step,
        "generations": generations,
        "snapshot": snapshot is not None,
        "success_before_snapshot": success,
        "bowl_start": start_bowl.round(4).tolist(),
        "bowl_end": scene.position(BOWL).round(4).tolist(),
        "plate": scene.position(PLATE).round(4).tolist(),
        "stove": scene.position(STOVE).round(4).tolist(),
        "seconds": round(time.perf_counter() - started, 1),
    }
    if snapshot is not None:
        np.save(out / f"{args.task}_seed{seed}.npy", snapshot)
    log.write(json.dumps(record) + "\n")
    log.flush()
    print(record, flush=True)
scene.close()
