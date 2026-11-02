"""Check LIBERO runs here: list the LIBERO-Goal tasks, build both scenes, render and step them."""
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.envs.libero import LiberoEnv, _get_suite, get_libero_dummy_action

out_dir = Path("evidence/setup")
out_dir.mkdir(parents=True, exist_ok=True)

suite = _get_suite("libero_goal")
tasks = {i: suite.tasks[i].language for i in range(len(suite.tasks))}
print(json.dumps(tasks, indent=2))
wanted = {
    "plate": next(i for i, text in tasks.items() if text == "put the bowl on the plate"),
    "stove": next(i for i, text in tasks.items() if text == "put the bowl on the stove"),
}

report = {"task_ids": wanted, "tasks": tasks}
for name, task_id in wanted.items():
    start = time.perf_counter()
    env = LiberoEnv(task_suite=suite, task_id=task_id, task_suite_name="libero_goal", init_states=True)
    observation, _ = env.reset(seed=0)
    build_seconds = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(20):
        observation, reward, terminated, truncated, info = env.step(np.array(get_libero_dummy_action()))
    step_seconds = (time.perf_counter() - start) / 20

    images = observation["pixels"]
    for camera, image in images.items():
        Image.fromarray(image).save(out_dir / f"libero_{name}_{camera}.png")
    report[name] = {
        "task_id": task_id,
        "language": env.task_description,
        "build_and_reset_seconds": round(build_seconds, 2),
        "seconds_per_sim_step": round(step_seconds, 4),
        "image_shapes": {camera: list(image.shape) for camera, image in images.items()},
        "max_episode_steps": env._max_episode_steps,
        "success_after_noop": bool(info["is_success"]),
    }
    env.close()

(out_dir / "libero_check.json").write_text(json.dumps(report, indent=2))
print(json.dumps({k: report[k] for k in ("plate", "stove")}, indent=2))
