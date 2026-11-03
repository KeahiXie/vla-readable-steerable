"""Build kaggle/stage1.ipynb: LIBERO-Goal plate and stove baselines, with the needed code embedded."""
import json
from pathlib import Path

FILES = ["scripts/eval_libero.py", "config/smoke.yaml"]
COMMIT = "8e71efbebbb7ea2942b579b1d21e3ec11ac2d271"


def cell(kind, text):
    body = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        body.update(outputs=[], execution_count=None)
    return body


embedded = {name: Path(name).read_text() for name in FILES}
write_files = (
    "import json, pathlib\n"
    f"files = json.loads({json.dumps(json.dumps(embedded))})\n"
    "for name, text in files.items():\n"
    "    path = pathlib.Path(name)\n"
    "    path.parent.mkdir(parents=True, exist_ok=True)\n"
    "    path.write_text(text)\n"
    "print(sorted(files))\n"
)
environment_test = '''import os, subprocess, sys
test = """
import sys
sys.path.insert(0, "scripts")
import eval_libero
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env
for task_id in (8, 1):
    env = make_env(LiberoEnv(task="libero_goal", task_ids=[task_id]), n_envs=1)["libero_goal"][task_id]
    observation, _ = env.reset(seed=1000)
    print("TEST task", task_id, env.call("task_description"), observation["pixels"]["image"].shape)
    env.close()
print("TEST PASSED")
"""
subprocess.run("echo N | python -c 'import libero.libero'", shell=True)
result = subprocess.run([sys.executable, "-c", test], capture_output=True, text=True)
print(result.stdout[-1500:])
print(result.stderr[-3000:])
assert "TEST PASSED" in result.stdout, "LIBERO did not start on this machine"
'''
run_stage1 = '''import os, shutil, subprocess, sys, time
import torch
import yaml
from huggingface_hub import snapshot_download

pins = yaml.safe_load(open("config/smoke.yaml"))["pins"]
model_dir = snapshot_download(pins["model"], revision=pins["model_revision"])
base_dir = snapshot_download(
    pins["base_vlm"],
    revision=pins["base_vlm_revision"],
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
)
n_gpus = torch.cuda.device_count()
print("GPUs:", n_gpus, [torch.cuda.get_device_name(i) for i in range(n_gpus)])
TASKS = {"plate": 8, "stove": 1}
EPISODES = 10


def run_phase(phase, n_action_steps, ensemble_size):
    started = time.time()
    jobs = []
    for index, (name, task_id) in enumerate(TASKS.items()):
        output = f"stage1/{phase}/{name}_{EPISODES}ep"
        os.makedirs(os.path.dirname(output), exist_ok=True)
        command = [
            sys.executable, "scripts/eval_libero.py",
            f"--policy.path={model_dir}",
            f"--policy.vlm_checkpoint={base_dir}",
            "--policy.device=cuda",
            "--policy.use_amp=false",
            f"--policy.n_action_steps={n_action_steps}",
            f"--policy.ensemble_size={ensemble_size}",
            "--env.type=libero",
            "--env.task=libero_goal",
            f"--env.task_ids=[{task_id}]",
            "--eval.batch_size=1",
            f"--eval.n_episodes={EPISODES}",
            f"--output_dir={output}",
        ]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(index % max(n_gpus, 1)))
        log = open(f"{output}.log", "w")
        jobs.append((name, output, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)))
    for name, output, job in jobs:
        code = job.wait()
        print(phase, name, "exit", code, "elapsed so far", round((time.time() - started) / 60, 1), "min")
        if code != 0:
            print(open(f"{output}.log", errors="ignore").read()[-3000:])
        else:
            import json
            overall = json.load(open(f"{output}/eval_info.json"))["overall"]
            print(phase, name, {k: overall[k] for k in ("pc_success", "n_episodes", "eval_s", "eval_ep_s")})
    shutil.make_archive("/kaggle/working/stage1_results", "zip", "stage1")
    print("saved stage1_results.zip after", phase)


# Cheaper executor first, so a usable result exists early; then the model card's ensembling.
run_phase("no_ensembling", n_action_steps=8, ensemble_size=0)
run_phase("ensembling", n_action_steps=0, ensemble_size=8)
'''
cells = [
    cell(
        "markdown",
        "# VLA-0-Smol LIBERO-Goal baseline\n\n"
        "10 episodes each of plate and stove, float32: first with no ensembling, then with the model card's ensembling.\n\n"
        "Settings: **Accelerator = GPU T4 x2** and **Internet = on**.\n",
    ),
    cell("code", "!nvidia-smi -L\n"),
    cell(
        "code",
        '!pip install -q "cmake<4"\n'
        f'!pip install -q "lerobot[vla0_smol,libero] @ git+https://github.com/Robot-Learning-Collective/lerobot-experiments.git@{COMMIT}" pyyaml\n',
    ),
    cell("code", write_files),
    cell("code", environment_test),
    cell("code", run_stage1),
    cell("code", "!pip freeze > /kaggle/working/stage1_pip_freeze.txt\n!ls -la /kaggle/working\n"),
]
notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = Path("kaggle/stage1.ipynb")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(notebook, indent=1))
print("wrote", out, out.stat().st_size, "bytes")
