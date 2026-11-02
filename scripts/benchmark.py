"""Measure forward-pass time, generation time and peak memory on one device. Run once per device."""
import argparse
import json
import os
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from load_policy import load_config, load_policy
from snapshots import load_frames, make_batch, prompt_inputs

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config/smoke.yaml")
parser.add_argument("--device", default="cpu")
parser.add_argument("--forwards", type=int, default=20)
parser.add_argument("--generations", type=int, default=10)
parser.add_argument("--out", required=True)
args = parser.parse_args()
cfg = load_config(args.config)


def sync():
    if args.device == "cuda":
        torch.cuda.synchronize()
    elif args.device == "mps":
        torch.mps.synchronize()


def peak_rss_gb():
    # ru_maxrss is bytes on macOS and kilobytes on Linux.
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e9 if sys.platform == "darwin" else raw * 1024 / 1e9


def summary(values):
    return {
        "n": len(values),
        "median_s": round(statistics.median(values), 4),
        "mean_s": round(statistics.mean(values), 4),
        "min_s": round(min(values), 4),
        "max_s": round(max(values), 4),
        "raw_s": [round(v, 4) for v in values],
    }


start = time.perf_counter()
policy, pre, post, _ = load_policy(cfg["pins"], device=args.device)
load_seconds = time.perf_counter() - start

frame = load_frames(cfg["pins"], cfg["snapshots"])[0]
batches = [make_batch(frame, text, pre) for text in cfg["instructions"].values()]
inputs = [prompt_inputs(policy, batch) for batch in batches]

mps_peak = 0


def sample_device_memory():
    global mps_peak
    if args.device == "mps":
        mps_peak = max(mps_peak, torch.mps.driver_allocated_memory())


# The first pass of each kind is timed separately because it includes warm-up.
forward_times = []
for i in range(args.forwards + 1):
    start = time.perf_counter()
    with torch.no_grad():
        policy.model.vlm(**inputs[i % 2], use_cache=False)
    sync()
    forward_times.append(time.perf_counter() - start)
    sample_device_memory()

generate_times, fallbacks = [], 0
for i in range(args.generations + 1):
    start = time.perf_counter()
    with torch.no_grad():
        actions = policy.model.generate_actions(batches[i % 2])
    sync()
    generate_times.append(time.perf_counter() - start)
    sample_device_memory()
    fallbacks += int((actions < -0.99).all())

result = {
    "device": args.device,
    "machine": platform.platform(),
    "processor": platform.processor(),
    "cuda_device": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
    "mps_cpu_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),  # image resize only
    "torch": torch.__version__,
    "episode_index": frame["episode_index"],
    "frame_index": frame["frame_index"],
    "load_seconds": round(load_seconds, 2),
    "forward_warmup_s": round(forward_times[0], 4),
    "forward": summary(forward_times[1:]),
    "generate_warmup_s": round(generate_times[0], 4),
    "generate": summary(generate_times[1:]),
    "bin_zero_fallbacks": fallbacks,
    "peak_process_rss_gb": round(peak_rss_gb(), 2),
    "peak_mps_driver_allocated_gb": round(mps_peak / 1e9, 2) if args.device == "mps" else None,
    "peak_cuda_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if args.device == "cuda" else None,
}
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k not in ("forward", "generate")}, indent=2))
print("forward", {k: v for k, v in result["forward"].items() if k != "raw_s"})
print("generate", {k: v for k, v in result["generate"].items() if k != "raw_s"})
