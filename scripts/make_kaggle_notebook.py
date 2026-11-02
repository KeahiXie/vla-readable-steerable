"""Build kaggle/gpu_timing.ipynb with the benchmark code embedded, so Kaggle needs no repo access."""
import json
from pathlib import Path

FILES = ["src/load_policy.py", "src/snapshots.py", "scripts/benchmark.py", "config/smoke.yaml"]
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
cells = [
    cell(
        "markdown",
        "# VLA-0-Smol GPU timing\n\n"
        "Settings: **Accelerator = GPU T4** and **Internet = on**. Then choose Run All.\n\n"
        "When it finishes, download `gpu_t4.json` from the Output panel into `evidence/benchmark/`.\n",
    ),
    cell("code", "!nvidia-smi\n"),
    cell(
        "code",
        f'!pip install -q "lerobot[vla0_smol] @ git+https://github.com/Robot-Learning-Collective/lerobot-experiments.git@{COMMIT}" pyyaml\n',
    ),
    cell("code", write_files),
    cell(
        "code",
        "!python scripts/benchmark.py --device cuda --forwards 20 --generations 50 --out /kaggle/working/gpu_t4.json\n",
    ),
    cell("code", "!pip freeze > /kaggle/working/gpu_t4_pip_freeze.txt\n!cat /kaggle/working/gpu_t4.json\n"),
]
notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = Path("kaggle/gpu_timing.ipynb")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(notebook, indent=1))
print("wrote", out, out.stat().st_size, "bytes")
