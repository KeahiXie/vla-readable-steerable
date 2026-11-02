#!/usr/bin/env bash
# Regenerate the Stage 0 evidence bundle. Run from the repo root:
#   PYTHON=/path/to/env/bin/python bash scripts/make_evidence.sh
# Close heavy apps first: the benchmarks measure timing.
set -euo pipefail

PYTHON="${PYTHON:-python}"
mkdir -p evidence/environment evidence/logs evidence/smoke evidence/cpu evidence/benchmark evidence/setup

{
  date -u +"generated_utc: %Y-%m-%dT%H:%M:%SZ"
  echo "repo_commit: $(git rev-parse --verify -q HEAD || echo none-yet)"
  echo "repo_uncommitted_files: $(git status --porcelain | wc -l | tr -d ' ')"
  uname -srm
  "$PYTHON" --version
} > evidence/environment/system.txt
if command -v sysctl >/dev/null; then
  sysctl -n machdep.cpu.brand_string hw.memsize >> evidence/environment/system.txt
fi
"$PYTHON" -m pip freeze > evidence/environment/pip_freeze.txt
cp environment.yml evidence/environment/environment.yml

run() {
  local name="$1"
  shift
  echo "=== $name: $*"
  "$@" 2>&1 | sed "s#$HOME#~#g" | tee "evidence/logs/$name.log"
}

run inspect_model "$PYTHON" scripts/inspect_model.py config/smoke.yaml
run check_inputs "$PYTHON" scripts/check_inputs.py cpu
# LIBERO asks a question on first import; "N" accepts its default config.
[ -f "$HOME/.libero/config.yaml" ] || echo N | "$PYTHON" -c "import libero.libero" > /dev/null
run check_libero "$PYTHON" scripts/check_libero.py
run check_capture "$PYTHON" scripts/check_capture.py
run check_replace "$PYTHON" scripts/check_replace.py

cp config/smoke.yaml evidence/smoke/config.yaml
run smoke "$PYTHON" scripts/run_readout.py --config config/smoke.yaml --out evidence/smoke
cp config/cpu.yaml evidence/cpu/config.yaml
run cpu "$PYTHON" scripts/run_readout.py --config config/cpu.yaml --out evidence/cpu

run benchmark_cpu "$PYTHON" scripts/benchmark.py --device cpu --forwards 20 --generations 10 --out evidence/benchmark/cpu.json
if "$PYTHON" -c "import torch, sys; sys.exit(0 if torch.backends.mps.is_available() else 1)"; then
  # The image resize isn't implemented on MPS; PyTorch runs that one operation on the CPU.
  PYTORCH_ENABLE_MPS_FALLBACK=1 run benchmark_mps "$PYTHON" scripts/benchmark.py --device mps --forwards 20 --generations 10 --out evidence/benchmark/mps.json
fi

echo "Evidence bundle complete."
