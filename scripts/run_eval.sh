#!/usr/bin/env bash
# Run lerobot-eval on LIBERO-Goal with the model card's executor settings and pinned revisions.
#   PYTHON=/path/to/env/bin/python bash scripts/run_eval.sh <task_id> <n_episodes> <batch_size> <device> <output_dir>
# LIBERO-Goal task IDs: 8 = put the bowl on the plate, 1 = put the bowl on the stove.
set -euo pipefail

PYTHON="${PYTHON:-python}"
TASK_ID="$1"
N_EPISODES="$2"
BATCH_SIZE="$3"
DEVICE="$4"
OUTPUT_DIR="$5"

# Local snapshot folders pin both models, since the model code loads its base with no revision.
read -r MODEL_DIR BASE_DIR < <("$PYTHON" - <<'EOF'
import yaml
from huggingface_hub import snapshot_download

pins = yaml.safe_load(open("config/smoke.yaml"))["pins"]
model = snapshot_download(pins["model"], revision=pins["model_revision"])
base = snapshot_download(
    pins["base_vlm"],
    revision=pins["base_vlm_revision"],
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
)
print(model, base)
EOF
)

mkdir -p "$(dirname "$OUTPUT_DIR")"
"$PYTHON" scripts/eval_libero.py \
  --policy.path="$MODEL_DIR" \
  --policy.vlm_checkpoint="$BASE_DIR" \
  --policy.device="$DEVICE" \
  --policy.n_action_steps=0 \
  --policy.ensemble_size=8 \
  --env.type=libero \
  --env.task=libero_goal \
  --env.task_ids="[$TASK_ID]" \
  --eval.batch_size="$BATCH_SIZE" \
  --eval.n_episodes="$N_EPISODES" \
  --output_dir="$OUTPUT_DIR" \
  2>&1 | sed "s#$HOME#~#g" | tee "$OUTPUT_DIR.log"
