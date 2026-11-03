#!/usr/bin/env bash
# Stage 1 baseline: 10 episodes per task with the model card's ensembling settings, in float32
# so the numeric mode matches the readout and patching runs. Episode starts differ by seed, and
# both tasks use the same seeds.
#   PYTHON=/path/to/env/bin/python bash scripts/run_stage1.sh
set -euo pipefail

bash scripts/run_eval.sh 8 10 1 cpu evidence/stage1/plate_10ep --policy.use_amp=false
#ash scripts/run_eval.sh 1 10 1 cpu evidence/stage1/stove_10ep --policy.use_amp=false
