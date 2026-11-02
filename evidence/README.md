# Stage 0 evidence bundle

Regenerate everything here with `scripts/make_evidence.sh`, except `benchmark/gpu_t4*`, which comes from `kaggle/gpu_timing.ipynb`, and `stage1_trial/`, which comes from `scripts/run_eval.sh`.

| Path | What it holds |
|---|---|
| `environment/` | System details, resolved package versions and the environment file |
| `setup/` | Weight check, module tree, prompt alignment, one generation per instruction, and rendered LIBERO scenes |
| `smoke/` | 1-frame readout: config, `readout.csv`, `readout_by_layer.csv`, `checks.json`, `summary.json`, `readout.png` |
| `cpu/` | The same for 10 frames, CPU only |
| `benchmark/` | Forward time, generation time and peak memory per device |
| `logs/` | Full console output of each step |

The revisions used are pinned in each run's `config.yaml`. The installed LeRobot commit appears in `environment/pip_freeze.txt`.

**Data source:** these runs use recorded demo frames from `HuggingFaceVLA/libero`, taken 60% of the way through an episode, because LIBERO wasn't running yet. They are training frames, not saved simulator states, and they aren't checked to be after the bowl is lifted. They show that the code works. They aren't evidence for Gate 1.
