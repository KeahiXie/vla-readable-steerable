# vla-readable-steerable

When a robot model's goal can be read from a middle layer with a lens, does editing that readable part move the robot? This repo tests that on VLA-0-Smol (0.5B) with LIBERO-Goal "put the bowl on the plate" and "put the bowl on the stove".

**Status: Stage 0. The readout and hook checks run on recorded frames (see `evidence/`). The simulator experiments haven't started, so there are no research results yet.**

## Layout

| Path | What it holds |
|---|---|
| `config/smoke.yaml` | 1 frame: prompt-only readout at every layer, plus the self-patch identity check |
| `config/cpu.yaml` | The same readout on 10 frames, CPU only |
| `config/prereg.yaml` | Gate thresholds, δ, layers and positions. Fixed before any held-out run |
| `src/` | Hooks, logit lens and patching |
| `scripts/` | Run, benchmark and evidence scripts |
| `evidence/` | One folder per run: config, pins, environment, raw outputs, logs, timing, one plot |

## Setup

```bash
mamba env create -f environment.yml
```

## Stage 0 data

Stage 0 uses recorded demo frames from `HuggingFaceVLA/libero`, not saved simulator states, because LIBERO isn't running yet. The frames aren't checked to be after the bowl is lifted, so Stage 0 readouts show that the code works and aren't evidence for Gate 1.
