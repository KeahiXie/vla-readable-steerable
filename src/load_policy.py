"""Load VLA-0-Smol with every Hugging Face revision pinned. Plumbing, no research logic."""
from pathlib import Path

import torch
import yaml
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.vla0_smol.modeling_vla0_smol import VLA0SmolPolicy


def load_config(path):
    return yaml.safe_load(Path(path).read_text())


def fetch_pinned(pins):
    """Download the exact revisions and return their local folders."""
    model_dir = snapshot_download(pins["model"], revision=pins["model_revision"])
    # VLA0.__init__ loads vlm_checkpoint with no revision, so hand it a pinned local folder.
    base_dir = snapshot_download(
        pins["base_vlm"],
        revision=pins["base_vlm_revision"],
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
    )
    return model_dir, base_dir


def check_weights_loaded(policy, model_dir):
    """from_pretrained uses strict=False, which would skip a mismatched key without an error.

    Compare every checkpoint tensor with the live model. Returns a dict of counts.
    """
    ckpt = load_file(str(Path(model_dir) / "model.safetensors"))
    live = policy.state_dict()
    missing_in_model = [k for k in ckpt if k not in live]
    not_in_ckpt = [k for k in live if k not in ckpt]
    mismatched = [
        k
        for k in ckpt
        if k in live and not torch.equal(ckpt[k].to("cpu"), live[k].detach().to("cpu"))
    ]
    return {
        "checkpoint_tensors": len(ckpt),
        "model_tensors": len(live),
        "in_checkpoint_not_model": missing_in_model,
        "in_model_not_checkpoint": not_in_ckpt,
        "values_differ": mismatched,
    }


def load_policy(pins, device="cpu"):
    """Returns (policy, preprocessor, postprocessor, model_dir)."""
    model_dir, base_dir = fetch_pinned(pins)

    cfg = PreTrainedConfig.from_pretrained(model_dir)
    cfg.device = device  # the saved config says "cuda"
    cfg.use_amp = False
    cfg.vlm_checkpoint = base_dir

    policy = VLA0SmolPolicy.from_pretrained(model_dir, config=cfg)
    policy.eval()

    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=model_dir,
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    return policy, pre, post, model_dir
