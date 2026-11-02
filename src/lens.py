"""The logit lens: read a middle layer as if it were the last one."""


def logit_lens(h, norm, lm_head):
    """Turn hidden states h of shape [batch, positions, hidden] into word logits."""
    return lm_head(norm(h))
