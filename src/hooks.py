"""Forward hooks: read or overwrite the output of one layer."""


def capture_output(layer, store, key):
    """Save this layer's output into store[key] whenever the model runs. Return the hook handle."""
    def hook(module, inputs, output):
        store[key] = output.detach().clone()
    handle = layer.register_forward_hook(hook)
    return handle


def replace_output(layer, replacement):
    """Make this layer output `replacement` instead of what it computed. Return the hook handle."""
    def hook(module, inputs, output):
        return replacement
    handle = layer.register_forward_hook(hook)
    return handle
