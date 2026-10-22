import functools

import torch


@functools.lru_cache(maxsize=None)
def pdl_kwargs(device):
    """Keep the original Hopper dependent-launch protocol in Triton kernels."""
    enabled = torch.version.hip is None and torch.cuda.get_device_capability(device)[0] >= 9
    return {"USE_PDL": True, "launch_pdl": True} if enabled else {"USE_PDL": False}
