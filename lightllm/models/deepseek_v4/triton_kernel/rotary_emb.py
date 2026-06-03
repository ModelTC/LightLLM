import torch

# Interleaved (GPT-J) rotary application for DeepSeek-V4. Unlike llama/gemma's NeoX-style
# rotary_emb_fwd (rotate-half: pairs channel i with i+d/2 over a real cos/sin table), V4 rotates
# adjacent pairs (x0,x1),(x2,x3),... — a different channel pairing — so it cannot reuse
# rotary_emb_fwd, but it consumes the same real cos/sin tables (built in model.py:_init_to_get_rotary
# as _cos_cached_*/_sin_cached_*, gemma4-style). Correctness-first pure-torch; a fused triton port is
# a perf follow-up.


def apply_rotary_emb(x, cos, sin, inverse=False):
    """Apply interleaved rope to the LAST dim of x (size = 2*cos.size(-1)).

    x: [..., rope_dim] (real). cos/sin: [..., rope_dim//2], broadcastable to x's paired view.
    For x of shape [N, H, rope_dim], pass cos/sin [N, 1, rope_dim//2]; for [N, rope_dim] pass [N, rope_dim//2].
    Returns a new tensor of x's dtype (not in-place). inverse=True applies the conjugate rotation.
    """
    dtype = x.dtype
    x = x.float().reshape(*x.shape[:-1], -1, 2)
    x0, x1 = x[..., 0], x[..., 1]
    cos = cos.float()
    sin = sin.float()
    if inverse:
        sin = -sin
    out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
    return out.flatten(-2).to(dtype)
