import pytest
import torch

from lightllm.models.phi3.triton_kernel.rotary_emb import rotary_emb_fwd


def ref_rotary(x, cos, sin, head_dim):
    rot = head_dim // 2
    x = x.float()
    x1, x2 = x[..., :rot], x[..., rot:head_dim]
    c, s = cos.float()[:, None, :], sin.float()[:, None, :]
    out = x.clone()
    out[..., :rot] = x1 * c - x2 * s
    out[..., rot:head_dim] = x1 * s + x2 * c
    return out


@pytest.mark.parametrize(
    "shape",
    [
        # (total_len, q_heads, k_heads, head_dim); 96 is Phi-3-mini's head_dim,
        # whose rot_dim 48 is not a power of two, so BLOCK_DMODEL has padding
        # lanes that every load must mask off
        (33, 32, 32, 96),
        (128, 32, 8, 80),
        (64, 16, 16, 128),
    ],
)
@pytest.mark.parametrize("partial_rotary_factor", [1.0, 0.5])
def test_rotary_matches_reference(shape, partial_rotary_factor):
    torch.manual_seed(0)
    device = "cuda"
    T, HQ, HK, D = shape
    rot_dim = int(D * partial_rotary_factor) // 2

    q = torch.randn(T, HQ, D, device=device, dtype=torch.float16)
    k = torch.randn(T, HK, D, device=device, dtype=torch.float16)
    pos = torch.arange(T, device=device, dtype=torch.float32)
    inv = 1.0 / (10000 ** (torch.arange(0, rot_dim, device=device, dtype=torch.float32) / rot_dim))
    angles = pos[:, None] * inv[None, :]
    cos = angles.cos().to(torch.float16).contiguous()
    sin = angles.sin().to(torch.float16).contiguous()

    q_ref = ref_rotary(q, cos, sin, int(D * partial_rotary_factor))
    k_ref = ref_rotary(k, cos, sin, int(D * partial_rotary_factor))

    rotary_emb_fwd(q, k, cos, sin, partial_rotary_factor=partial_rotary_factor)

    assert torch.allclose(q.float(), q_ref, atol=5e-3, rtol=1e-2)
    assert torch.allclose(k.float(), k_ref, atol=5e-3, rtol=1e-2)
