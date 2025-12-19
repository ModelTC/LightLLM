import torch
import pytest

# Import the Triton kernel function under test. Adjust the import path as needed.
from lightllm.models.qwen2_vl.triton_kernel.mrope import mrope_triton_fused

# Reference Python implementation for multimodal rotary positional embeddings


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


@pytest.mark.parametrize(
    "B,H_q,H_k,L,D,mrope_section",
    [
        (1, 4, 2, 32, 16, [4, 2, 2]),
    ],
)
def test_mrope_triton_correctness(B, H_q, H_k, L, D, mrope_section):
    """
    Test that the Triton kernel matches the reference PyTorch implementation.
    """
    torch.manual_seed(0)
    device = "cuda"
    HALF = D // 2

    q = torch.rand((B, H_q, L, D), dtype=torch.float32, device=device)
    k = torch.rand((B, H_k, L, D), dtype=torch.float32, device=device)

    cos_half = torch.rand((1, L, HALF), dtype=torch.float32, device=device).expand(3, -1, -1)
    sin_half = torch.rand((1, L, HALF), dtype=torch.float32, device=device).expand(3, -1, -1)

    cos_full = torch.cat([cos_half, cos_half], dim=-1)
    sin_full = torch.cat([sin_half, sin_half], dim=-1)

    cos_ref = cos_full.unsqueeze(1)
    sin_ref = sin_full.unsqueeze(1)

    ref_q, ref_k = apply_multimodal_rotary_pos_emb(q, k, cos_ref, sin_ref, mrope_section, unsqueeze_dim=1)

    q = q.transpose(1, 2).contiguous().view(L, H_q, D)
    k = k.transpose(1, 2).contiguous().view(L, H_k, D)
    mrope_section = torch.tensor(mrope_section, dtype=torch.int32, device="cuda")
    mrope_triton_fused(q, k, cos_half, sin_half, mrope_section, is_interleaved=False)
    q = q.transpose(0, 1).contiguous().view(B, H_q, L, D)
    k = k.transpose(0, 1).contiguous().view(B, H_k, L, D)
    assert torch.allclose(q, ref_q, rtol=1e-3, atol=1e-3)
    assert torch.allclose(k, ref_k, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main()
