import sys
from pathlib import Path

import pytest
import torch

CUR_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str((CUR_DIR / "../../../lightllm/models/deepseek3_2/triton_kernel").resolve()))

from destindex_copy_kv_flashmla_fp8 import dequantize_kv_reference, pack_kv_reference


def _manual_sparse_decode(
    q: torch.Tensor, dense_kv: torch.Tensor, indices: torch.Tensor, sm_scale: float
) -> torch.Tensor:
    batch, _, heads, _ = q.shape
    topk = indices.shape[-1]
    out = torch.zeros((batch, heads, 512), dtype=torch.float32, device=q.device)

    for b in range(batch):
        valid = indices[b, 0] >= 0
        cur_idx = indices[b, 0, valid]
        assert cur_idx.numel() > 0
        cur_k = dense_kv[cur_idx, 0, :]
        cur_v = cur_k[:, :512]
        logits = torch.einsum("hd,td->ht", q[b, 0].float(), cur_k.float()) * sm_scale
        probs = torch.softmax(logits, dim=-1)
        out[b] = torch.einsum("ht,td->hd", probs, cur_v.float())

        if cur_idx.numel() < topk:
            assert torch.all(indices[b, 0, cur_idx.numel() :] == -1)

    return out.to(torch.bfloat16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flashmla_fp8_sparse_decode_matches_manual_reference():
    import flash_mla

    batch = 2
    seq_q = 1
    heads = 64
    token_num = 128
    topk = 64
    dtype = torch.bfloat16
    device = "cuda"

    q = torch.randn((batch, seq_q, heads, 576), dtype=dtype, device=device)
    kv = torch.randn((token_num, 1, 576), dtype=dtype, device=device)
    packed = pack_kv_reference(kv).view(token_num, 1, 1, 656)

    indices = torch.randint(0, token_num, (batch, seq_q, topk), dtype=torch.int32, device=device)
    indices[0, 0, -3:] = -1
    indices[1, 0, -5:] = -1
    sm_scale = 576 ** (-0.5)

    sched_meta, _ = flash_mla.get_mla_metadata()
    out, _ = flash_mla.flash_mla_with_kvcache(
        q=q,
        k_cache=packed,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=512,
        tile_scheduler_metadata=sched_meta,
        num_splits=None,
        softmax_scale=sm_scale,
        causal=False,
        is_fp8_kvcache=True,
        indices=indices,
    )
    torch.cuda.synchronize()

    dense_kv = dequantize_kv_reference(packed)
    ref = _manual_sparse_decode(q, dense_kv, indices, sm_scale)
    assert torch.allclose(out[:, 0], ref, rtol=7e-2, atol=7e-2)
