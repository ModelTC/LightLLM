import pytest
import torch

from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv_split


def _strided_indices(capacity, tokens, index_dtype):
    """返回 stride 不为 1 的目标索引，覆盖 DestLoc.stride(0)。"""
    storage = torch.empty(tokens, 2, device="cuda", dtype=index_dtype)
    storage[:, 0] = torch.randperm(capacity, device="cuda")[:tokens]
    return storage[:, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("heads", [1, 8, 40])
@pytest.mark.parametrize("head_dim", [80, 128])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_destindex_copy_kv_split_matches_reference(heads, head_dim, index_dtype):
    torch.manual_seed(0)
    tokens, capacity = 17, 22
    dtype = torch.bfloat16
    kv_storage = torch.randn(tokens, 2 * heads, head_dim * 2, device="cuda", dtype=dtype)
    kv = kv_storage[:, :, ::2]
    dest = _strided_indices(capacity, tokens, index_dtype)
    out_k = torch.full((capacity, heads, head_dim), 7, device="cuda", dtype=dtype)
    out_v = torch.full((capacity, heads, head_dim), 9, device="cuda", dtype=dtype)

    destindex_copy_kv_split(kv, dest, out_k, out_v)

    ref_k = torch.full_like(out_k, 7)
    ref_v = torch.full_like(out_v, 9)
    ref_k[dest.long()] = kv[:, :heads]
    ref_v[dest.long()] = kv[:, heads:]
    torch.testing.assert_close(out_k, ref_k)
    torch.testing.assert_close(out_v, ref_v)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_destindex_copy_kv_split_empty_indices_leave_storage_unchanged():
    heads, head_dim = 8, 80
    kv = torch.randn(0, 2 * heads, head_dim, device="cuda", dtype=torch.bfloat16)
    dest = torch.empty(0, device="cuda", dtype=torch.int64)
    out_k = torch.full((3, heads, head_dim), 7, device="cuda", dtype=torch.bfloat16)
    out_v = torch.full((3, heads, head_dim), 9, device="cuda", dtype=torch.bfloat16)
    before_k, before_v = out_k.clone(), out_v.clone()

    destindex_copy_kv_split(kv, dest, out_k, out_v)

    torch.testing.assert_close(out_k, before_k)
    torch.testing.assert_close(out_v, before_v)
