import os
import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")


def _randn_like(shape, dtype=torch.float32, device="cuda"):
    torch.manual_seed(1234)
    return torch.randn(shape, dtype=dtype, device=device).contiguous()


def _gather_along_size(t, idx):
    return t.index_select(dim=1, index=idx)


@pytest.mark.parametrize("head_dim", [64, 96])
@pytest.mark.parametrize(
    "kv_head_num,tp_world_size",
    [
        (4, 1),
        (4, 2),
        (8, 1),
        (8, 2),
        (8, 4),
        (16, 2),
    ],
)
def test_page_io_roundtrip_with_tp(head_dim, kv_head_num, tp_world_size):
    from lightllm.common.kv_trans_kernel.nixl_kv_trans import page_io

    device = "cuda"
    dtype = torch.bfloat16

    # Shapes
    layer_num = 3
    size = 32
    assert kv_head_num % 2 == 0
    page_size = 32
    page_head_num = kv_head_num * tp_world_size

    tp_indices = list(range(tp_world_size))

    kv_buffer = _randn_like((layer_num, size, kv_head_num, head_dim), dtype=dtype, device=device)
    page_tensor = torch.zeros((page_size, layer_num, page_head_num, head_dim), dtype=dtype, device=device).contiguous()

    # Select a handful of token positions to move (tid count <= page_size)
    mem_indexes = torch.tensor([2, 5, 7, 9, 11], dtype=torch.int64, device=device).contiguous()

    # Write: kv_buffer -> page_tensor, done by all tp ranks to fill their partition
    for tp_index in tp_indices:
        page_io(
            mem_indexes=mem_indexes,
            page_tensor=page_tensor,
            kv_buffer=kv_buffer,
            tp_index=tp_index,
            tp_world_size=tp_world_size,
            mode="write",
        )

    # After-write expectation check
    token_num = mem_indexes.numel()
    k_head_num = kv_head_num // 2
    page_k_head_num = page_head_num // 2
    page_write_head_num = page_k_head_num // tp_world_size

    expected_page = torch.zeros_like(page_tensor)
    for tid in range(token_num):
        mem_index = int(mem_indexes[tid].item())
        for layer_index in range(layer_num):
            for tp_index in tp_indices:
                page_head_start = tp_index * page_write_head_num
                for kv_head_id in range(page_write_head_num):
                    # K half
                    expected_page[tid, layer_index, page_head_start + kv_head_id, :] = kv_buffer[
                        layer_index, mem_index, kv_head_id, :
                    ]
                    # V half
                    expected_page[tid, layer_index, page_k_head_num + page_head_start + kv_head_id, :] = kv_buffer[
                        layer_index, mem_index, k_head_num + kv_head_id, :
                    ]

    assert torch.allclose(page_tensor[:token_num], expected_page[:token_num], atol=1e-3, rtol=1e-3)

    # Read back to a fresh buffer
    kv_buffer_rt = torch.zeros_like(kv_buffer)
    for tp_index in tp_indices:
        page_io(
            mem_indexes=mem_indexes,
            page_tensor=page_tensor,
            kv_buffer=kv_buffer_rt,
            tp_index=tp_index,
            tp_world_size=tp_world_size,
            mode="read",
        )

    # Check equality only at selected positions along size-dim
    ref = _gather_along_size(kv_buffer, mem_indexes)
    out = _gather_along_size(kv_buffer_rt, mem_indexes)
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("head_dim", [32, 80])  # include non-power-of-two
def test_mla_page_io_roundtrip(head_dim):
    from lightllm.common.kv_trans_kernel.nixl_kv_trans import mla_page_io

    device = "cuda"
    dtype = torch.bfloat16

    # Shapes (single-head)
    layer_num = 2
    size = 20
    kv_head_num = 1
    page_size = 10
    page_head_num = 1

    kv_buffer = _randn_like((layer_num, size, kv_head_num, head_dim), dtype=dtype, device=device)
    page_tensor = torch.zeros((page_size, layer_num, page_head_num, head_dim), dtype=dtype, device=device).contiguous()

    mem_indexes = torch.tensor([0, 3, 6, 7], dtype=torch.int64, device=device).contiguous()

    # Write kv -> page
    mla_page_io(mem_indexes=mem_indexes, page_tensor=page_tensor, kv_buffer=kv_buffer, mode="write")

    # After-write expectation check
    token_num = mem_indexes.numel()
    expected_page = torch.zeros_like(page_tensor)
    for tid in range(token_num):
        mem_index = int(mem_indexes[tid].item())
        for layer_index in range(layer_num):
            expected_page[tid, layer_index, 0, :] = kv_buffer[layer_index, mem_index, 0, :]
    assert torch.allclose(page_tensor[:token_num], expected_page[:token_num], atol=1e-3, rtol=1e-3)

    # Read back page -> kv
    kv_buffer_rt = torch.zeros_like(kv_buffer)
    mla_page_io(mem_indexes=mem_indexes, page_tensor=page_tensor, kv_buffer=kv_buffer_rt, mode="read")

    ref = kv_buffer.index_select(dim=1, index=mem_indexes)
    out = kv_buffer_rt.index_select(dim=1, index=mem_indexes)
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
