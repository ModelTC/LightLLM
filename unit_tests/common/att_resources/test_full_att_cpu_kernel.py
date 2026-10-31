from unit_tests.common.att_resources.model_helpers import full_state_test_fields

"""CPU 页直接拷贝：真实索引、跨页布局、原始编码与未参与区域的保持。"""

from dataclasses import replace

import pytest
import torch

from unit_tests.common.att_resources.model_helpers import full_state_test_buffers
from unit_tests.common.att_resources.test_state_identity import allocated_state
from unit_tests.common.att_resources.test_quantized_full_state import transfer_layout
from lightllm.common.attention_state.types import AxisMapping


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", [
    "gqa", "mla", "int4", "int8", "static_head", "static_tensor", "packed", "packed_fp8", "packed_glm",
])
@pytest.mark.parametrize("token_major", [False, True])
def test_direct_cpu_pages_preserve_bytes_without_gather(kind: str, token_major: bool, monkeypatch: pytest.MonkeyPatch):
    source = allocated_state(kind, device="cuda")
    target = allocated_state(kind, device="cuda")
    for tensor in full_state_test_buffers(source).values():
        tensor.view(torch.uint8).random_(0, 256)
    for tensor in full_state_test_buffers(target).values():
        tensor.view(torch.uint8).fill_(173)
    layout = transfer_layout(source)
    if token_major:
        regions = []
        for region in layout.regions:
            axes = ("token", "layer", *region.axes[2:])
            regions.append(replace(region, axes=axes, shape=tuple(region.shape[region.axes.index(a)] for a in axes)))
        layout = replace(layout, regions=tuple(regions))
    # 页池本身是更大联合页的视图，检验真实 page stride，而不是按字段长度推算下一页。
    pages = torch.full(
        size=(4, layout.byte_end + 64),
        fill_value=91,
        dtype=torch.uint8,
        pin_memory=True,
    )
    page_view = pages[:, 32:-32]
    src_ids = torch.tensor([6, 6, 1, 1, 7, 7, 0, 0, 3, 3], dtype=torch.int32, device="cuda")[::2]
    dst_ids = torch.tensor([2, 2, 6, 6, 4, 4, 1, 1, 7, 7], dtype=torch.int64, device="cuda")[::2]
    expected = {name: tensor[:, dst_ids].clone() for name, tensor in full_state_test_buffers(target).items()}
    source_fields = full_state_test_fields(source, full_state_test_buffers(source))
    expected_fields = full_state_test_fields(target, expected)
    for name, field in source_fields.items():
        expected_fields[name].tensor.copy_(field.tensor[:, src_ids])

    def reject_gather(*args: object, **kwargs: object) -> None:
        pytest.fail("CPU transfer must not allocate gathered KV or scatter it back")

    tensor_to = torch.Tensor.to
    tensor_contiguous = torch.Tensor.contiguous

    def reject_index_to(tensor: torch.Tensor, *args: object, **kwargs: object):
        if tensor is src_ids or tensor is dst_ids:
            pytest.fail("token indices must be passed directly to the kernel")
        return tensor_to(tensor, *args, **kwargs)

    def reject_index_contiguous(tensor: torch.Tensor, *args: object, **kwargs: object):
        if tensor is src_ids or tensor is dst_ids:
            pytest.fail("token index views must not be materialized")
        return tensor_contiguous(tensor, *args, **kwargs)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "to", reject_index_to)
        patch.setattr(torch.Tensor, "contiguous", reject_index_contiguous)
        patch.setattr(torch.Tensor, "index_select", reject_gather)
        patch.setattr(torch.Tensor, "index_copy_", reject_gather)
        with torch.cuda.stream(stream):
            source.write_to_cpu(
                token_indices=src_ids,
                cpu_page_buffers=page_view,
                cpu_page_indices=[2, 0],
                layout=layout,
            )
            target.read_from_cpu(
                cpu_page_buffers=page_view,
                cpu_page_indices=[2, 0],
                token_indices=dst_ids,
                layout=layout,
            )
        stream.synchronize()
    for name, tensor in full_state_test_buffers(target).items():
        assert torch.equal(tensor[:, dst_ids].contiguous().view(torch.uint8), expected[name].view(torch.uint8))
        assert (tensor[:, [0, 3, 5, 8, 9, 10, 11]].contiguous().view(torch.uint8) == 173).all()
    assert (pages[:, :32] == 91).all() and (pages[:, -32:] == 91).all()
    assert (pages[[1, 3]] == 91).all()
    for region in layout.regions:
        external = page_view[0, region.byte_start:region.byte_end].view(region.dtype).view(region.shape)
        # 最后一页只有一个有效 token，其余槽位仍保留填充值。
        unused = [slice(None)] * len(region.shape)
        unused[region.axes.index("token")] = slice(1, None)
        assert (external[tuple(unused)].contiguous().view(torch.uint8) == 91).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_direct_cpu_pages_only_update_selected_head_slice_and_skip_replica_writes():
    state = allocated_state("gqa", device="cuda")
    state.k_storage.fill_(3)
    state.v_storage.fill_(7)
    layout = transfer_layout(state)
    regions = []
    for region in layout.regions:
        slices = tuple(
            AxisMapping(
                axis="head",
                local_start=1,
                external_start=0,
                length=1,
            ) if item.axis == "head" else item
            for item in region.axis_mappings
        )
        regions.append(replace(region, axis_mappings=slices))
    layout = replace(layout, regions=tuple(regions))
    pages = torch.zeros((1, layout.byte_end), dtype=torch.uint8, pin_memory=True)
    ids = torch.tensor([1, 5], dtype=torch.int64, device="cuda")
    state.write_to_cpu(
        token_indices=ids,
        cpu_page_buffers=pages,
        cpu_page_indices=[0],
        layout=layout,
    )
    torch.cuda.synchronize()
    snapshot = pages.clone()
    state.k_storage.fill_(19)
    state.v_storage.fill_(19)
    read_only = replace(layout, regions=tuple(replace(region, write_enabled=False) for region in layout.regions))
    state.write_to_cpu(
        token_indices=ids,
        cpu_page_buffers=pages,
        cpu_page_indices=[0],
        layout=read_only,
    )
    state.read_from_cpu(
        cpu_page_buffers=pages,
        cpu_page_indices=[0],
        token_indices=ids,
        layout=read_only,
    )
    torch.cuda.synchronize()
    assert torch.equal(pages, snapshot)
    assert (state.k_storage[:, ids, 1] == 3).all() and (state.v_storage[:, ids, 1] == 7).all()
    assert (state.k_storage[:, :, 0] == 19).all() and (state.v_storage[:, :, 0] == 19).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_direct_cpu_pages_accept_registered_external_memory():
    """共享缓存通过 cudaHostRegister 注册，而非 torch 的 pinned allocator 分配。"""
    import ctypes
    import mmap

    state = allocated_state("gqa", device="cuda")
    state.k_storage.fill_(3)
    state.v_storage.fill_(7)
    layout = transfer_layout(state)
    memory = mmap.mmap(-1, layout.byte_end)
    pages = torch.frombuffer(memory, dtype=torch.uint8).view(1, -1)
    cuda = ctypes.CDLL("/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so")
    cuda.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    cuda.cudaHostUnregister.argtypes = [ctypes.c_void_p]
    pointer = ctypes.c_void_p(pages.data_ptr())
    assert cuda.cudaHostRegister(pointer, layout.byte_end, 3) == 0
    try:
        assert pages.is_pinned()
        ids = torch.tensor([1, 5], dtype=torch.int32, device="cuda")
        state.write_to_cpu(
            token_indices=ids,
            cpu_page_buffers=pages,
            cpu_page_indices=[0],
            layout=layout,
        )
        state.k_storage.zero_()
        state.v_storage.zero_()
        state.read_from_cpu(
            cpu_page_buffers=pages,
            cpu_page_indices=[0],
            token_indices=ids,
            layout=layout,
        )
        torch.cuda.synchronize()
        assert (state.k_storage[:, ids] == 3).all() and (state.v_storage[:, ids] == 7).all()
    finally:
        torch.cuda.synchronize()
        assert cuda.cudaHostUnregister(pointer) == 0
        del pages
        memory.close()


@pytest.mark.parametrize("kind", ["gqa", "int8", "static_head", "mla", "packed_fp8"])
@pytest.mark.parametrize("write", [True, False])
def test_cpu_cache_transfer_rejects_cpu_state(kind: str, write: bool):
    state = allocated_state(kind, device="cpu")
    layout = transfer_layout(state)
    pages = torch.zeros((1, layout.byte_end), dtype=torch.uint8)
    ids = torch.tensor([0], dtype=torch.int64)
    with pytest.raises(AssertionError, match="CUDA State storage"):
        if write:
            state.write_to_cpu(
                token_indices=ids,
                cpu_page_buffers=pages,
                cpu_page_indices=[0],
                layout=layout,
            )
        else:
            state.read_from_cpu(
                cpu_page_buffers=pages,
                cpu_page_indices=[0],
                token_indices=ids,
                layout=layout,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("token_count", [0, 1])
def test_gqa_single_cpu_page_and_empty_transfer(token_count: int) -> None:
    source = allocated_state("gqa", device="cuda")
    target = allocated_state("gqa", device="cuda")
    source.k_storage.fill_(3)
    source.v_storage.fill_(7)
    target.k_storage.fill_(19)
    target.v_storage.fill_(19)
    layout = transfer_layout(source)
    pages = torch.full(
        size=(3, layout.byte_end),
        fill_value=91,
        dtype=torch.uint8,
        pin_memory=True,
    )
    src_ids = torch.tensor(data=[5], dtype=torch.int32, device="cuda")[:token_count]
    dst_ids = torch.tensor(data=[2], dtype=torch.int64, device="cuda")[:token_count]
    page_indices = [1] if token_count else []

    source.write_to_cpu(
        token_indices=src_ids,
        cpu_page_buffers=pages,
        cpu_page_indices=page_indices,
        layout=layout,
    )
    target.read_from_cpu(
        cpu_page_buffers=pages,
        cpu_page_indices=page_indices,
        token_indices=dst_ids,
        layout=layout,
    )
    torch.cuda.synchronize()
    if token_count:
        assert (target.k_storage[:, 2] == 3).all() and (target.v_storage[:, 2] == 7).all()
        assert (target.k_storage[:, [0, 1, 3, 4, 5, 6, 7]] == 19).all()
        assert (target.v_storage[:, [0, 1, 3, 4, 5, 6, 7]] == 19).all()
    else:
        assert (target.k_storage == 19).all() and (target.v_storage == 19).all()
        assert (pages[1] == 91).all()
    assert (pages[[0, 2]] == 91).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", [
    "gqa", "mla", "int4", "int8", "static_head", "static_tensor", "packed", "packed_fp8", "packed_glm",
])
def test_full_state_interfaces_reject_cpu_token_indices(kind: str) -> None:
    state = allocated_state(kind, device="cuda")
    layout = transfer_layout(state)
    cpu_ids = torch.tensor([0], dtype=torch.int32)
    cuda_ids = torch.tensor([1], dtype=torch.int64, device="cuda")
    cpu_pages = torch.empty((1, layout.byte_end), dtype=torch.uint8, pin_memory=True)
    pd_pages = torch.empty((1, layout.byte_end), dtype=torch.uint8, device="cuda")
    with pytest.raises(AssertionError, match="token indices must reside on CUDA"):
        state.write_full_att_kv(state.config.logical_layer_groups[0][0], cpu_ids, torch.empty(0, device="cuda"))
    for src, dst in ((cpu_ids, cuda_ids), (cuda_ids, cpu_ids)):
        with pytest.raises(AssertionError, match="token indices must reside on CUDA"):
            state.copy_full_att_tokens(src, dst)
        with pytest.raises(AssertionError, match="token indices must reside on CUDA"):
            state.copy_tokens_from_dp(src, dst, state)
    with pytest.raises(AssertionError, match="token indices must reside on CUDA"):
        state.write_to_cpu(
            token_indices=cpu_ids,
            cpu_page_buffers=cpu_pages,
            cpu_page_indices=[0],
            layout=layout,
        )
    with pytest.raises(AssertionError, match="token indices must reside on CUDA"):
        state.read_from_cpu(
            cpu_page_buffers=cpu_pages,
            cpu_page_indices=[0],
            token_indices=cpu_ids,
            layout=layout,
        )
    with pytest.raises(AssertionError, match="token indices must reside on CUDA"):
        state.write_to_pd_page(
            token_indices=cpu_ids,
            pd_page_buffers=pd_pages,
            pd_page_index=0,
            layout=layout,
        )
    with pytest.raises(AssertionError, match="token indices must reside on CUDA"):
        state.read_from_pd_page(
            pd_page_buffers=pd_pages,
            token_indices=cpu_ids,
            pd_page_index=0,
            layout=layout,
        )
