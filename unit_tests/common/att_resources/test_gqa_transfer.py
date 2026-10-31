"""GQA 共用搬运 kernel 在受限 grid 下完整覆盖多轮、跨页和尾块数据。"""

from typing import TYPE_CHECKING

import pytest
import torch

from lightllm.common.attention_state.full_att import GQAAttentionConfig, GQAFullAttentionState
from lightllm.common.attention_state.full_att.gqa_transfer import transfer_gqa_pages
from lightllm.common.attention_state.types import StateTransferLayout

if TYPE_CHECKING:
    from easydict import EasyDict


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("grid_num", [1, 7, 36])
@pytest.mark.parametrize("page_device", ["cpu", "cuda"])
def test_gqa_grid_stride_covers_all_tokens(grid_num: int, page_device: str, pd_args: "EasyDict") -> None:
    pd_args.pd_kv_page_size = 32
    config = GQAAttentionConfig(
        logical_layer_groups=((0,), (2,), (4,)),
        global_head_num=3,
        head_dim=128,
        dtype=torch.bfloat16,
    )
    source = GQAFullAttentionState(layout_config=config, page_size=32, device="cuda")
    target = GQAFullAttentionState(layout_config=config, page_size=32, device="cuda")
    source.allocate_storage(token_capacity=128)
    target.allocate_storage(token_capacity=128)
    for tensor in (source.k_storage, source.v_storage):
        tensor.view(torch.uint8).random_(0, 256)
    for tensor in (target.k_storage, target.v_storage):
        tensor.view(torch.uint8).fill_(173)
    regions = source.get_pd_transfer_regions()
    layout = StateTransferLayout(
        state_name=source.state_name,
        tokens_per_page=32,
        byte_start=0,
        byte_end=max(region.byte_end for region in regions),
        regions=regions,
    )
    # 3 layers × 97 tokens × 3 heads × 128 dims，超过 36×256，且最后一块只有 128 个元素。
    token_count = 97
    src_ids = torch.randperm(n=128, device="cuda")[:token_count].to(dtype=torch.int32)
    dst_order = torch.randperm(n=128, device="cuda")
    dst_ids = dst_order[:token_count]
    pages = torch.full(
        size=(6, layout.byte_end + 32),
        fill_value=91,
        dtype=torch.uint8,
        device=page_device,
        pin_memory=page_device == "cpu",
    )
    page_view = pages[:, 16:-16]
    page_indices = [3, 0, 4, 1]
    transfer_gqa_pages(
        k_storage=source.k_storage,
        v_storage=source.v_storage,
        token_indices=src_ids,
        page_buffers=page_view,
        page_indices=page_indices,
        layout=layout,
        write=True,
        grid_num=grid_num,
    )
    transfer_gqa_pages(
        k_storage=target.k_storage,
        v_storage=target.v_storage,
        token_indices=dst_ids,
        page_buffers=page_view,
        page_indices=page_indices,
        layout=layout,
        write=False,
        grid_num=grid_num,
    )
    torch.cuda.synchronize()
    for src, dst in ((source.k_storage, target.k_storage), (source.v_storage, target.v_storage)):
        assert torch.equal(dst[:, dst_ids].contiguous().view(torch.uint8), src[:, src_ids].contiguous().view(torch.uint8))
        assert (dst[:, dst_order[token_count:]].contiguous().view(torch.uint8) == 173).all()
        assert (dst[:, 128:].view(torch.uint8) == 173).all()
    assert (pages[[2, 5]] == 91).all()
    assert (pages[:, :16] == 91).all() and (pages[:, -16:] == 91).all()
    for region in regions:
        last_page = page_view[1, region.byte_start:region.byte_end].view(region.dtype).view(region.shape)
        assert (last_page[:, 1:].contiguous().view(torch.uint8) == 91).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("page_device", ["cpu", "cuda"])
@pytest.mark.parametrize("num_stages", [1, 3])
@pytest.mark.parametrize("byte_start, extra_bytes, dim_stride, element_bytes", [
    (8, 32, 1, 8),
    (2, 32, 1, 2),
    (4, 32, 1, 4),
    (8, 34, 1, 2),
    (8, 36, 1, 4),
    (8, 32, 2, 2),
])
def test_wide_transfer_preserves_unaligned_pages_and_strided_fields(
    page_device: str,
    num_stages: int,
    byte_start: int,
    extra_bytes: int,
    dim_stride: int,
    element_bytes: int,
    pd_args: "EasyDict",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightllm.common.attention_state.full_att import gqa_transfer

    pd_args.pd_kv_page_size = 2
    config = GQAAttentionConfig(
        logical_layer_groups=((0,), (2,)),
        global_head_num=2,
        head_dim=128,
        dtype=torch.bfloat16,
    )
    source = GQAFullAttentionState(layout_config=config, page_size=2, device="cuda")
    target = GQAFullAttentionState(layout_config=config, page_size=2, device="cuda")
    source.allocate_storage(token_capacity=8)
    target.allocate_storage(token_capacity=8)
    for name in ("k_storage", "v_storage"):
        storage = getattr(source, name)
        if dim_stride != 1:
            storage = torch.empty(size=(*storage.shape[:-1], 256), dtype=storage.dtype, device="cuda")
        storage.view(torch.uint8).random_(0, 256)
        setattr(source, name, storage[..., ::dim_stride])
        getattr(target, name).view(torch.uint8).fill_(173)
    regions = source.get_pd_transfer_regions()
    layout = StateTransferLayout(
        state_name=source.state_name,
        tokens_per_page=2,
        byte_start=0,
        byte_end=max(region.byte_end for region in regions),
        regions=regions,
    )
    # 页起点、页跨度和最内维 stride 分别破坏宽类型对齐，回退必须保持完整字段及保护区。
    pages = torch.full(
        size=(3, layout.byte_end + extra_bytes),
        fill_value=91,
        dtype=torch.uint8,
        device=page_device,
        pin_memory=page_device == "cpu",
    )
    page_view = pages[:, byte_start:byte_start + layout.byte_end]
    src_ids = torch.tensor(data=[3, 3, 1, 1, 7, 7], dtype=torch.int32, device="cuda")[::2]
    dst_ids = torch.tensor(data=[6, 2, 4], dtype=torch.int64, device="cuda")
    expected = {}
    for name in ("k_storage", "v_storage"):
        expected[name] = getattr(target, name).clone()
        expected[name][:, dst_ids] = getattr(source, name)[:, src_ids]

    actual_kernel = gqa_transfer._copy_gqa_pages
    write_units = []

    class CaptureLaunch:
        def __getitem__(self, grid: tuple[int]):
            def launch(**kwargs):
                if kwargs["WRITE"]:
                    write_units.append(kwargs["ELEMENT_BYTES"])
                return actual_kernel[grid](**kwargs)
            return launch

    monkeypatch.setattr(gqa_transfer, "_copy_gqa_pages", CaptureLaunch())
    for state, ids, write in ((source, src_ids, True), (target, dst_ids, False)):
        transfer_gqa_pages(
            k_storage=state.k_storage,
            v_storage=state.v_storage,
            token_indices=ids,
            page_buffers=page_view,
            page_indices=[2, 0],
            layout=layout,
            write=write,
            num_stages=num_stages,
        )
    torch.cuda.synchronize()
    assert write_units == [element_bytes, element_bytes]
    for name in ("k_storage", "v_storage"):
        assert torch.equal(getattr(target, name).view(torch.uint8), expected[name].view(torch.uint8))
    assert (pages[1] == 91).all()
    assert (pages[:, :byte_start] == 91).all()
    assert (pages[:, byte_start + layout.byte_end:] == 91).all()
    for region in regions:
        tail_page = page_view[0, region.byte_start:region.byte_end].view(region.dtype).view(region.shape)
        assert (tail_page[:, 1:].contiguous().view(torch.uint8) == 91).all()
