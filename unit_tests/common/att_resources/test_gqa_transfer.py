"""GQA 共用搬运 kernel 在受限 grid 下完整覆盖多轮、跨页和尾块数据。"""

from typing import TYPE_CHECKING

import pytest
import torch

from lightllm.common.att_resources.full_att import FullAttentionConfig, GQAFullAttentionState
from lightllm.common.att_resources.full_att.gqa_transfer import transfer_gqa_pages
from lightllm.common.att_resources.types import StateTransferLayout

if TYPE_CHECKING:
    from easydict import EasyDict


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("grid_num", [1, 7, 36])
@pytest.mark.parametrize("page_device", ["cpu", "cuda"])
def test_gqa_grid_stride_covers_all_tokens(grid_num: int, page_device: str, pd_args: "EasyDict") -> None:
    pd_args.pd_kv_page_size = 32
    config = FullAttentionConfig(layer_ids=(0, 2, 4), global_head_num=3, head_dim=128, dtype=torch.bfloat16)
    source = GQAFullAttentionState(layout_config=config, page_size=32, device="cuda")
    target = GQAFullAttentionState(layout_config=config, page_size=32, device="cuda")
    source.initialize_storage(token_capacity=128)
    target.initialize_storage(token_capacity=128)
    for tensor in (source.k_storage, source.v_storage):
        tensor.view(torch.uint8).random_(0, 256)
    for tensor in (target.k_storage, target.v_storage):
        tensor.view(torch.uint8).fill_(173)
    regions = source.get_pd_transfer_regions()
    layout = StateTransferLayout(
        resource_id=source.resource_id,
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
