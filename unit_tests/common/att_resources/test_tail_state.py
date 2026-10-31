from __future__ import annotations

import pytest
import torch

from lightllm.common.attention_state.att_tail import KPoolAttentionTailState, KPoolStateConfig
from lightllm.common.attention_state.types import (
    AxisMapping, StateTransferLayout, TransferRegion,
)


def make_tail( device: str | torch.device | int = "cpu", mtp_step: int = 2):
    return KPoolAttentionTailState(
        config=KPoolStateConfig(
            logical_layer_ids=(3, 7),
            pool_size=4,
            head_dim=2,
            dtype=torch.float32,
        ),
        request_capacity=3,
        mtp_step=mtp_step,
        device=device,
    )


def tail_layout(state: KPoolAttentionTailState):
    shape = (2, state.config.pool_size, 4)
    nbytes = shape[0] * shape[1] * shape[2] * 4
    return StateTransferLayout(
        state_name=state.state_name,
        tokens_per_page=None,
        byte_start=0,
        byte_end=nbytes + 16,
        regions=(
            TransferRegion(
                field="tail",
                byte_start=16,
                byte_end=16 + nbytes,
                dtype=torch.float32,
                axes=("layer", "window", "channel"),
                shape=shape,
                axis_mappings=(
                    AxisMapping(
                        axis="window",
                        local_start=0,
                        external_start=0,
                        length=shape[1],
                        local_circular=True,
                    ),
                    AxisMapping(
                        axis="channel",
                        local_start=0,
                        external_start=0,
                        length=4,
                    ),
                ),
            ),
        ),
    )


def test_tail_reset_clears_selected_requests_only():
    state = make_tail()
    state.tail_buffer.fill_(17)
    state.reset_attention_tail_state([0, 1])
    assert not state.tail_buffer[:, :2].any()
    assert (state.tail_buffer[:, 2] == 17).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("source_step,target_step", [(0, 0), (0, 5), (2, 0), (1, 3)])
@pytest.mark.parametrize("length", [0, 1, 3, 4, 5, 11, 12])
def test_pd_logical_window_restores_different_mtp_rings(
    device: str,
    source_step: int,
    target_step: int,
    length: int,
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    source = make_tail(device=device, mtp_step=source_step)
    target = make_tail(device=device, mtp_step=target_step)
    source.tail_buffer.copy_(torch.arange(source.tail_buffer.numel(), device=device).reshape_as(source.tail_buffer))
    original = source.tail_buffer.clone()
    target.tail_buffer.fill_(37)
    layout = tail_layout(source)
    assert layout == tail_layout(target)
    assert source.get_pd_transfer_regions() == target.get_pd_transfer_regions()
    pages = torch.full(
        size=(3, layout.byte_end + 16),
        fill_value=253,
        dtype=torch.uint8,
        device=device,
    )
    source.write_to_pd_page(
        req_idx=1,
        pd_page_buffers=pages,
        pd_page_index=1,
        state_token_len=length,
        layout=layout,
    )
    target.read_from_pd_page(
        pd_page_buffers=pages,
        req_idx=2,
        pd_page_index=1,
        state_token_len=length,
        layout=layout,
    )
    expected = torch.full_like(target.tail_buffer[:, 2], 37)
    for position in range(length - 4, length):
        slot = position % (4 + target_step)
        if position < 0:
            expected[:, slot].zero_()
        else:
            expected[:, slot] = source.tail_buffer[:, 1, position % (4 + source_step)]
    assert torch.equal(target.tail_buffer[:, 2], expected)
    assert (target.tail_buffer[:, :2] == 37).all()
    assert torch.equal(source.tail_buffer, original)
    assert (pages[0] == 253).all() and (pages[2] == 253).all()
    assert (pages[1, :16] == 253).all() and (pages[1, layout.byte_end:] == 253).all()
    if length < 4:
        external = pages[1, 16:layout.byte_end].view(torch.float32).view(2, 4, 4)
        assert not external[:, :4 - length].any()
