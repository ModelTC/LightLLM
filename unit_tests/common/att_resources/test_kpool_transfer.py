"""K-pool 融合传输：环形寻址、完整位模式、窗口外槽位保留及编译缓存。"""

from dataclasses import replace

import pytest
import torch

from lightllm.common.attention_state.att_tail import KPoolAttentionTailState, KPoolStateConfig
from lightllm.common.attention_state.att_tail import kpool_transfer
from lightllm.common.attention_state.types import AxisMapping, StateTransferLayout


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype,head_dim,byte_start", [
    (torch.bfloat16, 128, 16),
    (torch.float16, 33, 2),
    (torch.float32, 128, 4),
])
@pytest.mark.parametrize("page_device", ["cuda", "cpu"])
def test_kpool_single_launch_preserves_bits_and_other_slots(
    dtype: torch.dtype,
    head_dim: int,
    byte_start: int,
    page_device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = KPoolStateConfig(
        logical_layer_ids=(0, 3, 7),
        pool_size=4,
        head_dim=head_dim,
        dtype=dtype,
    )
    source = KPoolAttentionTailState(
        config=config,
        request_capacity=3,
        mtp_step=1,
        device='cuda',
    )
    target = KPoolAttentionTailState(
        config=config,
        request_capacity=3,
        mtp_step=5,
        device='cuda',
    )
    source.tail_buffer.view(torch.uint8).random_(0, 256)
    original = source.tail_buffer.clone()
    target.tail_buffer.view(torch.uint8).fill_(173)
    region = source.get_pd_transfer_regions()[0]
    # 同时验证非默认的外部轴顺序和字段起点；不能假定 window/channel 连续排列。
    region = replace(
        region,
        axes=("channel", "layer", "window"),
        shape=(2 * head_dim, 3, 4),
        byte_start=byte_start,
        byte_end=byte_start + region.byte_end,
    )
    layout = StateTransferLayout(
        state_name=source.state_name,
        tokens_per_page=None,
        byte_start=byte_start,
        byte_end=region.byte_end,
        regions=(region,),
    )
    pages = torch.full(
        size=(3, layout.byte_end + 16),
        fill_value=253,
        dtype=torch.uint8,
        device=page_device,
        pin_memory=page_device == "cpu",
    )
    kernel = kpool_transfer._transfer_kpool_pages
    launches = []

    class CountLaunches:
        def __getitem__(self, grid: tuple[int]):
            def launch(**kwargs: object) -> None:
                launches.append((grid, kwargs["WRITE"]))
                kernel[grid](**kwargs)
            return launch

    def reject_copy(*args: object, **kwargs: object) -> None:
        pytest.fail("K-pool transfer must copy in one Triton launch")

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with monkeypatch.context() as patch:
        patch.setattr(kpool_transfer, "_transfer_kpool_pages", CountLaunches())
        for method in ("copy_", "fill_", "zero_", "index_select", "index_copy_", "roll"):
            patch.setattr(torch.Tensor, method, reject_copy)
        with torch.cuda.stream(stream):
            source.write_to_pd_page(
                req_idx=1,
                pd_page_buffers=pages,
                pd_page_index=1,
                state_token_len=11,
                layout=layout,
            )
            target.read_from_pd_page(
                pd_page_buffers=pages,
                req_idx=2,
                pd_page_index=1,
                state_token_len=11,
                layout=layout,
            )
    stream.synchronize()
    assert len(launches) == 2 and [item[1] for item in launches] == [True, False]
    expected = torch.empty_like(target.tail_buffer[:, 2])
    expected.view(torch.uint8).fill_(173)
    for position in range(7, 11):
        expected[:, position % 9] = source.tail_buffer[:, 1, position % 5]
    assert torch.equal(target.tail_buffer[:, 2].view(torch.uint8), expected.view(torch.uint8))
    assert (target.tail_buffer[:, :2].view(torch.uint8) == 173).all()
    assert torch.equal(source.tail_buffer.view(torch.uint8), original.view(torch.uint8))
    assert (pages[[0, 2]] == 253).all()
    assert (pages[1, :byte_start] == 253).all() and (pages[1, layout.byte_end:] == 253).all()


@pytest.mark.parametrize("local_circular", [False, True])
@pytest.mark.parametrize("external_circular", [False, True])
def test_kpool_runtime_axis_overrides_are_bidirectional(
    local_circular: bool,
    external_circular: bool,
) -> None:
    config = KPoolStateConfig(
        logical_layer_ids=(0, 2),
        pool_size=4,
        head_dim=128,
        dtype=torch.bfloat16,
    )
    state = KPoolAttentionTailState(
        config=config,
        request_capacity=2,
        mtp_step=3,
        device='cuda',
    )
    state.tail_buffer.view(torch.uint8).random_(0, 256)
    original = state.tail_buffer[:, 1].clone()
    regions = state.get_pd_transfer_regions()
    layout = StateTransferLayout(
        state_name=state.state_name,
        tokens_per_page=None,
        byte_start=0,
        byte_end=regions[0].byte_end,
        regions=regions,
    )
    pages = torch.empty((1, layout.byte_end), dtype=torch.uint8, device="cuda")
    window = AxisMapping(
        axis="window",
        local_start=5 if local_circular else 1,
        external_start=3 if external_circular else 0,
        length=4,
        local_circular=local_circular,
        external_circular=external_circular,
    )
    kpool_transfer.transfer_kpool_pages(
        tail=state.tail_buffer[:, 1],
        page_buffers=pages,
        page_index=0,
        state_token_len=11,
        layout=layout,
        write=True,
        axis_mapping_overrides=(window,),
        grid_num=1,
    )
    external = pages[0].view(config.dtype).view(2, 4, 256)
    for index in range(4):
        assert torch.equal(
            external[:, (window.external_start + index) % 4].view(torch.uint8),
            original[:, (window.local_start + index) % 7].view(torch.uint8),
        )
    state.tail_buffer[:, 1].fill_(37)
    kpool_transfer.transfer_kpool_pages(
        tail=state.tail_buffer[:, 1],
        page_buffers=pages,
        page_index=0,
        state_token_len=11,
        layout=layout,
        write=False,
        axis_mapping_overrides=(window,),
        grid_num=1,
    )
    expected = torch.full_like(original, 37)
    for index in range(4):
        slot = (window.local_start + index) % 7
        expected[:, slot] = original[:, slot]
    assert torch.equal(state.tail_buffer[:, 1].view(torch.uint8), expected.view(torch.uint8))
    assert layout.regions[0].axis_mappings[0].local_start == 0


def test_kpool_tp_replica_skips_writes_but_restores_tail() -> None:
    config = KPoolStateConfig(
        logical_layer_ids=(0,),
        pool_size=4,
        head_dim=128,
        dtype=torch.bfloat16,
        tp_rank=1,
        tp_world_size=2,
    )
    state = KPoolAttentionTailState(
        config=config,
        request_capacity=2,
        mtp_step=2,
        device='cuda',
    )
    regions = state.get_pd_transfer_regions()
    layout = StateTransferLayout(
        state_name=state.state_name,
        tokens_per_page=None,
        byte_start=0,
        byte_end=regions[0].byte_end,
        regions=regions,
    )
    pages = torch.empty((1, layout.byte_end), dtype=torch.uint8, device="cuda")
    pages.view(torch.bfloat16).fill_(13)
    state.write_to_pd_page(
        req_idx=1,
        pd_page_buffers=pages,
        pd_page_index=0,
        state_token_len=11,
        layout=layout,
    )
    assert (pages.view(torch.bfloat16) == 13).all()
    state.read_from_pd_page(
        pd_page_buffers=pages,
        req_idx=1,
        pd_page_index=0,
        state_token_len=11,
        layout=layout,
    )
    expected = torch.zeros_like(state.tail_buffer[:, 1])
    for position in range(7, 11):
        expected[:, position % 6].fill_(13)
    assert torch.equal(state.tail_buffer[:, 1], expected)


def test_kpool_window_positions_do_not_create_kernel_variants() -> None:
    config = KPoolStateConfig(
        logical_layer_ids=(0,),
        pool_size=4,
        head_dim=128,
        dtype=torch.bfloat16,
    )
    state = KPoolAttentionTailState(
        config=config,
        request_capacity=2,
        mtp_step=5,
        device='cuda',
    )
    regions = state.get_pd_transfer_regions()
    layout = StateTransferLayout(
        state_name=state.state_name,
        tokens_per_page=None,
        byte_start=0,
        byte_end=regions[0].byte_end,
        regions=regions,
    )
    pages = torch.empty((1, layout.byte_end), dtype=torch.uint8, device="cuda")
    kernel = kpool_transfer._transfer_kpool_pages
    cache_size = None
    for length in (*range(30), (1 << 33) + 7):
        state.write_to_pd_page(
            req_idx=1,
            pd_page_buffers=pages,
            pd_page_index=0,
            state_token_len=length,
            layout=layout,
        )
        state.read_from_pd_page(
            pd_page_buffers=pages,
            req_idx=0,
            pd_page_index=0,
            state_token_len=length,
            layout=layout,
        )
        current_size = len(kernel.device_caches[torch.cuda.current_device()][0])
        if cache_size is None:
            cache_size = current_size
        assert current_size == cache_size
    torch.cuda.synchronize()


@pytest.mark.parametrize("length", [0, 3, 515])
def test_kpool_large_window_with_one_program(length: int) -> None:
    config = KPoolStateConfig(
        logical_layer_ids=(0, 3, 9, 12),
        pool_size=128,
        head_dim=128,
        dtype=torch.bfloat16,
    )
    source = KPoolAttentionTailState(
        config=config,
        request_capacity=2,
        mtp_step=2,
        device="cuda",
    )
    target = KPoolAttentionTailState(
        config=config,
        request_capacity=2,
        mtp_step=7,
        device="cuda",
    )
    source.tail_buffer.view(torch.uint8).random_(0, 256)
    target.tail_buffer.fill_(37)
    regions = source.get_pd_transfer_regions()
    layout = StateTransferLayout(
        state_name=source.state_name,
        tokens_per_page=None,
        byte_start=0,
        byte_end=regions[0].byte_end,
        regions=regions,
    )
    pages = torch.empty((1, layout.byte_end), dtype=torch.uint8, device="cuda")
    for state, write in ((source, True), (target, False)):
        kpool_transfer.transfer_kpool_pages(
            tail=state.tail_buffer[:, 1],
            page_buffers=pages,
            page_index=0,
            state_token_len=length,
            layout=layout,
            write=write,
            grid_num=1,
        )
    expected = torch.full_like(target.tail_buffer[:, 1], 37)
    for position in range(length - 128, length):
        if position < 0:
            expected[:, position % 135].zero_()
        else:
            expected[:, position % 135] = source.tail_buffer[:, 1, position % 130]
    assert torch.equal(target.tail_buffer[:, 1].view(torch.uint8), expected.view(torch.uint8))
    assert (target.tail_buffer[:, 0] == 37).all()
