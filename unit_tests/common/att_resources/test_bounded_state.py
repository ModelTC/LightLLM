from __future__ import annotations
from lightllm.common.attention_state.bounded_att.operations import bind_state_operations
from unit_tests.common.att_resources.transfer_helpers import copy_bounded_peer, copy_pd_page, copy_snapshot_pages

from lightllm.common.attention_state.storage import allocate_runtime_storage, initialize_state_page_pools

from typing import TYPE_CHECKING
from dataclasses import replace

import pytest
import torch

from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.bounded_att import (
    GDNPagesPool,
    GDNState,
    KDAPagesPool,
    KDAState,
    ConvSSMStateConfig,
)
from lightllm.common.attention_state.types import (
    AxisMapping,
    StateTransferLayout,
    TransferRegion,
)

if TYPE_CHECKING:
    from lightllm.common.attention_state.types import PageKind


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for aggregated copies")


def make_state(
    cls: type[KDAState] | type[GDNState] = KDAState,
    device: str | torch.device | int = "cpu",
    heads: int = 1,
    mtp_step: int = 2,
):
    config = ConvSSMStateConfig(
        logical_layer_ids=(1, 3),
        global_k_heads=heads,
        global_v_heads=heads,
        key_dim=2,
        value_dim=2,
        conv_width=2,
        conv_dtype=torch.float32,
        ssm_dtype=torch.float32,
    )
    state = allocate_runtime_storage(cls(config=config, request_capacity=3, mtp_step=mtp_step), device)
    manager = AttentionStateManager(request_capacity=3)
    manager.register_state("main", state)
    initialize_state_page_pools(
        state, big_page_capacity=3, small_page_capacity=2, pin_memory=torch.device(device).type == "cuda"
    )
    return state


def page_layout(state: KDAState | GDNState):
    cfg = state.config
    conv_shape = (cfg.conv_channels, len(cfg.logical_layer_ids), cfg.conv_width)
    conv_bytes = cfg.conv_channels * len(cfg.logical_layer_ids) * cfg.conv_width * 4
    ssm_shape = (len(cfg.logical_layer_ids), cfg.num_v_heads, cfg.key_dim, cfg.value_dim)
    ssm_bytes = len(cfg.logical_layer_ids) * cfg.num_v_heads * cfg.key_dim * cfg.value_dim * 4
    end = 16 + conv_bytes + ssm_bytes
    return StateTransferLayout(
        state_name=state.state_name,
        tokens_per_page=None,
        byte_start=0,
        byte_end=end,
        regions=(
            TransferRegion(
                field="conv",
                byte_start=16,
                byte_end=16 + conv_bytes,
                dtype=torch.float32,
                axes=("channel", "layer", "window"),
                shape=conv_shape,
                axis_mappings=(
                    AxisMapping(
                        axis="channel",
                        local_start=0,
                        external_start=0,
                        length=cfg.conv_channels,
                    ),
                    AxisMapping(
                        axis="window",
                        local_start=0,
                        external_start=0,
                        length=cfg.conv_width,
                    ),
                ),
            ),
            TransferRegion(
                field="ssm",
                byte_start=16 + conv_bytes,
                byte_end=end,
                dtype=torch.float32,
                axes=("layer", "head", "key_dim", "value_dim"),
                shape=ssm_shape,
                axis_mappings=(
                    AxisMapping(
                        axis="head",
                        local_start=0,
                        external_start=0,
                        length=cfg.num_v_heads,
                    ),
                    AxisMapping(
                        axis="key_dim",
                        local_start=0,
                        external_start=0,
                        length=cfg.key_dim,
                    ),
                    AxisMapping(
                        axis="value_dim",
                        local_start=0,
                        external_start=0,
                        length=cfg.value_dim,
                    ),
                ),
            ),
        ),
    )


@pytest.mark.parametrize("cls,pool_cls", [(KDAState, KDAPagesPool), (GDNState, GDNPagesPool)])
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("page_kind", ["big", "small"])
def test_prefill_state_is_saved_and_restored(
    cls: type[KDAState] | type[GDNState],
    pool_cls: type[KDAPagesPool] | type[GDNPagesPool],
    device: str | torch.device | int,
    page_kind: PageKind,
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    config = ConvSSMStateConfig(
        logical_layer_ids=(1, 3),
        global_k_heads=1,
        global_v_heads=1,
        key_dim=2,
        value_dim=2,
        conv_width=2,
        conv_dtype=torch.float32,
        ssm_dtype=torch.float32,
    )
    state = allocate_runtime_storage(cls(config=config, request_capacity=3, mtp_step=2), device)
    assert state.state_name is None and not hasattr(config, "state_name")
    assert state.big_pages_pool is None and state.small_pages_pool is None
    conv, ssm = state.get_layer_view(3)
    assert conv.data_ptr() == state.conv_state[1].data_ptr()
    assert ssm.data_ptr() == state.ssm_state[1].data_ptr()
    for j in range(4):
        state.conv_state[:, 0, :, j].fill_(j + 10)
    for j in range(3):
        state.ssm_state[:, j].fill_(j + 20)
    state.mtp_state_index[2].fill_(1)
    # 延后创建页池不能重建运行态或重置已接受的 MTP 游标。
    initialize_state_page_pools(state, big_page_capacity=3, small_page_capacity=2, pin_memory=device == "cuda")
    assert conv.data_ptr() == state.conv_state[1].data_ptr()
    assert ssm.data_ptr() == state.ssm_state[1].data_ptr()
    from lightllm.common.attention_state.storage import StatePagesPool

    assert type(state.big_pages_pool) is StatePagesPool
    assert type(state.small_pages_pool) is StatePagesPool
    assert state.big_pages_pool is not state.small_pages_pool
    assert state.mtp_state_index.tolist() == [0, 0, 1]
    bind_state_operations(state).save(
        req_indices=[0],
        state_token_lens=[8],
        page_kind=page_kind,
        page_slot_ids=[1],
    )
    if device == "cuda":
        torch.cuda.synchronize()
    pool = state.big_pages_pool if page_kind == "big" else state.small_pages_pool
    assert torch.equal(pool.conv_state[1, ..., 0], torch.full_like(pool.conv_state[1, ..., 0], 10))
    assert torch.equal(pool.conv_state[1, ..., 1], torch.full_like(pool.conv_state[1, ..., 1], 11))
    assert (pool.ssm_state[1] == 20).all()
    state.mtp_state_index[1] = 2
    bind_state_operations(state).restore(
        req_indices=[1],
        state_token_lens=[8],
        page_kind=page_kind,
        page_slot_ids=[1],
    )
    assert state.mtp_state_index[1].item() == 0
    assert torch.equal(state.conv_state[:, 1, :, :2].cpu(), pool.conv_state[1])
    assert torch.equal(state.ssm_state[:, 3].cpu(), pool.ssm_state[1])


def test_gdn_dimensions_and_page_restore():
    config = replace(
        make_state().config,
        global_k_heads=1,
        global_v_heads=2,
        key_dim=2,
        value_dim=3,
    )
    state = allocate_runtime_storage(GDNState(config=config, request_capacity=3, mtp_step=2), "cuda")
    initialize_state_page_pools(state, big_page_capacity=2, small_page_capacity=2, pin_memory=True)
    conv = torch.arange(40, dtype=config.conv_dtype).reshape(2, 10, 2)
    ssm = torch.arange(24, dtype=config.ssm_dtype).reshape(
        shape=(2, 2, 2, 3),
    )
    state.conv_state[:, 0, :, :2].copy_(conv)
    state.ssm_state[:, 0].copy_(ssm)
    for page_kind in ("big", "small"):
        bind_state_operations(state).save(
            req_indices=[0],
            state_token_lens=[8],
            page_kind=page_kind,
            page_slot_ids=[0],
        )
    bind_state_operations(state).reset([0])
    for req, page_kind in enumerate(("big", "small"), start=1):
        bind_state_operations(state).restore(
            req_indices=[req],
            state_token_lens=[8],
            page_kind=page_kind,
            page_slot_ids=[0],
        )
        assert torch.equal(state.conv_state[:, req, :, :2].cpu(), conv)
        assert torch.equal(state.ssm_state[:, req * 3].cpu(), ssm)
    with pytest.raises(AssertionError, match="equal Q/K/V"):
        KDAPagesPool(2, config, pin_memory=False)


@pytest.mark.parametrize("cls", [KDAState, GDNState])
def test_reset_requests_preserves_pages(cls: type[KDAState] | type[GDNState]):
    state = make_state(cls, device="cuda")
    state.conv_state[:, 0].fill_(7)
    state.ssm_state[:, 0:3].fill_(11)
    bind_state_operations(state).save(
        req_indices=[0],
        state_token_lens=[6],
        page_kind="small",
        page_slot_ids=[0],
    )
    # 推进候选区和游标后，重置应清零全部运行态，同时保留已保存的页快照。
    state.conv_state[:, 0, :, 2:4].fill_(19)
    state.ssm_state[:, 2].fill_(23)
    state.mtp_state_index[0].fill_(2)
    bind_state_operations(state).reset([0])
    assert not state.conv_state[:, 0].any() and not state.ssm_state[:, :3].any()
    assert state.mtp_state_index[0] == 0
    assert (state.small_pages_pool.conv_state[0] == 7).all()
    assert (state.small_pages_pool.ssm_state[0] == 11).all()
    with pytest.raises(ValueError, match="recurrent state device"):
        bind_state_operations(state).commit(torch.tensor([0]), torch.tensor([1], dtype=torch.int32))


@pytest.mark.parametrize("cls", [KDAState, GDNState])
def test_cpu_pages_roundtrip_offsets(cls: type[KDAState] | type[GDNState]):
    source = make_state(cls, device="cuda")
    source.conv_state[:, 0].fill_(13)
    source.ssm_state[:, 0].fill_(29)
    bind_state_operations(source).save(
        req_indices=[0],
        state_token_lens=[8],
        page_kind="big",
        page_slot_ids=[0],
    )
    layout = page_layout(source)
    buffer = torch.full((3, layout.byte_end), 255, dtype=torch.uint8, pin_memory=True)
    copy_snapshot_pages(source, "big", [0], buffer, [2], layout, write=True, ready=None)
    torch.cuda.synchronize()
    assert (buffer[:2] == 255).all() and (buffer[2, :16] == 255).all()
    target = make_state(cls, device="cuda")
    copy_snapshot_pages(target, "big", [1], buffer, [2], layout, write=False, ready=None)
    bind_state_operations(target).restore(
        req_indices=[2],
        state_token_lens=[8],
        page_kind="big",
        page_slot_ids=[1],
    )
    assert (target.conv_state[:, 2, :, :2] == 13).all()
    assert (target.ssm_state[:, 6] == 29).all()
    # 页池只保存快照数据；前缀关联由调用方维护，读回的快照可以再次写出。
    copy_snapshot_pages(target, "big", [0], buffer, [2], layout, write=False, ready=None)
    copy_snapshot_pages(target, "big", [0], buffer, [0], layout, write=True, ready=None)
    torch.cuda.synchronize()
    assert torch.equal(buffer[0], buffer[2])


@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("cls", [KDAState, GDNState])
def test_pd_roundtrip_uses_baseline_state_and_page_relative_offset(
    cls: type[KDAState] | type[GDNState],
    device: str | torch.device | int,
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    source = make_state(cls, device=device)
    source.conv_state.fill_(-1)
    source.ssm_state.fill_(-1)
    source.conv_state[:, 0, :, :2].fill_(23)
    source.ssm_state[:, 0].fill_(47)
    layout = page_layout(source)
    # PD 只传 Conv/SSM，仍保留非零区域起点以覆盖组合页偏移。
    allocation = torch.full(
        size=(3, layout.byte_end),
        fill_value=254,
        dtype=torch.uint8,
        device=device,
    )
    page = allocation[1]
    copy_pd_page(source, allocation, 1, layout, write=True, req_idx=0, state_token_len=8)
    target = make_state(cls, device=device, mtp_step=1)
    copy_pd_page(target, allocation, 1, layout, write=False, req_idx=2, state_token_len=8)
    assert (target.conv_state[:, 2, :, :2] == 23).all()
    assert (target.ssm_state[:, 4] == 47).all()
    assert (allocation[0] == 254).all() and (allocation[2] == 254).all()
    assert (page[:16] == 254).all()
    assert target.mtp_state_index[2] == 0


@pytest.mark.parametrize("cls", [KDAState, GDNState])
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("src_mtp_step,dst_mtp_step", [(0, 2), (2, 1)])
def test_dp_runtime_copy_preserves_pages_and_other_requests(cls, device, src_mtp_step, dst_mtp_step):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    source = make_state(cls, device=device, mtp_step=src_mtp_step)
    target = make_state(cls, device=device, mtp_step=dst_mtp_step)
    source.conv_state.fill_(-7)
    source.ssm_state.fill_(-9)
    target.conv_state.fill_(-1)
    target.ssm_state.fill_(-2)
    target.mtp_state_index.fill_(1)
    target.big_pages_pool.conv_state.fill_(91)
    target.big_pages_pool.ssm_state.fill_(92)
    for req in (0, 2):
        source.conv_state[:, req, :, : source.config.conv_width].fill_(10 + req)
        source.ssm_state[:, req * (src_mtp_step + 1)].fill_(20 + req)
    src_conv = source.conv_state.clone()
    src_ssm = source.ssm_state.clone()

    copy_bounded_peer(source, target, [2, 0], [0, 1], runtime=True)

    for src_req, dst_req in ((2, 0), (0, 1)):
        assert (target.conv_state[:, dst_req, :, :2] == 10 + src_req).all()
        assert (target.conv_state[:, dst_req, :, 2:] == -1).all()
        dst_slot = dst_req * (dst_mtp_step + 1)
        assert (target.ssm_state[:, dst_slot] == 20 + src_req).all()
        assert (target.ssm_state[:, dst_slot + 1 : dst_slot + dst_mtp_step + 1] == -2).all()
        assert target.mtp_state_index[dst_req] == 0
    assert (target.conv_state[:, 2] == -1).all()
    assert (target.ssm_state[:, 2 * (dst_mtp_step + 1) :] == -2).all()
    assert target.mtp_state_index[2] == 1
    assert (target.big_pages_pool.conv_state == 91).all()
    assert (target.big_pages_pool.ssm_state == 92).all()
    assert torch.equal(source.conv_state, src_conv)
    assert torch.equal(source.ssm_state, src_ssm)


@pytest.mark.parametrize("cls", [KDAState, GDNState])
def test_dp_big_page_copy_preserves_runtime_and_other_pages(cls):
    source = make_state(cls, device="cuda")
    target = make_state(cls, device="cuda")
    for page in range(3):
        source.big_pages_pool.conv_state[page].fill_(10 + page)
        source.big_pages_pool.ssm_state[page].fill_(20 + page)
    target.big_pages_pool.conv_state.fill_(-1)
    target.big_pages_pool.ssm_state.fill_(-2)
    target.small_pages_pool.conv_state.fill_(31)
    target.small_pages_pool.ssm_state.fill_(32)
    target.conv_state.fill_(41)
    target.ssm_state.fill_(42)
    target.mtp_state_index.fill_(1)
    src_conv = source.big_pages_pool.conv_state.clone()
    src_ssm = source.big_pages_pool.ssm_state.clone()

    copy_bounded_peer(source, target, [2, 0], [0, 1], runtime=False)

    torch.cuda.synchronize()
    for src_page, dst_page in ((2, 0), (0, 1)):
        assert torch.equal(target.big_pages_pool.conv_state[dst_page], src_conv[src_page])
        assert torch.equal(target.big_pages_pool.ssm_state[dst_page], src_ssm[src_page])
    assert (target.big_pages_pool.conv_state[2] == -1).all()
    assert (target.big_pages_pool.ssm_state[2] == -2).all()
    assert (target.small_pages_pool.conv_state == 31).all()
    assert (target.small_pages_pool.ssm_state == 32).all()
    assert (target.conv_state == 41).all() and (target.ssm_state == 42).all()
    assert (target.mtp_state_index == 1).all()
    assert torch.equal(source.big_pages_pool.conv_state, src_conv)
    assert torch.equal(source.big_pages_pool.ssm_state, src_ssm)


@pytest.mark.parametrize("cls", [KDAState, GDNState])
@pytest.mark.parametrize("accepted", [0, 1, 2, 3])
@pytest.mark.parametrize("route", ["checkpoint", "dp", "peer_dp", "pd"])
def test_recurrent_transfer_selects_accepted_candidate(cls, accepted, route):
    if route == "peer_dp" and torch.cuda.device_count() < 2:
        pytest.skip("peer copy requires two CUDA devices")
    source = make_state(cls, device="cuda:0", mtp_step=3)
    target_device = "cuda:1" if route == "peer_dp" else "cuda:0"
    target = make_state(cls, device=target_device, mtp_step=1)
    source.conv_state.copy_(torch.arange(source.conv_state.numel(), device="cuda:0").reshape(source.conv_state.shape))
    source.ssm_state.copy_(torch.arange(source.ssm_state.numel(), device="cuda:0").reshape(source.ssm_state.shape))
    source.mtp_state_index[0] = accepted
    target.conv_state.fill_(-1)
    target.ssm_state.fill_(-2)
    target.mtp_state_index.fill_(1)
    expected_conv = source.conv_state[:, 0, :, accepted : accepted + source.config.conv_width].clone()
    expected_ssm = source.ssm_state[:, accepted].clone()
    layout = page_layout(source)
    pages = torch.full((2, layout.byte_end), 254, dtype=torch.uint8, device="cuda:0")
    torch.cuda.synchronize(0)

    with torch.cuda.device(target_device):
        if route == "peer_dp":
            from cuda.bindings import runtime

            if not torch.cuda.can_device_access_peer(1, 0):
                pytest.skip("CUDA peer access required")
            (error,) = runtime.cudaDeviceEnablePeerAccess(0, 0)
            assert error in (runtime.cudaError_t.cudaSuccess, runtime.cudaError_t.cudaErrorPeerAccessAlreadyEnabled)
        previous = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            if route == "checkpoint":
                bind_state_operations(source).save([0], [37], "big", [1])
                copy_bounded_peer(source, target, [1], [2], runtime=False)
                bind_state_operations(target).restore([1], [37], "big", [2])
            elif route in ("dp", "peer_dp"):
                copy_bounded_peer(source, target, [0], [1], runtime=True)
            else:
                copy_pd_page(source, pages, 1, layout, write=True, req_idx=0, state_token_len=37)
                copy_pd_page(target, pages, 1, layout, write=False, req_idx=1, state_token_len=37)
        finally:
            torch.cuda.set_sync_debug_mode(previous)

    torch.cuda.synchronize(target_device)
    torch.testing.assert_close(target.conv_state[:, 1, :, :2].cpu(), expected_conv.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(target.ssm_state[:, 2].cpu(), expected_ssm.cpu(), rtol=0, atol=0)
    assert (target.conv_state[:, 1, :, 2:] == -1).all()
    assert (target.conv_state[:, [0, 2]] == -1).all()
    assert (target.ssm_state[:, [0, 1, 3, 4, 5]] == -2).all()
    assert target.mtp_state_index.tolist() == [1, 0, 1]
    assert source.mtp_state_index[0] == accepted
    if route == "pd":
        assert (pages[0] == 254).all() and (pages[1, :16] == 254).all()
