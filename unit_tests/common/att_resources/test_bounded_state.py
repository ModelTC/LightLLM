from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import replace

import pytest
import torch

from lightllm.common.att_resources import AttResourceManager
from lightllm.common.att_resources.bounded_att import (
    GDNPagesPool,
    GDNState,
    KDAPagesPool,
    KDAState,
    ConvSSMStateConfig,
)
from lightllm.common.att_resources.types import (
    AxisSlice, StateTransferLayout, TransferRegion,
)

if TYPE_CHECKING:
    from lightllm.common.att_resources.types import PageKind


def make_state(
    cls: type[KDAState] | type[GDNState] = KDAState,
    *,
    device: str | torch.device | int = "cpu",
    heads: int = 1,
    mtp_step: int = 2,
):
    config = ConvSSMStateConfig(
        layer_ids=(1, 3), global_k_heads=heads, global_v_heads=heads, key_dim=2, value_dim=2, conv_width=2,
        conv_dtype=torch.float32, ssm_dtype=torch.float32,
    )
    state = cls(config, request_capacity=3, mtp_step=mtp_step, device=device)
    manager = AttResourceManager(request_capacity=3)
    manager.set_att_state("main", state)
    state.initialize_page_pools(big_page_capacity=3, small_page_capacity=2, pin_memory=device != "cpu")
    return state


def page_layout(state: KDAState | GDNState):
    cfg = state.config
    conv_shape = (cfg.conv_channels, len(cfg.layer_ids), cfg.conv_width)
    conv_bytes = cfg.conv_channels * len(cfg.layer_ids) * cfg.conv_width * 4
    ssm_shape = (len(cfg.layer_ids), cfg.num_v_heads, cfg.key_dim, cfg.value_dim)
    ssm_bytes = len(cfg.layer_ids) * cfg.num_v_heads * cfg.key_dim * cfg.value_dim * 4
    end = 16 + conv_bytes + ssm_bytes
    return StateTransferLayout(
        resource_id=state.resource_id,
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
                axis_slices=(
                    AxisSlice(axis="channel", src_start=0, dst_start=0, length=cfg.conv_channels),
                    AxisSlice(axis="window", src_start=0, dst_start=0, length=cfg.conv_width),
                ),
            ),
            TransferRegion(
                field="ssm",
                byte_start=16 + conv_bytes,
                byte_end=end,
                dtype=torch.float32,
                axes=("layer", "head", "key_dim", "value_dim"),
                shape=ssm_shape,
                axis_slices=(
                    AxisSlice(axis="head", src_start=0, dst_start=0, length=cfg.num_v_heads),
                    AxisSlice(axis="key_dim", src_start=0, dst_start=0, length=cfg.key_dim),
                    AxisSlice(axis="value_dim", src_start=0, dst_start=0, length=cfg.value_dim),
                ),
            ),
        ),
    )


@pytest.mark.parametrize("cls,pool_cls", [(KDAState, KDAPagesPool), (GDNState, GDNPagesPool)])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
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
        layer_ids=(1, 3), global_k_heads=1, global_v_heads=1, key_dim=2, value_dim=2, conv_width=2,
        conv_dtype=torch.float32, ssm_dtype=torch.float32,
    )
    state = cls(config, request_capacity=3, mtp_step=2, device=device)
    assert state.resource_id is None and not hasattr(config, "resource_id")
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
    state.initialize_page_pools(big_page_capacity=3, small_page_capacity=2, pin_memory=device == "cuda")
    assert conv.data_ptr() == state.conv_state[1].data_ptr()
    assert ssm.data_ptr() == state.ssm_state[1].data_ptr()
    assert type(state.big_pages_pool) is pool_cls
    assert type(state.small_pages_pool) is pool_cls
    assert state.big_pages_pool is not state.small_pages_pool
    assert state.mtp_state_index.tolist() == [0, 0, 1]
    state.save_bounded_att_pages([0], page_kind=page_kind, slot_ids=[1])
    if device == "cuda":
        torch.cuda.synchronize()
    pool = state.big_pages_pool if page_kind == "big" else state.small_pages_pool
    assert torch.equal(pool.conv_state[1, ..., 0], torch.full_like(pool.conv_state[1, ..., 0], 10))
    assert torch.equal(pool.conv_state[1, ..., 1], torch.full_like(pool.conv_state[1, ..., 1], 11))
    assert (pool.ssm_state[1] == 20).all()
    state.mtp_state_index[1] = 2
    state.restore_bounded_att_pages([1], page_kind=page_kind, slot_ids=[1])
    assert state.mtp_state_index[1].item() == 0
    assert torch.equal(state.conv_state[:, 1, :, :2].cpu(), pool.conv_state[1])
    assert torch.equal(state.ssm_state[:, 3].cpu(), pool.ssm_state[1])


def test_gdn_dimensions_and_page_restore():
    config = replace(
        make_state().config, global_k_heads=1, global_v_heads=2, key_dim=2, value_dim=3
    )
    state = GDNState(config, request_capacity=3, mtp_step=2, device="cpu")
    state.initialize_page_pools(big_page_capacity=2, small_page_capacity=2, pin_memory=False)
    conv = torch.arange(40, dtype=config.conv_dtype).reshape(2, 10, 2)
    ssm = torch.arange(24, dtype=config.ssm_dtype).reshape(2, 2, 2, 3)
    state.conv_state[:, 0, :, :2].copy_(conv)
    state.ssm_state[:, 0].copy_(ssm)
    for page_kind in ("big", "small"):
        state.save_bounded_att_pages([0], page_kind=page_kind, slot_ids=[0])
    state.reset_bounded_att_state([0])
    for req, page_kind in enumerate(("big", "small"), start=1):
        state.restore_bounded_att_pages([req], page_kind=page_kind, slot_ids=[0])
        assert torch.equal(state.conv_state[:, req, :, :2], conv)
        assert torch.equal(state.ssm_state[:, req * 3], ssm)
    with pytest.raises(AssertionError, match="equal Q/K/V"):
        KDAPagesPool(2, config, pin_memory=False)


@pytest.mark.parametrize("cls", [KDAState, GDNState])
def test_reset_requests_preserves_pages(cls: type[KDAState] | type[GDNState]):
    state = make_state(cls)
    state.conv_state[:, 0].fill_(7)
    state.ssm_state[:, 0:3].fill_(11)
    state.save_bounded_att_pages([0], page_kind="small", slot_ids=[0])
    # 推进候选区和游标后，重置应清零全部运行态，同时保留已保存的页快照。
    state.conv_state[:, 0, :, 2:4].fill_(19)
    state.ssm_state[:, 2].fill_(23)
    state.mtp_state_index[0].fill_(2)
    state.reset_bounded_att_state([0])
    assert not state.conv_state[:, 0].any() and not state.ssm_state[:, :3].any()
    assert state.mtp_state_index[0] == 0
    assert (state.small_pages_pool.conv_state[0] == 7).all()
    assert (state.small_pages_pool.ssm_state[0] == 11).all()
    with pytest.raises(AssertionError, match="CUDA tensors"):
        state.commit_bounded_att_state(torch.tensor([0]), torch.tensor([1], dtype=torch.int32))


@pytest.mark.parametrize("cls", [KDAState, GDNState])
def test_cpu_pages_roundtrip_offsets(cls: type[KDAState] | type[GDNState]):
    source = make_state(cls)
    source.conv_state[:, 0].fill_(13)
    source.ssm_state[:, 0].fill_(29)
    source.save_bounded_att_pages([0], page_kind="big", slot_ids=[0])
    layout = page_layout(source)
    buffer = torch.full((3, layout.byte_end), 255, dtype=torch.uint8)
    source.write_page_state_to_cpu("big", [0], buffer, [2], layout=layout)
    assert (buffer[:2] == 255).all() and (buffer[2, :16] == 255).all()
    target = make_state(cls)
    target.read_page_state_from_cpu(buffer, [2], big_page_slot_ids=[1], layout=layout)
    target.restore_bounded_att_pages([2], page_kind="big", slot_ids=[1])
    assert (target.conv_state[:, 2, :, :2] == 13).all()
    assert (target.ssm_state[:, 6] == 29).all()
    # 页池只保存快照数据；前缀关联由调用方维护，读回的快照可以再次写出。
    target.read_page_state_from_cpu(buffer, [2], big_page_slot_ids=[0], layout=layout)
    target.write_page_state_to_cpu("big", [0], buffer, [0], layout=layout)
    assert torch.equal(buffer[0], buffer[2])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
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
    allocation = torch.full((3, layout.byte_end), 254, dtype=torch.uint8, device=device)
    page = allocation[1]
    source.write_to_pd_buffer(0, allocation, pd_page_index=1, layout=layout)
    target = make_state(cls, device=device, mtp_step=1)
    target.read_from_pd_buffer(allocation, 2, pd_page_index=1, layout=layout)
    assert (target.conv_state[:, 2, :, :2] == 23).all()
    assert (target.ssm_state[:, 4] == 47).all()
    assert (allocation[0] == 254).all() and (allocation[2] == 254).all()
    assert (page[:16] == 254).all()
    assert target.mtp_state_index[2] == 0


@pytest.mark.parametrize("cls", [KDAState, GDNState])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
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
        source.conv_state[:, req, :, :source.config.conv_width].fill_(10 + req)
        source.ssm_state[:, req * (src_mtp_step + 1)].fill_(20 + req)
    src_conv = source.conv_state.clone()
    src_ssm = source.ssm_state.clone()

    target.copy_run_time_state_from_other_dp(
        src_req_indices=[2, 0], dst_req_indices=[0, 1], remote_dp_bounded_state=source,
    )

    for src_req, dst_req in ((2, 0), (0, 1)):
        assert (target.conv_state[:, dst_req, :, :2] == 10 + src_req).all()
        assert (target.conv_state[:, dst_req, :, 2:] == -1).all()
        dst_slot = dst_req * (dst_mtp_step + 1)
        assert (target.ssm_state[:, dst_slot] == 20 + src_req).all()
        assert (target.ssm_state[:, dst_slot + 1:dst_slot + dst_mtp_step + 1] == -2).all()
        assert target.mtp_state_index[dst_req] == 0
    assert (target.conv_state[:, 2] == -1).all()
    assert (target.ssm_state[:, 2 * (dst_mtp_step + 1):] == -2).all()
    assert target.mtp_state_index[2] == 1
    assert (target.big_pages_pool.conv_state == 91).all()
    assert (target.big_pages_pool.ssm_state == 92).all()
    assert torch.equal(source.conv_state, src_conv)
    assert torch.equal(source.ssm_state, src_ssm)


@pytest.mark.parametrize("cls", [KDAState, GDNState])
def test_dp_big_page_copy_preserves_runtime_and_other_pages(cls):
    source = make_state(cls)
    target = make_state(cls)
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

    target.copy_big_page_state_from_other_dp(
        src_page_ids=[2, 0], dst_page_ids=[0, 1], remote_dp_bounded_state=source,
    )

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
