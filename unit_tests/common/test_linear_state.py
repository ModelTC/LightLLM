from dataclasses import replace
from importlib import import_module

import pytest
import torch

from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.bounded_att import ConvSSMStateConfig, GDNState, KDAState


def make_state(cls=GDNState, mode="native", step=2, dtype=torch.float32, device="cuda", config=None):
    config = config or ConvSSMStateConfig(
        logical_layer_ids=(0, 1),
        global_k_heads=1,
        global_v_heads=1 if cls is KDAState else 2,
        key_dim=32,
        value_dim=32,
        conv_width=3,
        conv_dtype=torch.bfloat16,
        ssm_dtype=dtype,
    )
    state = cls(config, 4, mtp_step=step, device=device, ssm_state_mode=mode, ssm_replay_config={"cache_len": 4})
    resources = AttentionStateManager(request_capacity=4)
    resources.register_state("main", state)
    resources.finish_state_registration()
    layout = resources.get_pd_operator().get_pd_bounded_att_page_layouts()[state.state_name]
    return state, resources, layout


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("cls", [GDNState, KDAState])
@pytest.mark.parametrize("mode,step", [("native", 0), ("native", 2), ("compact", 2), ("replay", 0), ("replay", 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_checkpoint_and_pd_export_same_accepted_prefix_and_restore_reused_slot(
    runtime_start_args, cls, mode, step, dtype
):
    state, _, layout = make_state(cls, mode, step, dtype)
    state.initialize_page_pools(big_page_capacity=2, small_page_capacity=2)
    src, dst, offset = 1, 2, min(1, step)
    state.conv_state.copy_(torch.arange(3 + step, device="cuda").to(torch.bfloat16))
    for slot in range(state.ssm_state.shape[1]):
        state.ssm_state[:, slot].fill_(slot + 1)
    if state.mtp_state_index is not None:
        state.mtp_state_index[src] = offset
    expected_conv = state.conv_state[:, src, ..., offset : offset + 3].clone()
    expected_ssm = state.ssm_state[:, src * state.ssm_slots_per_req + (offset if mode == "native" else 0)].clone()
    if mode == "replay":
        state._ssm.raw_keys.zero_()
        state._ssm.raw_values.zero_()
        state._ssm.gates.fill_(-0.5)
        state._ssm.betas.zero_()
        state._ssm.cursors[src] = 1
        expected_ssm = (expected_ssm.float() * torch.exp(torch.tensor(-0.5))).to(dtype)
    before = state.ssm_state.clone()
    block = torch.empty((1, layout.byte_end), dtype=torch.uint8, device="cuda")
    state.write_to_pd_page(src, block, 0, 8, layout)
    for kind in ("small", "big"):
        state.save_bounded_att_runtime_to_pages([src], [8], kind, [0])
    torch.cuda.synchronize()
    for pool in (state.small_pages_pool, state.big_pages_pool):
        torch.testing.assert_close(pool.conv_state[0], expected_conv.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(pool.ssm_state[0], expected_ssm.cpu(), rtol=2e-5, atol=2e-5)
    assert torch.equal(state.ssm_state, before)
    if mode == "replay":
        assert state._ssm.cursors[src].item() == 1

    for restore in (
        lambda: state.restore_bounded_att_runtime_from_pages([dst], [8], "small", [0]),
        lambda: state.restore_bounded_att_runtime_from_pages([dst], [8], "big", [0]),
        lambda: state.read_from_pd_page(block, dst, 0, 8, layout),
    ):
        if state.mtp_state_index is not None:
            state.mtp_state_index[dst] = step
        if mode == "replay":
            state._ssm.cursors[dst] = 3
        restore()
        actual = state.get_accepted_state(dst)
        torch.testing.assert_close(actual.conv, expected_conv, rtol=0, atol=0)
        torch.testing.assert_close(actual.ssm, expected_ssm, rtol=2e-5, atol=2e-5)
        if state.mtp_state_index is not None:
            assert state.mtp_state_index[dst].item() == 0
        if mode == "replay":
            assert state._ssm.cursors[dst].item() == 0
    state.reset_bounded_att_state([dst])
    assert not state.conv_state[:, dst].any()
    start = dst * state.ssm_slots_per_req
    assert not state.ssm_state[:, start : start + state.ssm_slots_per_req].any()


def copy_on_cpu(*, copies):
    for source, destination in copies:
        destination.copy_(source)


def test_pd_block_roundtrip_preserves_replicated_heads_and_accepted_mtp_state(monkeypatch):
    # Exercise layout and accepted-state selection independently of the CUDA copy kernel.
    monkeypatch.setattr(
        import_module("lightllm.common.attention_state.bounded_att.conv_ssm_state"),
        "copy_bounded_att_tensors",
        copy_on_cpu,
    )
    config = ConvSSMStateConfig(
        logical_layer_ids=(0,),
        global_k_heads=1,
        global_v_heads=2,
        key_dim=2,
        value_dim=3,
        conv_width=3,
        conv_dtype=torch.float32,
        ssm_dtype=torch.float32,
        tp_world_size=4,
    )
    states, layouts, snapshots = [], [], []
    src, dst = 1, 2
    for rank in range(4):
        state, _, layout = make_state(device="cpu", config=replace(config, tp_rank=rank))
        state.conv_state.fill_(-1)
        state.ssm_state.fill_(-2)
        state.mtp_state_index.fill_(1)
        state.conv_state[:, src, :2, 1:4].fill_(10 + rank)
        state.conv_state[:, src, 2:4, 1:4].fill_(20 + rank)
        state.conv_state[:, src, 4:, 1:4].fill_(30 + rank)
        state.ssm_state[:, src * state.ssm_slots_per_req + 1].fill_(40 + rank)
        states.append(state)
        layouts.append(layout)
        snapshots.append((state.conv_state.clone(), state.ssm_state.clone()))
    size = layouts[0].byte_end
    block = torch.full((1, size + 16), 0xA5, dtype=torch.uint8)
    for state, layout in zip(states, layouts):
        state.write_to_pd_page(src, block, 0, 8, layout)
    expected_conv = torch.cat(
        [torch.full((1, channels, 3), value) for channels, value in ((2, 10), (2, 20), (3, 30), (3, 32))],
        dim=1,
    ).float()
    expected_ssm = torch.stack((torch.full((1, 2, 3), 40), torch.full((1, 2, 3), 42)), dim=1).float()
    conv_bytes = expected_conv.numel() * 4
    torch.testing.assert_close(block[0, :conv_bytes].view(torch.float32).view_as(expected_conv), expected_conv)
    torch.testing.assert_close(block[0, conv_bytes:size].view(torch.float32).view_as(expected_ssm), expected_ssm)
    assert torch.all(block[0, size:] == 0xA5)
    exported = block.clone()
    other_requests = torch.tensor([0, 1, 3])
    for rank, (state, layout, (conv_before, ssm_before)) in enumerate(zip(states, layouts, snapshots)):
        torch.testing.assert_close(state.conv_state, conv_before)
        torch.testing.assert_close(state.ssm_state, ssm_before)
        state.read_from_pd_page(block, dst, 0, 8, layout)
        expected_local_conv = torch.cat(
            (expected_conv[:, :4], expected_conv[:, 4 + rank // 2 * 3 : 7 + rank // 2 * 3]), dim=1
        )
        torch.testing.assert_close(state.conv_state[:, dst, ..., :3], expected_local_conv)
        torch.testing.assert_close(state.ssm_state[:, dst * 3], expected_ssm[:, rank // 2 : rank // 2 + 1])
        assert state.mtp_state_index.tolist() == [1, 1, 0, 1]
        torch.testing.assert_close(state.conv_state[:, other_requests], conv_before[:, other_requests])
        other_slots = (other_requests[:, None] * 3 + torch.arange(3)).flatten()
        torch.testing.assert_close(state.ssm_state[:, other_slots], ssm_before[:, other_slots])
    assert torch.equal(block, exported)
