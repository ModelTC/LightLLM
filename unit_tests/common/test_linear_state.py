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
    if mode == "replay":
        from lightllm.common.attention_state.bounded_att.gdn.replay_gdn_state import ReplayGDNState
        from lightllm.common.attention_state.bounded_att.kda.replay_kda_state import ReplayKDAState

        cls = ReplayKDAState if cls is KDAState else ReplayGDNState
        state = cls(config, 4, mtp_step=step, device=device, ssm_replay_config={"cache_len": 4})
    else:
        state = cls(config, 4, mtp_step=step, device=device)
    resources = AttentionStateManager(request_capacity=4)
    resources.register_state("main", state)
    resources.finish_state_registration()
    layout = resources.get_pd_operator().get_pd_bounded_att_page_layouts()[state.state_name]
    return state, resources, layout


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("cls", [GDNState, KDAState])
@pytest.mark.parametrize("mode,step", [("native", 0), ("native", 2), ("replay", 0), ("replay", 2)])
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
    if mode == "native":
        expected_conv = state.conv_state[:, src, ..., :3].clone()
        expected_ssm = state.ssm_state[:, src * state.ssm_slots_per_req].clone()
    else:
        expected_conv = state.conv_state[:, src, ..., offset : offset + 3].clone()
        expected_ssm = state.ssm_state[:, src].clone()
    if mode == "replay":
        state._ssm_executor.raw_keys.zero_()
        state._ssm_executor.raw_values.zero_()
        state._ssm_executor.gates.fill_(-0.5)
        state._ssm_executor.betas.zero_()
        state._ssm_executor.cursors[src] = 1
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
        assert state._ssm_executor.cursors[src].item() == 1

    for restore in (
        lambda: state.restore_bounded_att_runtime_from_pages([dst], [8], "small", [0]),
        lambda: state.restore_bounded_att_runtime_from_pages([dst], [8], "big", [0]),
        lambda: state.read_from_pd_page(block, dst, 0, 8, layout),
    ):
        if state.mtp_state_index is not None:
            state.mtp_state_index[dst] = step
        if mode == "replay":
            state._ssm_executor.cursors[dst] = 3
        restore()
        if mode == "replay":
            actual = state.get_accepted_state(dst)
            actual_conv, actual_ssm = actual.conv, actual.ssm
        else:
            actual_conv = state.conv_state[:, dst, ..., :3]
            actual_ssm = state.ssm_state[:, dst * state.ssm_slots_per_req]
        torch.testing.assert_close(actual_conv, expected_conv, rtol=0, atol=0)
        torch.testing.assert_close(actual_ssm, expected_ssm, rtol=2e-5, atol=2e-5)
        if state.mtp_state_index is not None:
            assert state.mtp_state_index[dst].item() == 0
        if mode == "replay":
            assert state._ssm_executor.cursors[dst].item() == 0
    state.reset_bounded_att_state([dst])
    assert not state.conv_state[:, dst].any()
    start = dst * state.ssm_slots_per_req
    assert not state.ssm_state[:, start : start + state.ssm_slots_per_req].any()


def copy_on_cpu(*, copies):
    for source, destination in copies:
        destination.copy_(source)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("cls", [GDNState, KDAState])
@pytest.mark.parametrize("mode", ["native", "replay"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("destination", ["cpu", "cuda"])
def test_checkpoint_batch_export(runtime_start_args, monkeypatch, cls, mode, dtype, destination):
    state, _, _ = make_state(cls, mode, 2, dtype)
    state.initialize_page_pools(3, device=destination)
    state.conv_state.copy_(torch.arange(5, device="cuda").to(torch.bfloat16))
    for slot in range(state.ssm_state.shape[1]):
        state.ssm_state[:, slot].fill_(slot + 1)
    state.mtp_state_index.copy_(torch.tensor([0, 2, 1, 0], device="cuda"))
    if mode == "replay":
        state._ssm_executor.raw_keys.zero_()
        state._ssm_executor.raw_values.zero_()
        state._ssm_executor.gates.fill_(-0.5)
        state._ssm_executor.betas.zero_()
        state._ssm_executor.cursors.copy_(torch.tensor([1, 2, 3, 0], device="cuda"))
    requests = [2, 0, 1]
    if mode == "replay":
        expected = [(item.conv.clone(), item.ssm.clone()) for item in map(state.get_accepted_state, requests)]
    else:
        expected = [
            (
                state.conv_state[:, req, :, : state.config.conv_width].clone(),
                state.ssm_state[:, req * state.ssm_slots_per_req].clone(),
            )
            for req in requests
        ]
    before = state.ssm_state.clone()
    cursors = state.runtime.cursors.clone() if mode == "replay" else None

    def no_snapshot(*args):
        raise AssertionError("batch export must write directly to the destination")

    if mode == "replay":
        monkeypatch.setattr(state, "get_accepted_state", no_snapshot)
        monkeypatch.setattr(state.runtime, "get_accepted_state", no_snapshot)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        state.save_bounded_att_runtime_to_pages(requests, [8] * 3, "big", [1, 2, 0])
    stream.synchronize()
    pool = state.big_pages_pool
    for slot, (conv, ssm) in zip([1, 2, 0], expected):
        torch.testing.assert_close(pool.conv_state[slot].cpu(), conv.cpu(), atol=0, rtol=0)
        torch.testing.assert_close(pool.ssm_state[slot].cpu(), ssm.cpu(), atol=0, rtol=0)
    assert torch.equal(state.ssm_state, before)
    if cursors is not None:
        assert torch.equal(state._ssm_executor.cursors, cursors)
    target, _, _ = make_state(cls, mode, 2, dtype)
    if target.mtp_state_index is not None:
        target.mtp_state_index.fill_(2)
    target.copy_runtime_state_from_dp(requests, [0, 1, 2], state)
    for req, (conv, ssm) in enumerate(expected):
        if mode == "replay":
            actual = target.get_accepted_state(req)
            actual_conv, actual_ssm = actual.conv, actual.ssm
        else:
            actual_conv = target.conv_state[:, req, :, : target.config.conv_width]
            actual_ssm = target.ssm_state[:, req * target.ssm_slots_per_req]
        torch.testing.assert_close(actual_conv, conv, atol=0, rtol=0)
        torch.testing.assert_close(actual_ssm, ssm, atol=0, rtol=0)
    assert torch.equal(state.ssm_state, before)
    if cursors is not None:
        assert torch.equal(state._ssm_executor.cursors, cursors)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_checkpoint_scratch_is_bounded(runtime_start_args):
    config = ConvSSMStateConfig(
        logical_layer_ids=(0, 1, 2, 3),
        global_k_heads=1,
        global_v_heads=2,
        key_dim=128,
        value_dim=128,
        conv_width=3,
        conv_dtype=torch.bfloat16,
        ssm_dtype=torch.float32,
    )
    state = GDNState(config, 65, mtp_step=2, device="cuda")
    state.initialize_page_pools(64)
    requests = list(range(64))
    state.save_bounded_att_runtime_to_pages(requests, [128] * 64, "big", requests)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    state.save_bounded_att_runtime_to_pages(requests, [128] * 64, "big", requests)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - before < 1024 * 1024


def test_pd_block_roundtrip_preserves_replicated_heads_and_accepted_mtp_state(monkeypatch):
    # Exercise layout and accepted-state selection independently of the CUDA copy kernel.
    monkeypatch.setattr(
        import_module("lightllm.common.attention_state.bounded_att.gdn.gdn_state"),
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
        state.conv_state[:, src, :2, :3].fill_(10 + rank)
        state.conv_state[:, src, 2:4, :3].fill_(20 + rank)
        state.conv_state[:, src, 4:, :3].fill_(30 + rank)
        state.ssm_state[:, src * state.ssm_slots_per_req].fill_(40 + rank)
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
