from types import SimpleNamespace

import pytest
import torch

from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import Qwen3NextLinearAttPageHelper
from lightllm.common.linear_state import LinearStateManager
from lightllm.common.req_manager.linear_att import ReqManagerForMamba
from lightllm.common.req_manager.linear_att_pd_state import LinearAttPDState
from lightllm.common.state_cache_manager import LinearAttCacheConfig, LinearAttCacheManager


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode,step", [("native", 0), ("native", 2), ("compact", 2), ("replay", 0), ("replay", 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_checkpoint_and_pd_export_same_accepted_prefix_and_restore_reused_slot(mode, step, dtype):
    config = LinearAttCacheConfig(
        tp_world_size=1,
        full_att_all_num_kv_heads=1,
        full_att_dtype=torch.bfloat16,
        full_att_num_kv_heads=1,
        full_att_head_dim=32,
        global_linear_k_heads=1,
        global_linear_v_heads=2,
        num_linear_k_heads=1,
        num_linear_v_heads=2,
        head_linear_k_dim=32,
        head_linear_v_dim=32,
        conv_kernel_size=4,
        linear_layer_num=2,
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=dtype,
        full_attention_interval=3,
        all_layer_num=3,
    )
    args = SimpleNamespace(mtp_step=step, ssm_state_mode=mode, ssm_replay_config={"cache_len": 4})
    state = LinearStateManager(3, config, args)
    manager = object.__new__(ReqManagerForMamba)
    manager.linear_state_manager = state
    manager.pd_state = LinearAttPDState(state)
    cache = LinearAttCacheManager(3, config)
    manager.mem_manager = SimpleNamespace(big_page_buffers=cache)
    mem = SimpleNamespace(linear_config=config, req_manager=manager)
    helper = Qwen3NextLinearAttPageHelper(mem)
    src, dst, offset = 1, 2, min(1, step)
    width = config.get_mtp_conv_state_shape(step)[-1]
    state.conv_state.buffer.copy_(torch.arange(width, device="cuda").to(torch.bfloat16))
    for slot in range(state.ssm_state.buffer.shape[1]):
        state.ssm_state.buffer[:, slot].fill_(slot + 1)
    if state.accepted_state_index is not None:
        state.accepted_state_index[src] = offset
    expected_conv = state.conv_state.buffer[:, src, ..., offset : offset + 3].clone()
    expected_ssm = state.ssm_state.buffer[
        :, src * state.ssm_slots_per_req + (offset if mode == "native" else 0)
    ].clone()
    if mode == "replay":
        # One pending accepted update: beta=0 leaves only decay of the base.
        state.ssm_backend.raw_keys.zero_()
        state.ssm_backend.raw_values.zero_()
        state.ssm_backend.gates.fill_(-0.5)
        state.ssm_backend.betas.zero_()
        state.ssm_backend.cursors[src] = 1
        expected_ssm = (expected_ssm.float() * torch.exp(torch.tensor(-0.5))).to(dtype)
    before = state.ssm_state.buffer.clone()
    state_block = torch.empty(manager.get_pd_state_size(), dtype=torch.uint8, device="cuda")
    manager.get_pd_state(src, state_block, tp_rank=0)
    pd_conv, pd_ssm = manager.pd_state._view_block_to_linear_att_state(state_block)
    torch.testing.assert_close(pd_conv, expected_conv, rtol=0, atol=0)
    torch.testing.assert_close(pd_ssm, expected_ssm, rtol=2e-5, atol=2e-5)
    expected_ssm = pd_ssm.clone()

    manager.save_state(src, 0, cache)
    manager.save_big_page_states(torch.tensor([src], dtype=torch.int32, device="cuda"), [src], [1])
    mem.kv_move_buffer = torch.empty_like(state_block).unsqueeze(0)
    helper.write_req_to_page(0, src, [mem])
    conv_page, ssm_page = manager.pd_state._view_block_to_linear_att_state(mem.kv_move_buffer[0])
    torch.cuda.synchronize()
    for index in (0, 1):
        conv, ssm = cache.get_state_cache(index)
        torch.testing.assert_close(conv, expected_conv.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(ssm, expected_ssm.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(conv_page, expected_conv, rtol=0, atol=0)
    torch.testing.assert_close(ssm_page, expected_ssm, rtol=0, atol=0)
    assert torch.equal(state.ssm_state.buffer, before)
    if mode == "replay":
        assert state.ssm_backend.cursors[src].item() == 1

    def dirty_destination():
        if state.accepted_state_index is not None:
            state.accepted_state_index[dst] = step
        if mode == "replay":
            state.ssm_backend.cursors[dst] = 3

    for restore in (
        lambda: manager.restore_state(SimpleNamespace(req_idx=dst), cache, 0),
        lambda: helper.read_page_to_req(0, dst, [mem]),
        lambda: manager.load_pd_state(dst, state_block, tp_rank=0),
    ):
        dirty_destination()
        restore()
        actual = state.get_accepted_state(dst)
        torch.testing.assert_close(actual.conv, expected_conv, rtol=0, atol=0)
        torch.testing.assert_close(actual.ssm, expected_ssm, rtol=0, atol=0)
        if state.accepted_state_index is not None:
            assert state.accepted_state_index[dst].item() == 0
        if mode == "replay":
            assert state.ssm_backend.cursors[dst].item() == 0

    # A skipped checkpoint must not overwrite another cache owner.
    untouched = tuple(x.clone() for x in cache.get_state_cache(2))
    manager.save_big_page_states(torch.tensor([src], dtype=torch.int32, device="cuda"), [src], [-1])
    torch.cuda.synchronize()
    for actual, expected in zip(cache.get_state_cache(2), untouched):
        assert torch.equal(actual, expected)
    state.reset(dst)
    assert not state.conv_state.buffer[:, dst].any()
    start = dst * state.ssm_slots_per_req
    assert not state.ssm_state.buffer[:, start : start + state.ssm_slots_per_req].any()


def test_pd_block_roundtrip_preserves_replicated_heads_and_accepted_mtp_state():
    config = LinearAttCacheConfig(
        tp_world_size=4,
        full_att_all_num_kv_heads=1,
        full_att_dtype=torch.float32,
        full_att_num_kv_heads=1,
        full_att_head_dim=2,
        global_linear_k_heads=1,
        global_linear_v_heads=2,
        num_linear_k_heads=1,
        num_linear_v_heads=1,
        head_linear_k_dim=2,
        head_linear_v_dim=3,
        conv_kernel_size=4,
        linear_layer_num=1,
        conv_state_dtype=torch.float32,
        ssm_state_dtype=torch.float32,
        full_attention_interval=2,
        all_layer_num=2,
    )
    args = SimpleNamespace(mtp_step=2, ssm_state_mode="native")
    managers, snapshots = [], []
    src, dst = 1, 2
    for rank in range(config.tp_world_size):
        state = LinearStateManager(3, config, args, device="cpu")
        state.conv_state.buffer.fill_(-1)
        state.ssm_state.buffer.fill_(-2)
        state.accepted_state_index.fill_(1)
        state.conv_state.buffer[:, src, :2, 1:4].fill_(10 + rank)
        state.conv_state.buffer[:, src, 2:4, 1:4].fill_(20 + rank)
        state.conv_state.buffer[:, src, 4:, 1:4].fill_(30 + rank)
        state.ssm_state.buffer[:, src * state.ssm_slots_per_req + 1].fill_(40 + rank)
        manager = object.__new__(ReqManagerForMamba)
        manager.linear_state_manager = state
        manager.pd_state = LinearAttPDState(state)
        managers.append(manager)
        snapshots.append((state.conv_state.buffer.clone(), state.ssm_state.buffer.clone()))

    state_nbytes = managers[0].get_pd_state_size()
    block = torch.full((state_nbytes + 16,), 0xA5, dtype=torch.uint8)
    for rank, manager in enumerate(managers):
        manager.get_pd_state(src, block, rank)

    # Q/K have four replicas; V/SSM have two. Only ranks 0 and 2 own exports.
    expected_conv = torch.cat(
        [torch.full((1, channels, 3), value) for channels, value in ((2, 10), (2, 20), (3, 30), (3, 32))],
        dim=1,
    ).float()
    expected_ssm = torch.stack((torch.full((1, 2, 3), 40), torch.full((1, 2, 3), 42)), dim=1).float()
    conv_nbytes = expected_conv.numel() * expected_conv.element_size()
    ssm_nbytes = expected_ssm.numel() * expected_ssm.element_size()
    assert state_nbytes == conv_nbytes + ssm_nbytes
    torch.testing.assert_close(block[:conv_nbytes].view(torch.float32).view_as(expected_conv), expected_conv)
    torch.testing.assert_close(block[conv_nbytes:state_nbytes].view(torch.float32).view_as(expected_ssm), expected_ssm)
    assert torch.all(block[state_nbytes:] == 0xA5)

    exported = block.clone()
    other_requests = torch.tensor([0, 1, 3])
    for rank, (manager, (conv_before, ssm_before)) in enumerate(zip(managers, snapshots)):
        state = manager.linear_state_manager
        torch.testing.assert_close(state.conv_state.buffer, conv_before)
        torch.testing.assert_close(state.ssm_state.buffer, ssm_before)
        manager.load_pd_state(dst, block, rank)
        expected_local_conv = torch.cat(
            (expected_conv[:, :4], expected_conv[:, 4 + rank // 2 * 3 : 7 + rank // 2 * 3]), dim=1
        )
        torch.testing.assert_close(state.conv_state.buffer[:, dst, ..., :3], expected_local_conv)
        torch.testing.assert_close(
            state.ssm_state.buffer[:, dst * state.ssm_slots_per_req], expected_ssm[:, rank // 2 : rank // 2 + 1]
        )
        assert state.accepted_state_index.tolist() == [1, 1, 0, 1]
        torch.testing.assert_close(state.conv_state.buffer[:, other_requests], conv_before[:, other_requests])
        other_slots = (
            other_requests[:, None] * state.ssm_slots_per_req + torch.arange(state.ssm_slots_per_req)
        ).flatten()
        torch.testing.assert_close(state.ssm_state.buffer[:, other_slots], ssm_before[:, other_slots])
    assert torch.equal(block, exported)
