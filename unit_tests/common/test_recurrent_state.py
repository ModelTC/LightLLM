from types import SimpleNamespace

import pytest
import torch

from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import Qwen3NextLinearAttPageHelper
from lightllm.common.recurrent_state import RecurrentState
from lightllm.common.req_manager.linear_att import ReqManagerForMamba
from lightllm.common.state_cache_manager import LinearAttCacheConfig, LinearAttCacheManager


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


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
    state = RecurrentState(3, config, args)
    manager = object.__new__(ReqManagerForMamba)
    manager.recurrent_state = state
    cache = LinearAttCacheManager(3, config)
    manager.mem_manager = SimpleNamespace(big_page_buffers=cache)
    mem = SimpleNamespace(linear_config=config, recurrent_state=state)
    helper = Qwen3NextLinearAttPageHelper(mem)
    src, dst, offset = 1, 2, min(1, step)
    width = config.get_mtp_conv_state_shape(step)[-1]
    state.conv_state.buffer.copy_(torch.arange(width, device="cuda").to(torch.bfloat16))
    for slot in range(state.ssm_state.buffer.shape[1]):
        state.ssm_state.buffer[:, slot].fill_(slot + 1)
    if state.accepted_index is not None:
        state.accepted_index[src] = offset
    expected_conv = state.conv_state.buffer[:, src, ..., offset : offset + 3].clone()
    expected_ssm = state.ssm_state.buffer[:, src * state.slots_per_req + (offset if mode == "native" else 0)].clone()
    if mode == "replay":
        # One pending accepted update: beta=0 leaves only decay of the base.
        state.updates.raw_keys.zero_()
        state.updates.raw_values.zero_()
        state.updates.gates.fill_(-0.5)
        state.updates.betas.zero_()
        state.updates.cursors[src] = 1
        expected_ssm = (expected_ssm.float() * torch.exp(torch.tensor(-0.5))).to(dtype)
    before = state.ssm_state.buffer.clone()
    snapshot = state.snapshot(src)
    torch.testing.assert_close(snapshot.ssm, expected_ssm, rtol=2e-5, atol=2e-5)
    expected_ssm = snapshot.ssm.clone()

    manager.save_state(src, 0, cache)
    manager.save_big_page_states(torch.tensor([src], dtype=torch.int32, device="cuda"), [src], [1])
    conv_page = torch.empty(helper.conv_shape, device="cuda", dtype=torch.bfloat16)
    ssm_page = torch.empty(helper.ssm_shape, device="cuda", dtype=dtype)
    helper._write_one_rank(mem, 0, src, conv_page, ssm_page)
    torch.cuda.synchronize()
    for index in (0, 1):
        conv, ssm = cache.get_state_cache(index)
        torch.testing.assert_close(conv, expected_conv.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(ssm, expected_ssm.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(conv_page, expected_conv, rtol=0, atol=0)
    torch.testing.assert_close(ssm_page, expected_ssm, rtol=0, atol=0)
    assert torch.equal(state.ssm_state.buffer, before)
    if mode == "replay":
        assert state.updates.cursors[src].item() == 1

    def dirty_destination():
        if state.accepted_index is not None:
            state.accepted_index[dst] = step
        if mode == "replay":
            state.updates.cursors[dst] = 3

    for restore in (
        lambda: manager.restore_state(SimpleNamespace(req_idx=dst), cache, 0),
        lambda: helper._read_one_rank(mem, 0, dst, conv_page, ssm_page),
    ):
        dirty_destination()
        restore()
        actual = state.snapshot(dst)
        torch.testing.assert_close(actual.conv, expected_conv, rtol=0, atol=0)
        torch.testing.assert_close(actual.ssm, expected_ssm, rtol=0, atol=0)
        if state.accepted_index is not None:
            assert state.accepted_index[dst].item() == 0
        if mode == "replay":
            assert state.updates.cursors[dst].item() == 0

    # A skipped checkpoint must not overwrite another cache owner.
    untouched = tuple(x.clone() for x in cache.get_state_cache(2))
    manager.save_big_page_states(torch.tensor([src], dtype=torch.int32, device="cuda"), [src], [-1])
    torch.cuda.synchronize()
    for actual, expected in zip(cache.get_state_cache(2), untouched):
        assert torch.equal(actual, expected)
    state.reset(dst)
    assert not state.conv_state.buffer[:, dst].any()
    start = dst * state.slots_per_req
    assert not state.ssm_state.buffer[:, start : start + state.slots_per_req].any()
