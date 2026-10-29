"""P fills draft KV without speculative KDA scratch; D still verifies MTP."""

import dataclasses
from types import SimpleNamespace

import pytest
import torch

# Match model startup's import order before request managers import infer state.
import lightllm.common.kv_cache_mem_manager
from lightllm.common.req_manager import Glm5NextReqManager
from lightllm.common.state_cache_manager import Glm5NextCacheConfig, LinearAttCacheManager
from lightllm.models.glm5_next.triton_kernel.kpool import compress_pools
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.envs_utils import get_env_start_args, set_env_start_args, set_unique_server_name


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make_manager(monkeypatch, role, step, mode):
    monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_NODE", "0")
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    monkeypatch.setattr("lightllm.common.req_manager.req_sampling_params.get_vocab_size", lambda _: 128)
    args = StartArgs(
        run_mode=role,
        mtp_mode="eagle_with_att",
        mtp_step=step,
        linear_att_hash_page_size=4,
        linear_att_page_block_num=2,
        ssm_state_mode=mode,
        ssm_replay_config={"cache_len": 8, "compute_mode": "precompute"},
    )
    set_unique_server_name(args)
    get_env_start_args.cache_clear()
    set_env_start_args(dataclasses.asdict(args))
    config = Glm5NextCacheConfig(
        tp_world_size=1,
        full_att_all_num_kv_heads=1,
        full_att_dtype=torch.uint8,
        full_att_num_kv_heads=1,
        full_att_head_dim=720,
        global_linear_k_heads=2,
        global_linear_v_heads=2,
        num_linear_k_heads=2,
        num_linear_v_heads=2,
        head_linear_k_dim=128,
        head_linear_v_dim=128,
        conv_kernel_size=4,
        linear_layer_num=6,
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=torch.bfloat16,
        full_attention_interval=4,
        all_layer_num=8,
        draft_full_att_kv_layer_num=1,
    )
    return Glm5NextReqManager(3, 32, None, config)


@pytest.mark.parametrize("p_step,d_step", [(1, 3), (3, 3), (1, 1), (1, 7)])
@pytest.mark.parametrize("remainder", range(4))
def test_native_prefill_to_replay_decode_preserves_state_and_next_pool(monkeypatch, p_step, d_step, remainder):
    source = make_manager(monkeypatch, "prefill", p_step, "native")
    dest = make_manager(monkeypatch, "decode", d_step, "replay")
    p, d = source.linear_state_manager, dest.linear_state_manager
    assert source.mtp_step == p_step  # Keep the model/draft policy unchanged.
    assert p.mtp_step == 0 and p.ssm_slots_per_req == 1
    assert p.accepted_state_index is None and p.ssm_backend is None
    assert p.conv_state.buffer.shape[-1] == 3
    assert d.mtp_step == d_step and d.ssm_backend.verify_width == d_step + 1
    assert source.req_to_indexer_tail.buffer.shape[2] == 19
    assert source.get_pd_state_size() == dest.get_pd_state_size()
    torch.manual_seed(29)
    src, dst = 0, 2
    p.conv_state.buffer.normal_()
    p.ssm_state.buffer.normal_()
    source.req_to_indexer_tail.buffer.normal_()
    d.conv_state.buffer.normal_()
    d.ssm_state.buffer.normal_()
    dest.req_to_indexer_tail.buffer.normal_()
    d.accepted_state_index[dst] = d_step
    d.ssm_backend.cursors[dst] = 5
    untouched = d.ssm_state.buffer[:, 1].clone()
    wire = torch.zeros(source.get_pd_state_size(), dtype=torch.uint8, device="cuda")
    source.get_pd_state(src, wire, 0)
    dest.load_pd_state(dst, wire, 0)
    expected, actual = p.get_accepted_state(src), d.get_accepted_state(dst)
    torch.testing.assert_close(actual.conv, expected.conv, rtol=0, atol=0)
    torch.testing.assert_close(actual.ssm, expected.ssm, rtol=0, atol=0)
    torch.testing.assert_close(
        dest.req_to_indexer_tail.buffer[:, dst], source.req_to_indexer_tail.buffer[:, src], rtol=0, atol=0
    )
    torch.testing.assert_close(d.ssm_state.buffer[:, 1], untouched, rtol=0, atol=0)
    assert d.accepted_state_index[dst] == 0 and d.ssm_backend.cursors[dst] == 0

    # Native P has no acceptance array. Exercise both checkpoint save paths and
    # restore a fresh slot; aligned CPU cache prefixes intentionally reset tails.
    cache = LinearAttCacheManager(size=2, linear_config=source.linear_config)
    source.mem_manager = SimpleNamespace(big_page_buffers=cache)
    source.save_state(src, 0, cache)
    source.save_big_page_states(torch.tensor([src], device="cuda", dtype=torch.int32), [src], [1])
    torch.cuda.synchronize()
    for slot in [0, 1]:
        source.restore_state(SimpleNamespace(req_idx=1), cache, slot)
        restored = p.get_accepted_state(1)
        torch.testing.assert_close(restored.conv, expected.conv, rtol=0, atol=0)
        torch.testing.assert_close(restored.ssm, expected.ssm, rtol=0, atol=0)
        assert source.req_to_indexer_tail.buffer[:, 1].count_nonzero() == 0

    # Cross the next compression boundary using different request slots.
    # Absolute modulo addressing must agree even when P/D mtp_step differ.
    length = 8 + remainder
    count = 4 - remainder
    raw = torch.randn(count, 256, dtype=torch.bfloat16, device="cuda")
    ape = torch.randn(4, 128, device="cuda")
    tensor = lambda values: torch.tensor(values, device="cuda", dtype=torch.int32)
    outputs = []
    for manager, req_idx in [(source, src), (dest, dst)]:
        packed = torch.zeros(32, 1, 132, dtype=torch.uint8, device="cuda")
        compress_pools(
            raw=raw,
            tail=manager.get_indexer_tail_buffer(3),
            packed_buffer=packed,
            ape=ape,
            lengths=tensor(list(range(length + 1, length + count + 1))),
            starts=tensor([0] * count),
            ragged=tensor(list(range(32))),
            req_idx=tensor([req_idx]),
            cu_q_lens=tensor([0, count]),
            seq_lens=tensor([length + count]),
            max_q_len=count,
        )
        outputs.append(packed)
    torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)


def test_standalone_native_mtp_keeps_speculative_slots(monkeypatch):
    manager = make_manager(monkeypatch, "normal", 3, "native")
    assert manager.linear_state_manager.mtp_step == 3
    assert manager.linear_state_manager.ssm_slots_per_req == 4
    assert manager.req_to_indexer_tail.buffer.shape[2] == 7
