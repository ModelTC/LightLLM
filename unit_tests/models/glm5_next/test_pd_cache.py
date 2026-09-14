import dataclasses

import pytest
import torch

from lightllm.common.kv_cache_mem_manager import Glm5NextMemManager, MemoryManager
from lightllm.common.req_manager import Glm5NextReqManager
from lightllm.common.state_cache_manager import Glm5NextCacheConfig
from lightllm.models.glm5_next.triton_kernel.kpool import compress_pools
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.envs_utils import get_env_start_args, set_env_start_args, set_unique_server_name


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture
def make_mems(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_NODE", "0")
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    monkeypatch.setattr("lightllm.common.req_manager.req_sampling_params.get_vocab_size", lambda _: 128)
    args = StartArgs(data_type="bfloat16", linear_att_hash_page_size=4, linear_att_page_block_num=2)
    set_unique_server_name(args)
    get_env_start_args.cache_clear()
    set_env_start_args(dataclasses.asdict(args))

    def check_shm_refs(mem, req):
        assert mem.req_to_indexer_tail is req.req_to_indexer_tail
        assert mem.req_to_conv_state is req.req_to_conv_state
        assert mem.req_to_ssm_state is req.req_to_ssm_state
        assert mem.big_page_buffers is None

    monkeypatch.setattr(MemoryManager, "write_to_shm", check_shm_refs)

    def create(tp):
        config = Glm5NextCacheConfig(
            tp_world_size=tp,
            full_att_all_num_kv_heads=1,
            full_att_dtype=torch.bfloat16,
            full_att_num_kv_heads=1,
            full_att_head_dim=584,
            global_linear_k_heads=8,
            global_linear_v_heads=8,
            num_linear_k_heads=8 // tp,
            num_linear_v_heads=8 // tp,
            head_linear_k_dim=128,
            head_linear_v_dim=128,
            conv_kernel_size=4,
            linear_layer_num=6,
            conv_state_dtype=torch.bfloat16,
            ssm_state_dtype=torch.float32,
            full_attention_interval=4,
            all_layer_num=8,
        )
        mems = []
        for _ in range(tp):
            mem = Glm5NextMemManager(32, torch.bfloat16, 1, 584, 2, config)
            req = Glm5NextReqManager(3, 32, mem, config)
            mem.write_to_shm(req)
            assert mem.big_page_buffers is not None
            mems.append(mem)
        mems[0].alloc_paged_kv_move_buffer(1, 2048)
        return mems

    return create


@pytest.mark.parametrize("prefill_tp,decode_tp", [(1, 1), (4, 4), (1, 4), (4, 1)])
@pytest.mark.parametrize("remainder", range(4))
def test_pd_kv_and_runtime_state_roundtrip(make_mems, prefill_tp, decode_tp, remainder):
    source, dest = make_mems(prefill_tp), make_mems(decode_tp)
    src_req, dst_req = 0, 2
    length = 8 + remainder
    src_indexes = torch.randperm(32)[:length].tolist()
    dst_indexes = torch.randperm(32)[:length].tolist()
    packed = torch.randint(0, 256, (2, length, 1, 1168), dtype=torch.uint8, device="cuda")
    conv = torch.randn(6, 3, 8, 128, 3, dtype=torch.bfloat16, device="cuda")
    ssm = torch.randn(6, 8, 128, 128, dtype=torch.float32, device="cuda")
    # Non-live slots deliberately contain stale data. The sequence length, not
    # the bytes in these slots, controls subsequent K-pool completion.
    tail = torch.randn(2, 4, 256, dtype=torch.bfloat16, device="cuda")
    for rank, mem in enumerate(source):
        heads = slice(rank * 8 // prefill_tp, (rank + 1) * 8 // prefill_tp)
        mem.kv_buffer.view(torch.uint8)[:, src_indexes] = packed
        mem.req_to_conv_state.buffer[:, src_req].copy_(conv[:, :, heads].reshape(6, -1, 3))
        mem.req_to_ssm_state.buffer[:, src_req].copy_(ssm[:, heads])
        mem.req_to_indexer_tail.buffer[:, src_req].copy_(tail)

    untouched = []
    for mem in dest:
        mem.kv_buffer.zero_()
        mem.req_to_conv_state.buffer.normal_()
        mem.req_to_ssm_state.buffer.normal_()
        mem.req_to_indexer_tail.buffer.normal_()
        untouched.append(
            [
                x.buffer[:, [0, 1, 3]].clone()
                for x in (mem.req_to_conv_state, mem.req_to_ssm_state, mem.req_to_indexer_tail)
            ]
        )

    source[0].write_mem_to_page_kv_move_buffer(src_indexes, 0, 0, source, prefill_tp)
    # Model a byte-preserving transport between separate P/D page buffers.
    dest[0].kv_move_buffer.view(torch.uint8).copy_(source[0].kv_move_buffer.view(torch.uint8))
    dest[0].read_page_kv_move_buffer_to_mem(dst_indexes, 0, 0, dest, decode_tp)
    source[0].write_mem_to_page_kv_move_buffer([], 0, 0, source, prefill_tp, "att_state", src_req)
    dest[0].kv_move_buffer.view(torch.uint8).copy_(source[0].kv_move_buffer.view(torch.uint8))
    dest[0].read_page_kv_move_buffer_to_mem([], 0, 0, dest, decode_tp, "att_state", dst_req)
    torch.cuda.synchronize()

    for rank, mem in enumerate(dest):
        heads = slice(rank * 8 // decode_tp, (rank + 1) * 8 // decode_tp)
        assert torch.equal(mem.kv_buffer.view(torch.uint8)[:, dst_indexes], packed)
        assert torch.equal(mem.req_to_conv_state.buffer[:, dst_req], conv[:, :, heads].reshape(6, -1, 3))
        assert torch.equal(mem.req_to_ssm_state.buffer[:, dst_req], ssm[:, heads])
        assert torch.equal(mem.req_to_indexer_tail.buffer[:, dst_req], tail)
        for state, expected in zip(
            (mem.req_to_conv_state, mem.req_to_ssm_state, mem.req_to_indexer_tail), untouched[rank]
        ):
            assert torch.equal(state.buffer[:, [0, 1, 3]], expected)

    # Continue through the next pool boundary on P and on the restored D state.
    # Different physical KV/request slots must yield identical pooled FP8 bytes.
    count = 4 - remainder
    raw = torch.randn(count, 256, dtype=torch.bfloat16, device="cuda")
    ape = torch.randn(4, 128, device="cuda")
    tensor = lambda values: torch.tensor(values, dtype=torch.int32, device="cuda")
    outputs = []
    for mem, req_idx, indexes in ((source[0], src_req, src_indexes), (dest[0], dst_req, dst_indexes)):
        new_indexes = [i for i in range(32) if i not in indexes][:count]
        all_indexes = indexes + new_indexes
        compress_pools(
            raw=raw,
            tail=mem.req_to_indexer_tail.buffer[0],
            packed_buffer=mem.get_indexer_k_buffer(3),
            ape=ape,
            lengths=tensor(range(length + 1, length + count + 1)),
            starts=tensor([0] * count),
            ragged=tensor(all_indexes),
            req_idx=tensor([req_idx]),
            cu_q_lens=tensor([0, count]),
            seq_lens=tensor([length + count]),
            max_q_len=count,
        )
        outputs.append(mem.get_indexer_k_buffer(3)[new_indexes[-1]].clone())
    assert torch.equal(*outputs)


def test_pd_page_capacity_includes_tail_without_resizing(make_mems):
    mem = make_mems(1)[0]
    helper = mem.att_state_page_helper
    # A page that fits Conv+SSM but not the appended tail must be rejected.
    mem.kv_move_buffer = torch.empty((1, helper.tail_offset), dtype=torch.uint8, device="cuda")
    with pytest.raises(AssertionError, match="smaller than global linear att state"):
        helper.assert_page_size()
    shape = mem.get_paged_kv_move_buffer_shape(2, 2048)
    assert shape == (2, 2048, 2, 1, 584)
