from lightllm.common.kv_page_transfer import KVPageTransfer
from lightllm.common.kv_transfer_resources import KVTransferView
import dataclasses

import pytest
import torch

from lightllm.common.kv_cache_mem_manager import Glm5NextMemManager, MemoryManager
from lightllm.common.cache_coordinator import HybridCacheCoordinator
from lightllm.common.linear_state import LinearStateManager
from lightllm.common.indexer_state import IndexerStateManager
from lightllm.common.state_cache_manager import Glm5NextCacheConfig
from lightllm.models.glm5_next.triton_kernel.kpool import compress_pools
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.envs_utils import get_env_start_args, set_env_start_args, set_unique_server_name


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(params=[0, 2])
def make_mems(monkeypatch, request):
    monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_NODE", "0")
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    monkeypatch.setattr("lightllm.common.req_manager.req_sampling_params.get_vocab_size", lambda _: 128)
    mtp_step = request.param
    args = StartArgs(
        data_type="bfloat16",
        linear_att_hash_page_size=4,
        linear_att_page_block_num=2,
        mtp_step=mtp_step,
        mtp_mode="eagle_with_att" if mtp_step else None,
    )
    set_unique_server_name(args)
    get_env_start_args.cache_clear()
    set_env_start_args(dataclasses.asdict(args))

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
            draft_full_att_kv_layer_num=int(mtp_step > 0),
        )
        mems = []
        states = []
        for _ in range(tp):
            mem = Glm5NextMemManager(32, torch.bfloat16, 1, 584, 2 + int(mtp_step > 0))
            mem.set_kv_layer_mapping(config.get_full_att_kv_layer_mapping())
            coordinator = HybridCacheCoordinator(
                LinearStateManager(3, config, args), IndexerStateManager(3, config, mtp_step)
            )
            coordinator.bind_mem_manager(mem)
            assert not hasattr(mem, "req_to_conv_state")
            assert not hasattr(mem, "req_to_ssm_state")
            assert not hasattr(mem, "req_to_indexer_tail")
            assert not hasattr(mem, "big_page_buffers")
            states.append(coordinator.create_state_transfer())
            mems.append(mem)
        transfer = KVPageTransfer(KVTransferView.from_mem_manager(mems[0], include_page_layout=True))
        transfer.alloc_paged_kv_move_buffer(1, 2048)
        states[0].bind_page_buffer(transfer.kv_move_buffer)
        return mems, states, transfer

    return create


@pytest.mark.parametrize("prefill_tp,decode_tp", [(1, 1), (4, 4), (1, 4), (4, 1)])
@pytest.mark.parametrize("remainder", range(4))
def test_pd_kv_and_runtime_state_roundtrip(make_mems, prefill_tp, decode_tp, remainder):
    (source, source_states, source_transfer), (dest, dest_states, dest_transfer) = make_mems(prefill_tp), make_mems(
        decode_tp
    )
    mtp_size = get_env_start_args().mtp_step + 1
    layers = source[0].layer_num
    src_req, dst_req = 0, 2
    length = 8 + remainder
    src_indexes = torch.randperm(32)[:length].tolist()
    dst_indexes = torch.randperm(32)[:length].tolist()
    packed = torch.randint(0, 256, (layers, length, 1, 1168), dtype=torch.uint8, device="cuda")
    conv = torch.randn(6, 3, 8, 128, 3, dtype=torch.bfloat16, device="cuda")
    ssm = torch.randn(6, 8, 128, 128, dtype=torch.float32, device="cuda")
    # Non-live slots deliberately contain stale data. The sequence length, not
    # the bytes in these slots, controls subsequent K-pool completion.
    tail = torch.randn(layers, 3 + mtp_size, 256, dtype=torch.bfloat16, device="cuda")
    for rank, (mem, state) in enumerate(zip(source, source_states)):
        heads = slice(rank * 8 // prefill_tp, (rank + 1) * 8 // prefill_tp)
        mem.kv_buffer.view(torch.uint8)[:, src_indexes] = packed
        state.req_to_conv_state.buffer[:, src_req, ..., :3].copy_(conv[:, :, heads].reshape(6, -1, 3))
        state.req_to_ssm_state.buffer[:, src_req * mtp_size].copy_(ssm[:, heads])
        state.req_to_indexer_tail.buffer[:, src_req].copy_(tail)

    untouched = []
    for mem, state in zip(dest, dest_states):
        mem.kv_buffer.zero_()
        state.req_to_conv_state.buffer.normal_()
        state.req_to_ssm_state.buffer.normal_()
        state.req_to_indexer_tail.buffer.normal_()
        untouched.append(
            [
                x.buffer[:, torch.tensor([0, 1, 3], device="cuda") * stride].clone()
                for x, stride in (
                    (state.req_to_conv_state, 1),
                    (state.req_to_ssm_state, mtp_size),
                    (state.req_to_indexer_tail, 1),
                )
            ]
        )

    source_transfer.write_mem_to_page_kv_move_buffer(src_indexes, 0, 0, source, prefill_tp)
    # Model a byte-preserving transport between separate P/D page buffers.
    dest_transfer.kv_move_buffer.view(torch.uint8).copy_(source_transfer.kv_move_buffer.view(torch.uint8))
    dest_transfer.read_page_kv_move_buffer_to_mem(dst_indexes, 0, 0, dest, decode_tp)
    source_states[0].write_req_to_page(0, src_req, source_states[0].get_dp_states(source_states, 0, prefill_tp))
    dest_transfer.kv_move_buffer.view(torch.uint8).copy_(source_transfer.kv_move_buffer.view(torch.uint8))
    dest_states[0].read_page_to_req(0, dst_req, dest_states[0].get_dp_states(dest_states, 0, decode_tp))
    torch.cuda.synchronize()

    for rank, (mem, state) in enumerate(zip(dest, dest_states)):
        heads = slice(rank * 8 // decode_tp, (rank + 1) * 8 // decode_tp)
        assert torch.equal(mem.kv_buffer.view(torch.uint8)[:, dst_indexes], packed)
        assert torch.equal(state.req_to_conv_state.buffer[:, dst_req, ..., :3], conv[:, :, heads].reshape(6, -1, 3))
        assert torch.equal(state.req_to_ssm_state.buffer[:, dst_req * mtp_size], ssm[:, heads])
        assert torch.equal(state.req_to_indexer_tail.buffer[:, dst_req], tail)
        for (state, stride), expected in zip(
            ((state.req_to_conv_state, 1), (state.req_to_ssm_state, mtp_size), (state.req_to_indexer_tail, 1)),
            untouched[rank],
        ):
            indexes = torch.tensor([0, 1, 3], device="cuda") * stride
            assert torch.equal(state.buffer[:, indexes], expected)

    # Continue through the next pool boundary on P and on the restored D state.
    # Different physical KV/request slots must yield identical pooled FP8 bytes.
    count = 4 - remainder
    raw = torch.randn(count, 256, dtype=torch.bfloat16, device="cuda")
    ape = torch.randn(4, 128, device="cuda")
    tensor = lambda values: torch.tensor(values, dtype=torch.int32, device="cuda")
    outputs = []
    for mem, state, req_idx, indexes in (
        (source[0], source_states[0], src_req, src_indexes),
        (dest[0], dest_states[0], dst_req, dst_indexes),
    ):
        new_indexes = [i for i in range(32) if i not in indexes][:count]
        all_indexes = indexes + new_indexes
        compress_pools(
            raw=raw,
            tail=state.req_to_indexer_tail.buffer[0],
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
    mems, states, transfer = make_mems(1)
    mem, helper = mems[0], states[0]
    # A page that fits Conv+SSM but not the appended tail must be rejected.
    helper.kv_move_buffer = torch.empty((1, helper.tail_offset), dtype=torch.uint8, device="cuda")
    with pytest.raises(AssertionError, match="smaller than global linear att state"):
        helper.assert_page_size()
    shape = mem.get_paged_kv_move_buffer_shape(2, 2048)
    assert shape == (2, 2048, mem.layer_num, 1, 584)
