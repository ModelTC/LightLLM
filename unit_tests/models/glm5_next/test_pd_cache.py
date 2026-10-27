import dataclasses

import pytest
import torch

from lightllm.common.kv_cache_mem_manager import (
    Glm5NextMemManager,
    FP8Glm5NextMemManager,
    MemoryManager,
)
from lightllm.common.req_manager import Glm5NextReqManager
from lightllm.common.state_cache_manager import Glm5NextCacheConfig
from lightllm.models.glm5_next.triton_kernel.kpool import compress_pools
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.envs_utils import (
    get_env_start_args,
    set_env_start_args,
    set_unique_server_name,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(
    params=[
        (step, mode, fp8)
        for step, mode in [
            (0, "native"),
            (2, "native"),
            (0, "replay"),
            (2, "replay"),
            (3, "replay"),
            (2, "compact"),
        ]
        for fp8 in [False, True]
    ]
)
def make_mems(monkeypatch, request):
    monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_NODE", "0")
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    monkeypatch.setattr("lightllm.common.req_manager.req_sampling_params.get_vocab_size", lambda _: 128)
    mtp_step, state_mode, use_fp8_kv = request.param
    dtype, head_dim = (torch.uint8, 720) if use_fp8_kv else (torch.bfloat16, 584)
    mem_manager_class = FP8Glm5NextMemManager if use_fp8_kv else Glm5NextMemManager
    args = StartArgs(
        data_type="bfloat16",
        llm_kv_type="fp8kv_dsa" if use_fp8_kv else "None",
        linear_att_hash_page_size=4,
        linear_att_page_block_num=2,
        mtp_step=mtp_step,
        mtp_mode="eagle_with_att" if mtp_step else None,
    )
    set_unique_server_name(args)
    get_env_start_args.cache_clear()
    set_env_start_args(dataclasses.asdict(args))

    def check_shm_refs(mem, req):
        assert mem.req_manager is req
        for field in (
            "mem_manager",
            "small_page_buffers",
            "req_list",
            "req_sampling_params_manager",
        ):
            assert field not in vars(req)
        for field in ("linear_state_manager", "pd_state", "req_to_indexer_tail"):
            assert field not in vars(mem)
        assert mem.big_page_buffers is None

    monkeypatch.setattr(MemoryManager, "write_to_shm", check_shm_refs)

    def create(tp, mode=None):
        args.ssm_state_mode = mode or state_mode
        set_env_start_args(dataclasses.asdict(args))
        get_env_start_args.cache_clear()
        config = Glm5NextCacheConfig(
            tp_world_size=tp,
            full_att_all_num_kv_heads=1,
            full_att_dtype=dtype,
            full_att_num_kv_heads=1,
            full_att_head_dim=head_dim,
            global_linear_k_heads=8,
            global_linear_v_heads=8,
            num_linear_k_heads=8 // tp,
            num_linear_v_heads=8 // tp,
            head_linear_k_dim=128,
            head_linear_v_dim=128,
            conv_kernel_size=4,
            linear_layer_num=6,
            conv_state_dtype=torch.bfloat16,
            ssm_state_dtype=torch.bfloat16 if mtp_step == 3 else torch.float32,
            full_attention_interval=4,
            all_layer_num=8,
            draft_full_att_kv_layer_num=int(mtp_step > 0),
        )
        mems = []
        for _ in range(tp):
            mem = mem_manager_class(32, dtype, 1, head_dim, 2 + int(mtp_step > 0), config)
            req = Glm5NextReqManager(3, 32, mem, config)
            assert req.linear_state_manager.ssm_slots_per_req == (
                mtp_step + 1 if args.ssm_state_mode == "native" else 1
            )
            assert (req.linear_state_manager.ssm_backend is None) == (args.ssm_state_mode == "native")
            mem.write_to_shm(req)
            assert mem.big_page_buffers is not None
            mems.append(mem)
        mems[0].alloc_paged_kv_move_buffer(1, 4096)
        return mems

    return create


@pytest.mark.parametrize("prefill_tp,decode_tp", [(1, 1), (4, 4), (1, 4), (4, 1)])
@pytest.mark.parametrize("remainder", range(4))
@pytest.mark.parametrize("source_mode", ["native", "replay"])
def test_pd_kv_and_runtime_state_roundtrip(make_mems, prefill_tp, decode_tp, remainder, source_mode):
    source, dest = make_mems(prefill_tp, source_mode), make_mems(decode_tp)
    mtp_size = get_env_start_args().mtp_step + 1
    layers = source[0].layer_num
    src_req, dst_req = 0, 2
    length = 8 + remainder
    src_indexes = torch.randperm(32)[:length].tolist()
    dst_indexes = torch.randperm(32)[:length].tolist()
    token_bytes = source[0].head_dim * source[0].dtype.itemsize
    packed = torch.randint(0, 256, (layers, length, 1, token_bytes), dtype=torch.uint8, device="cuda")
    conv = torch.randn(6, 3, 8, 128, 3, dtype=torch.bfloat16, device="cuda")
    ssm = torch.randn(6, 8, 128, 128, dtype=source[0].linear_config.ssm_state_dtype, device="cuda")
    # Non-live slots deliberately contain stale data. The sequence length, not
    # the bytes in these slots, controls subsequent K-pool completion.
    tail = torch.randn(layers, 3 + mtp_size, 256, dtype=torch.bfloat16, device="cuda")
    for rank, mem in enumerate(source):
        heads = slice(rank * 8 // prefill_tp, (rank + 1) * 8 // prefill_tp)
        mem.kv_buffer.view(torch.uint8)[:, src_indexes] = packed
        mem.req_manager.linear_state_manager.conv_state.buffer[:, src_req, ..., :3].copy_(
            conv[:, :, heads].reshape(6, -1, 3)
        )
        mem.req_manager.linear_state_manager.ssm_state.buffer[
            :, src_req * mem.req_manager.linear_state_manager.ssm_slots_per_req
        ].copy_(ssm[:, heads])
        mem.req_manager.req_to_indexer_tail.buffer[:, src_req].copy_(tail)
        ssm_backend = mem.req_manager.linear_state_manager.ssm_backend
        if ssm_backend is not None:
            # The exported checkpoint must contain accepted, unfurled KDA history.
            h = mem.linear_config.num_linear_v_heads
            q = torch.randn(1, 1, h, 128, dtype=torch.bfloat16, device="cuda")
            gate = torch.randn(1, h * 128, dtype=torch.bfloat16, device="cuda")
            beta = torch.randn(1, h, dtype=torch.bfloat16, device="cuda")
            reqs = torch.tensor([src_req], dtype=torch.int32, device="cuda")
            cu = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
            ssm_backend.prepare_decode(reqs, cu)
            for layer in range(6):
                ssm_backend.forward(
                    layer,
                    q,
                    q,
                    q,
                    gate,
                    beta,
                    torch.zeros(h, device="cuda"),
                    torch.zeros(h * 128, device="cuda"),
                    reqs,
                    cu,
                )
            ssm_backend.accept_updates(reqs, torch.zeros(4, dtype=torch.int32, device="cuda"))
            ssm[:, heads].copy_(ssm_backend.get_accepted_state(src_req))

    untouched = []
    for mem in dest:
        mem.kv_buffer.zero_()
        mem.req_manager.linear_state_manager.conv_state.buffer.normal_()
        mem.req_manager.linear_state_manager.ssm_state.buffer.normal_()
        mem.req_manager.req_to_indexer_tail.buffer.normal_()
        if mem.req_manager.linear_state_manager.ssm_backend is not None and hasattr(
            mem.req_manager.linear_state_manager.ssm_backend, "cursors"
        ):
            mem.req_manager.linear_state_manager.ssm_backend.cursors[dst_req] = 5
        if mem.req_manager.linear_state_manager.accepted_state_index is not None:
            mem.req_manager.linear_state_manager.accepted_state_index[dst_req] = mtp_size - 1
        untouched.append(
            [
                x.buffer[:, torch.tensor([0, 1, 3], device="cuda") * stride].clone()
                for x, stride in (
                    (mem.req_manager.linear_state_manager.conv_state, 1),
                    (
                        mem.req_manager.linear_state_manager.ssm_state,
                        mem.req_manager.linear_state_manager.ssm_slots_per_req,
                    ),
                    (mem.req_manager.req_to_indexer_tail, 1),
                )
            ]
        )

    source[0].write_mem_to_page_kv_move_buffer(src_indexes, 0, 0, source, prefill_tp, 0, length)
    # Model a byte-preserving transport between separate P/D page buffers.
    dest[0].kv_move_buffer.view(torch.uint8).copy_(source[0].kv_move_buffer.view(torch.uint8))
    dest[0].read_page_kv_move_buffer_to_mem(dst_indexes, 0, 0, dest, decode_tp, 0, length)
    state_nbytes = source[0].req_manager.get_pd_state_size()
    source_state_bytes = source[0].kv_move_buffer[0].view(torch.uint8).reshape(-1)[:state_nbytes]
    source_state_bytes.zero_()
    source[0].write_mem_to_page_kv_move_buffer([], 0, 0, source, prefill_tp, 0, length, "att_state", src_req)
    # The public request interface packs the same complete block as the PD helper.
    request_state_block = torch.zeros(state_nbytes, dtype=torch.uint8, device="cuda")
    for rank, mem in enumerate(source):
        mem.req_manager.get_pd_state(src_req, request_state_block, rank)
    assert torch.equal(source_state_bytes, request_state_block)
    dest[0].kv_move_buffer.view(torch.uint8).copy_(source[0].kv_move_buffer.view(torch.uint8))
    dest[0].read_page_kv_move_buffer_to_mem([], 0, 0, dest, decode_tp, 0, length, "att_state", dst_req)
    torch.cuda.synchronize()

    for rank, mem in enumerate(dest):
        heads = slice(rank * 8 // decode_tp, (rank + 1) * 8 // decode_tp)
        assert torch.equal(mem.kv_buffer.view(torch.uint8)[:, dst_indexes], packed)
        assert torch.equal(
            mem.req_manager.linear_state_manager.conv_state.buffer[:, dst_req, ..., :3],
            conv[:, :, heads].reshape(6, -1, 3),
        )
        assert torch.equal(
            mem.req_manager.linear_state_manager.ssm_state.buffer[
                :, dst_req * mem.req_manager.linear_state_manager.ssm_slots_per_req
            ],
            ssm[:, heads],
        )
        assert torch.equal(mem.req_manager.req_to_indexer_tail.buffer[:, dst_req], tail)
        if mem.req_manager.linear_state_manager.ssm_backend is not None and hasattr(
            mem.req_manager.linear_state_manager.ssm_backend, "cursors"
        ):
            assert mem.req_manager.linear_state_manager.ssm_backend.cursors[dst_req] == 0
        if mem.req_manager.linear_state_manager.accepted_state_index is not None:
            assert mem.req_manager.linear_state_manager.accepted_state_index[dst_req] == 0
        for (state, stride), expected in zip(
            (
                (mem.req_manager.linear_state_manager.conv_state, 1),
                (
                    mem.req_manager.linear_state_manager.ssm_state,
                    mem.req_manager.linear_state_manager.ssm_slots_per_req,
                ),
                (mem.req_manager.req_to_indexer_tail, 1),
            ),
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
    for mem, req_idx, indexes in (
        (source[0], src_req, src_indexes),
        (dest[0], dst_req, dst_indexes),
    ):
        new_indexes = [i for i in range(32) if i not in indexes][:count]
        all_indexes = indexes + new_indexes
        compress_pools(
            raw=raw,
            tail=mem.req_manager.req_to_indexer_tail.buffer[0],
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
    # Reject a block missing even one byte of the complete state, including the tail.
    mem.kv_move_buffer = torch.empty((1, mem.req_manager.get_pd_state_size() - 1), dtype=torch.uint8, device="cuda")
    with pytest.raises(AssertionError, match="smaller than global linear att state"):
        helper.assert_page_size()
    shape = mem.get_paged_kv_move_buffer_shape(2, 2048)
    assert shape == (2, 2048, mem.layer_num, 1, mem.head_dim)
