from unit_tests.common.att_resources.model_helpers import full_state_test_fields

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.att_resources.allocators import FullAttKVAllocator
from unit_tests.common.att_resources.model_helpers import create_full_attention_roles

from lightllm.common.att_resources.full_att import (
    PackedMLAFullAttentionState,
    PackedMLAAttentionConfig,
)
from lightllm.server.router.model_infer.mode_backend.pd.resource_io import allocate_transfer_buffer, copy_page
from lightllm.models.glm5_next.att_resource_manager import Glm5NextAttResourceManager
from lightllm.models.glm5_next.att_resource_config import Glm5NextResourceConfig
from lightllm.models.glm5_next.triton_kernel.kpool import compress_pools
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.envs_utils import get_env_start_args, set_env_start_args, set_unique_server_name


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.usefixtures("kv_allocator_environment"),
]


@pytest.fixture(params=[0, 2])
def make_mems(monkeypatch, request):
    monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_NODE", "0")
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    monkeypatch.setattr("lightllm.common.req_manager.req_sampling_params.get_vocab_size", lambda _: 128)
    mtp_step = request.param
    args = StartArgs(
        data_type="bfloat16",
        pd_kv_page_size=2048,
        linear_att_hash_page_size=4,
        linear_att_page_block_num=2,
        mtp_step=mtp_step,
        mtp_mode="eagle_with_att" if mtp_step else None,
    )
    set_unique_server_name(args)
    get_env_start_args.cache_clear()
    set_env_start_args(dataclasses.asdict(args))

    def create(tp, target_mtp_step=None):
        local_mtp_step = mtp_step if target_mtp_step is None else target_mtp_step
        config = Glm5NextResourceConfig(
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
        for rank in range(tp):
            monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_DP", str(rank))
            monkeypatch.setenv("LIGHTLLM_DP_WORLD_SIZE", str(tp))
            full = PackedMLAFullAttentionState(
                PackedMLAAttentionConfig(
                    tuple((layer,) for layer in config.get_full_att_logical_layer_ids()),
                    1,
                    584,
                    torch.bfloat16,
                    tp_rank=rank,
                    tp_world_size=tp,
                    kv_dim=512,
                ),
                 page_size=args.page_size,
            )
            full.initialize_storage(32)
            resources = Glm5NextAttResourceManager(config, 4, local_mtp_step, tp_rank=rank)
            main, draft = create_full_attention_roles(full, config.all_layer_num)
            resources.bind_storage(
                main,
                FullAttKVAllocator(full.capacity, full.page_size),
                draft_full_state=draft,
                big_page_capacity=6,
                small_page_capacity=2,
            )
            mems.append(
                SimpleNamespace(
                    full=main,
                    full_states=tuple(state for state in (main, draft) if state is not None),
                    linear=resources.main_kda_state,
                    tail=resources.main_indexer_kpool_c4_tail_state,
                    tails=resources.get_attention_tail_states(),
                    resources=resources,
                )
            )
        return mems

    return create


@pytest.mark.parametrize("prefill_tp,decode_tp", [(1, 1), (4, 4), (1, 4), (4, 1)])
@pytest.mark.parametrize("remainder", range(4))
@pytest.mark.parametrize("decode_mtp_step", [0, 1, 5])
def test_pd_kv_and_runtime_state_roundtrip(make_mems, prefill_tp, decode_tp, remainder, decode_mtp_step):
    source, dest = make_mems(prefill_tp), make_mems(decode_tp, target_mtp_step=decode_mtp_step)
    mtp_size = get_env_start_args().mtp_step + 1
    dst_mtp_size = decode_mtp_step + 1
    layers = sum(len(state.config.logical_layer_groups) for state in source[0].full_states)
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
    for rank, mem in enumerate(source):
        heads = slice(rank * 8 // prefill_tp, (rank + 1) * 8 // prefill_tp)
        offset = 0
        for state in mem.full_states:
            end = offset + len(state.config.logical_layer_groups)
            state.storage.view(torch.uint8)[:, src_indexes] = packed[offset:end]
            offset = end
        mem.linear.conv_state[:, src_req, ..., :3].copy_(conv[:, :, heads].reshape(6, -1, 3))
        mem.linear.ssm_state[:, src_req * mtp_size].copy_(ssm[:, heads])
        offset = 0
        for state in mem.tails:
            end = offset + len(state.config.logical_layer_ids)
            state.tail_buffer[:, src_req].copy_(tail[offset:end])
            offset = end

    untouched = []
    for mem in dest:
        for state in mem.full_states:
            state.storage.zero_()
        mem.linear.conv_state.normal_()
        mem.linear.ssm_state.normal_()
        for state in mem.tails:
            state.tail_buffer.normal_()
        untouched.append(
            [
                x[:, torch.tensor([0, 1, 3], device="cuda") * stride].clone()
                for x, stride in (
                    (mem.linear.conv_state, 1),
                    (mem.linear.ssm_state, dst_mtp_size),
                    *((state.tail_buffer, 1) for state in mem.tails),
                )
            ]
        )

    source_resources = [mem.resources for mem in source]
    dest_resources = [mem.resources for mem in dest]
    source_buffer = allocate_transfer_buffer(source_resources[0], 1, device="cuda")
    dest_buffer = allocate_transfer_buffer(dest_resources[0], 1, device="cuda")
    assert source_buffer.shape == dest_buffer.shape
    assert source_resources[0].get_pd_bounded_att_or_tail_transfer_layouts().keys() == dest_resources[0].get_pd_bounded_att_or_tail_transfer_layouts().keys()
    for kind in ("kv", "att_state"):
        copy_page(
            source_resources,
            source_buffer,
            page_index=0,
            dp_index=0,
            tp_size=prefill_tp,
            token_indices=torch.tensor(src_indexes if kind == "kv" else [], dtype=torch.long, device="cuda"),
            kind=kind,
            req_idx=src_req,
            state_token_len=length,
            write=True,
        )
        dest_buffer.copy_(source_buffer)
        copy_page(
            dest_resources,
            dest_buffer,
            page_index=0,
            dp_index=0,
            tp_size=decode_tp,
            token_indices=torch.tensor(dst_indexes if kind == "kv" else [], dtype=torch.long, device="cuda"),
            kind=kind,
            req_idx=dst_req,
            state_token_len=length,
            write=False,
        )
    torch.cuda.synchronize()

    for rank, mem in enumerate(dest):
        heads = slice(rank * 8 // decode_tp, (rank + 1) * 8 // decode_tp)
        for state, source_state in zip(mem.full_states, source[0].full_states):
            expected = full_state_test_fields(source_state, {"kv": source_state.storage})
            for name, field in full_state_test_fields(state, {"kv": state.storage}).items():
                source_field = expected[name]
                assert torch.equal(
                    field.tensor[:, dst_indexes].contiguous().view(torch.uint8),
                    source_field.tensor[:, src_indexes].contiguous().view(torch.uint8),
                )
        assert torch.equal(mem.linear.conv_state[:, dst_req, ..., :3], conv[:, :, heads].reshape(6, -1, 3))
        assert torch.equal(mem.linear.ssm_state[:, dst_req * dst_mtp_size], ssm[:, heads])
        restored_tail = torch.cat([state.tail_buffer[:, dst_req] for state in mem.tails])
        expected_tail = torch.zeros_like(restored_tail)
        for position in range(length - 4, length):
            expected_tail[:, position % (4 + decode_mtp_step)] = tail[:, position % tail.shape[1]]
        assert torch.equal(restored_tail, expected_tail)
        if mem.linear.mtp_state_index is not None:
            assert mem.linear.mtp_state_index[dst_req] == 0
        for (state, stride), expected in zip(
            (
                (mem.linear.conv_state, 1),
                (mem.linear.ssm_state, dst_mtp_size),
                *((state.tail_buffer, 1) for state in mem.tails),
            ),
            untouched[rank],
        ):
            indexes = torch.tensor([0, 1, 3], device="cuda") * stride
            assert torch.equal(state[:, indexes], expected)

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
            tail=mem.tail.tail_buffer[0],
            packed_buffer=mem.full.get_layer_view(3).indexer_k,
            ape=ape,
            lengths=tensor(range(length + 1, length + count + 1)),
            starts=tensor([0] * count),
            ragged=tensor(all_indexes),
            req_idx=tensor([req_idx]),
            cu_q_lens=tensor([0, count]),
            seq_lens=tensor([length + count]),
            max_q_len=count,
        )
        outputs.append(mem.full.get_layer_view(3).indexer_k[new_indexes[-1]].clone())
    assert torch.equal(*outputs)


def test_pd_page_capacity_includes_tail_without_resizing(make_mems, monkeypatch, caplog):
    monkeypatch.setattr(get_env_start_args(), "pd_kv_page_size", 8)
    mem = make_mems(1)[0]
    full_bytes = max(layout.byte_end for layout in mem.resources.get_pd_full_att_transfer_layouts().values())
    runtime_bytes = max(
        layout.byte_end for layout in mem.resources.get_pd_bounded_att_or_tail_transfer_layouts().values()
    )
    assert runtime_bytes > full_bytes
    assert mem.resources.get_pd_page_bytes() == runtime_bytes
    assert "smaller than the complete attention runtime" in caplog.text


@pytest.mark.parametrize("source_step,target_step", [(0, 1), (0, 5), (3, 2), (5, 1)])
@pytest.mark.parametrize("prefix", [1, 3, 4, 5, 11, 12])
def test_pd_prefill_tail_continues_mtp_after_rejection(source_step, target_step, prefix):
    """真实 prefill 导出后，在不同大小的 ring 上验证、拒绝回退并再次验证。"""
    from lightllm.common.att_resources.att_tail import KPoolStateConfig, create_kpool_state
    from lightllm.common.att_resources.types import StateTransferLayout

    torch.manual_seed(53)
    config = KPoolStateConfig(logical_layer_ids=(0,), pool_size=4, head_dim=128, dtype=torch.bfloat16)
    source = create_kpool_state(config=config, request_capacity=2, mtp_step=source_step, device="cuda")
    target = create_kpool_state(config=config, request_capacity=2, mtp_step=target_step, device="cuda")
    raw = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    ape = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    packed = torch.zeros(64, 132, device="cuda", dtype=torch.uint8)
    ragged = torch.arange(64, device="cuda", dtype=torch.int32)

    def prefill(values, destination, tail):
        count = len(values)
        compress_pools(
            raw=values, tail=tail, packed_buffer=destination, ape=ape,
            lengths=torch.arange(1, count + 1, device="cuda", dtype=torch.int32),
            starts=torch.zeros(count, device="cuda", dtype=torch.int32), ragged=ragged,
            req_idx=torch.zeros(1, device="cuda", dtype=torch.int32),
            cu_q_lens=torch.tensor([0, count], device="cuda", dtype=torch.int32),
            seq_lens=torch.tensor([count], device="cuda", dtype=torch.int32), max_q_len=count,
        )

    prefill(raw[:prefix], packed, source.get_layer_view(0))
    regions = source.get_pd_transfer_regions()
    assert regions == target.get_pd_transfer_regions()
    layout = StateTransferLayout(
        resource_id=source.resource_id, tokens_per_page=None,
        byte_start=0, byte_end=regions[-1].byte_end, regions=regions,
    )
    pages = torch.empty((1, layout.byte_end), dtype=torch.uint8, device="cuda")
    source.write_to_pd_buffer(0, pages, pd_page_index=0, state_token_len=prefix, layout=layout)
    target.tail_buffer.fill_(17)
    target.read_from_pd_buffer(pages, 0, pd_page_index=0, state_token_len=prefix, layout=layout)
    assert (target.tail_buffer[:, 1] == 17).all()
    width = target_step + 1
    # 第一次验证后回访已接受位置；新增候选历史必须由 D 的本地 ring 正确保留。
    for start, count in [(prefix, width), (prefix + 1, 1), (prefix + 2, 1), (prefix + 1, width)]:
        raw[start:start + count] = torch.randn(count, 256, device="cuda", dtype=torch.bfloat16)
        lengths = torch.arange(start + 1, start + count + 1, device="cuda", dtype=torch.int32)
        compress_pools(
            raw=raw[start:start + count], tail=target.get_layer_view(0), packed_buffer=packed, ape=ape,
            lengths=lengths, starts=torch.zeros(count, device="cuda", dtype=torch.int32), ragged=ragged,
            req_idx=torch.zeros(count, device="cuda", dtype=torch.int32),
            cu_q_lens=torch.arange(count + 1, device="cuda", dtype=torch.int32),
            seq_lens=lengths, max_q_len=1,
            mtp_index=torch.arange(count, device="cuda", dtype=torch.int32),
        )
        expected = torch.zeros_like(packed)
        prefill(raw[:start + count], expected, torch.zeros_like(source.get_layer_view(0)))
        closing = torch.arange(start, start + count, device="cuda")
        closing = closing[(closing + 1) % 4 == 0]
        assert torch.equal(packed[closing], expected[closing])
