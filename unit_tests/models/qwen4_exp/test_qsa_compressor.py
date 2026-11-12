from dataclasses import replace

import pytest
import torch

from lightllm.common.attention_state.types import StateTransferLayout
from lightllm.common.qsa.indexer import QSAIndexer
from lightllm.models.qwen4_exp.batch import PackedBatchMetadata
from lightllm.models.qwen4_exp.qsa_compressor import QSACompressorState, pool_with_ring
from lightllm.models.qwen4_exp.state import QSABatch
from lightllm.models.qwen4_exp.model import Qwen4ExpTpPartModel
from lightllm.models.qwen4_exp.mtp.model import Qwen4ExpMTPModel
from lightllm.utils.envs_utils import get_env_start_args


@pytest.mark.parametrize("fp8", [False, True])
def test_compact_factory_accounts_for_storage_and_draft(config, fp8):
    args = get_env_start_args()
    args.data_type = "bfloat16"
    args.page_size = 4
    args.llm_kv_type = "fp8kv_spt" if fp8 else "None"
    args.mtp_step = 3
    (state,) = Qwen4ExpTpPartModel.create_full_att_states(config, args, (0, 1, 2, 3))
    state.device = torch.device("cpu")
    state.allocate_storage(16)
    assert state.positions is None
    assert state.indexer.v_storage is None
    assert state.compress_ratio == 4
    assert state.qsa_view(3).pooled_keys.shape == (5, 16)
    total = sum(t.numel() * t.element_size() for t in (state.k_storage, state.v_storage, state.indexer.k_storage))
    assert total == 20 * state.get_token_bytes()
    assert len(state._get_transfer_regions(8)) == 3
    for cls, layers in ((Qwen4ExpTpPartModel, (0, 1, 2, 3)), (Qwen4ExpMTPModel, (4,))):
        bounded = cls.create_bounded_att_states(config, args, layers, 4, device="cpu")
        compressor = next(s for s in bounded if isinstance(s, QSACompressorState))
        assert compressor.buffer.shape == (1, 4, 7, 24)
        assert not compressor.get_cpu_transfer_regions()
        assert compressor.get_pd_transfer_regions()[0].shape == (1, 4, 24)
        if cls is Qwen4ExpMTPModel:
            assert len(bounded) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("segment_capacity", [None, 257])
def test_ring_pool_matches_full_history_after_chunking_and_rejection(fp8, graph, segment_capacity):
    torch.manual_seed(2437)
    device, dtype = "cuda", torch.bfloat16
    dim, rotary, capacity = 16, 4, 64
    indexer = QSAIndexer(
        torch.randn(48, 32, device=device, dtype=dtype),
        torch.randn(dim, device=device, dtype=dtype),
        torch.randn(dim, device=device, dtype=dtype),
        query_heads=2,
        token_budget=8,
    )
    storage_dtype = torch.float8_e4m3fn if fp8 else dtype
    raw = torch.zeros(capacity, dim, device=device, dtype=dtype)
    cos = torch.zeros(capacity, rotary, device=device, dtype=dtype)
    sin = torch.zeros_like(cos)
    dense = torch.empty(4, 8, dim, device=device, dtype=storage_dtype)
    compact = torch.empty(capacity // 4, dim, device=device, dtype=storage_dtype)
    ring = QSACompressorState((3,), dim, rotary, 4, dtype, 4, 3, 0, 1, device).get_layer_view(3)
    table = torch.full((4, 32), -1, device=device, dtype=torch.int32)
    slots = torch.randperm(16, device=device).view(2, 8) * 4
    for row, pages in zip((1, 2), slots):
        table[row] = (pages[:, None] + torch.arange(4, device=device)).flatten().int()

    def phase(prefixes, counts, captured=False):
        lengths = [a + b for a, b in zip(prefixes, counts)]
        metadata = PackedBatchMetadata.prepare(
            torch.tensor([1, 2], device=device, dtype=torch.int32),
            torch.tensor(lengths, device=device, dtype=torch.int32),
            torch.tensor(counts, device=device, dtype=torch.int32),
            sum(counts),
            3,
            4,
            segment_capacity=segment_capacity,
        )
        ids = torch.cat([table[r, p : p + n] for r, p, n in zip((1, 2), prefixes, counts)]).long()
        projected = torch.randn(sum(counts), 48, device=device, dtype=dtype)
        current = projected[:, 32:]
        angles = torch.randn(sum(counts), rotary, device=device)
        c, s = angles.cos().to(dtype), angles.sin().to(dtype)
        batch = QSABatch.from_metadata(metadata, table, ids, c, s, 4)
        if captured:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            saved = ring.clone()
            with torch.cuda.stream(stream):
                for _ in range(3):
                    pool_with_ring(indexer, current, ring, batch, compact)
            torch.cuda.current_stream().wait_stream(stream)
            ring.copy_(saved)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                pool_with_ring(indexer, current, ring, batch, compact)
            ring.copy_(saved)
            g.replay()
        else:
            pool_with_ring(indexer, current, ring, batch, compact)
        raw.index_copy_(0, ids, current)
        cos.index_copy_(0, ids, c)
        sin.index_copy_(0, ids, s)
        indexer.pool(
            raw,
            table,
            batch.key_lengths,
            cos,
            sin,
            dense,
            batch.first_blocks,
            batch.max_new_blocks,
            request_rows=batch.active_rows,
        )
        for r, length in zip((1, 2), lengths):
            completed = length // 4
            physical = table[r, 3 : completed * 4 : 4].long() // 4
            torch.testing.assert_close(compact.float()[physical], dense.float()[r, :completed], atol=0, rtol=0)
        query = torch.randn(sum(counts), 2, dim, device=device, dtype=dtype).to(storage_dtype)
        actual = indexer.select(query, compact, batch.query_rows, batch.visible_lengths, table, physical_stride=4)
        expected = indexer.select(query, dense, batch.query_rows, batch.visible_lengths, table)
        torch.testing.assert_close(actual.sort(-1).values, expected.sort(-1).values, atol=0, rtol=0)

    phase([0, 0], [3, 2])
    phase([3, 2], [6, 7])
    phase([9, 9], [4, 4], graph)
    phase([10, 11], [4, 4], graph)
    phase([14, 15], [3, 2], graph)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("page_device", ["cpu", "cuda"])
def test_compact_token_storage_round_trip(config, fp8, page_device):
    args = get_env_start_args()
    args.data_type, args.page_size = "bfloat16", 4
    args.llm_kv_type = "fp8kv_spt" if fp8 else "None"
    args.cpu_cache_token_page_size = args.pd_kv_page_size = 8
    states = []
    for _ in range(2):
        (state,) = Qwen4ExpTpPartModel.create_full_att_states(config, args, (0, 1, 2, 3))
        state.device = torch.device("cuda")
        state.allocate_storage(16)
        states.append(state)
    source, target = states

    def tensors(state):
        return state.k_storage, state.v_storage, state.indexer.k_storage

    for tensor in tensors(source):
        tensor.view(torch.uint8).random_(0, 256)
    for tensor in tensors(target):
        tensor.view(torch.uint8).fill_(173)
    src = torch.arange(4, 12, device="cuda", dtype=torch.int32)
    dst = torch.cat([torch.arange(12, 16, device="cuda"), torch.arange(0, 4, device="cuda")]).long()
    expected = [tensor.view(torch.uint8)[:, src].clone() for tensor in tensors(source)]
    source.copy_full_att_tokens(src, dst)
    target.copy_tokens_from_dp(src, dst, source)
    for tensor, gold in zip(tensors(target), expected):
        assert torch.equal(tensor.view(torch.uint8)[:, dst], gold)
        tensor.view(torch.uint8).fill_(173)
    regions = source.get_cpu_transfer_regions() if page_device == "cpu" else source.get_pd_transfer_regions()
    layout = StateTransferLayout(
        source.state_name,
        8,
        16,
        16 + regions[-1].byte_end,
        tuple(replace(r, byte_start=r.byte_start + 16, byte_end=r.byte_end + 16) for r in regions),
    )
    pages = torch.full(
        (2, layout.byte_end + 16), 91, dtype=torch.uint8, device=page_device, pin_memory=page_device == "cpu"
    )
    if page_device == "cpu":
        source.write_to_cpu(src, pages, [1], layout)
        target.read_from_cpu(pages, [1], dst, layout)
    else:
        source.write_to_pd_page(src, pages, 1, layout)
        target.read_from_pd_page(pages, dst, 1, layout)
    torch.cuda.synchronize()
    for tensor, gold in zip(tensors(target), expected):
        assert torch.equal(tensor.view(torch.uint8)[:, dst], gold)
    assert (pages[0] == 91).all()
    assert (pages[:, :16] == 91).all() and (pages[:, layout.byte_end :] == 91).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("length", [0, 3, 4, 13])
def test_compressor_pd_restores_logical_window_across_mtp_capacities(length):
    source = QSACompressorState((3, 7), 16, 4, 4, torch.bfloat16, 4, 3, 0, 1, "cuda")
    target = QSACompressorState((3, 7), 16, 4, 4, torch.bfloat16, 4, 1, 0, 1, "cuda")
    source.buffer.normal_()
    target.buffer.fill_(173)
    source.state_name = target.state_name = "main_qsa"
    regions = source.get_pd_transfer_regions()
    layout = StateTransferLayout(
        source.state_name,
        None,
        16,
        16 + regions[-1].byte_end,
        tuple(replace(r, byte_start=r.byte_start + 16, byte_end=r.byte_end + 16) for r in regions),
    )
    pages = torch.full((2, layout.byte_end + 16), 91, dtype=torch.uint8, device="cuda")
    source.write_to_pd_page(1, pages, 1, length, layout)
    target.read_from_pd_page(pages, 2, 1, length, layout)
    for position in range(length - 4, length):
        actual = target.buffer[:, 2, position % 5]
        expected = source.buffer[:, 1, position % 7] if position >= 0 else torch.zeros_like(actual)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert (pages[0] == 91).all()
    assert (pages[:, :16] == 91).all() and (pages[:, layout.byte_end :] == 91).all()
    before = source.buffer.clone()
    source.save_bounded_att_runtime_to_pages([1], [512], "small", [0])
    source.restore_bounded_att_runtime_from_pages([2], [512], "small", [0])
    torch.testing.assert_close(source.buffer, before, atol=0, rtol=0)
    with pytest.raises(AssertionError, match="close a compressor stride"):
        source.restore_bounded_att_runtime_from_pages([2], [511], "small", [0])
