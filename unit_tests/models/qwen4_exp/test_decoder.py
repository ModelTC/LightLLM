from types import SimpleNamespace

import pytest
import torch

from lightllm.models.qwen4_exp.layer_weights.transformer_layer_weight import Qwen4ExpTransformerLayerWeight
from lightllm.models.qwen4_exp.layer_infer.transformer_layer_infer import Qwen4ExpTransformerLayerInfer
from lightllm.models.qwen4_exp.state import QSABatch, QSAExecutionState
from lightllm.models.qwen4_exp.batch import PackedBatchMetadata
from lightllm.models.qwen4_exp.position import combine_mrope
from lightllm.common.quantization import Quantcfg
from lightllm.models.qwen4_exp.attention_state import Qwen4ExpAttentionState
from lightllm.common.attention_state.full_att.gqa.gqa_state import GQAAttentionConfig
from .helpers import checkpoint
from .reference import reference_layer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def setup(config, lengths, physical_cache=False):
    source = checkpoint(config, 3)
    weight = Qwen4ExpTransformerLayerWeight(3, torch.bfloat16, config, Quantcfg(config))
    weight.load_hf_weights(dict(source))
    weight.verify_load()
    infer = Qwen4ExpTransformerLayerInfer(3, config)

    from lightllm.utils.dist_utils import get_dp_world_size

    manager = Qwen4ExpAttentionState(
        GQAAttentionConfig(
            logical_layer_groups=((3,),),
            global_head_num=config["num_key_value_heads"],
            head_dim=config["head_dim"],
            dtype=torch.bfloat16,
            tp_world_size=get_dp_world_size(),
        ),
        16,
        4,
        device="cuda",
    )
    manager.allocate_storage(47)
    rows = [5, 2]
    table = torch.full((8, 20), -1, device="cuda", dtype=torch.int32)
    slots = torch.randperm(48, device="cuda")[: sum(lengths)]
    offset = 0
    for row, count in zip(rows, lengths):
        table[row, :count] = slots[offset : offset + count].int()
        offset += count
    dtype = torch.bfloat16
    angle = torch.randn(3, sum(lengths), 2, device="cuda")
    pc, ps = angle.cos().to(dtype), angle.sin().to(dtype)
    cos, sin = combine_mrope(pc, ps, [1, 1, 0])
    key_cos, key_sin = torch.zeros(48, 4, device="cuda", dtype=dtype), torch.zeros(48, 4, device="cuda", dtype=dtype)
    key_cos.index_copy_(0, slots, cos)
    key_sin.index_copy_(0, slots, sin)
    metadata = PackedBatchMetadata.prepare(
        torch.tensor([5, 7, 2], device="cuda", dtype=torch.int32),
        torch.tensor([lengths[0], 0, lengths[1]], device="cuda", dtype=torch.int32),
        torch.tensor([lengths[0], 0, lengths[1]], device="cuda", dtype=torch.int32),
        sum(lengths),
        7,
        4,
    )
    batch = QSABatch.from_metadata(metadata, table, slots, cos, sin, 4)
    view = QSAExecutionState(
        torch.zeros(48, 16, device="cuda", dtype=dtype),
        torch.full((48, 16) if physical_cache else (8, 5, 16), float("nan"), device="cuda", dtype=dtype),
        key_cos,
        key_sin,
    )
    state = SimpleNamespace(
        is_prefill=True,
        need_dp_prefill_balance=False,
        dist_group=None,
        position_cos=pc,
        position_sin=ps,
        mem_index=slots.int(),
        att_state_role="main",
        att_state_manager=SimpleNamespace(get_state=lambda *args, **kwargs: manager),
        qwen4_exp_qsa={3: view},
        qwen4_exp_batch=batch,
    )
    return infer, weight, source, state


@pytest.mark.parametrize("physical_cache", [False, True])
def test_native_qsa_decoder_matches_reference(config, physical_cache):
    torch.manual_seed(937)
    lengths = [13, 5]
    infer, weights, source, state = setup(config, lengths, physical_cache)
    residual = torch.randn(sum(lengths), 64, device="cuda", dtype=torch.bfloat16)
    before = residual.clone()
    actual = infer.context_forward(residual, state, weights)
    expected = reference_layer(residual, source, config, state.qwen4_exp_batch.cos, state.qwen4_exp_batch.sin, lengths)
    torch.testing.assert_close(actual, expected, atol=0.016, rtol=0.016)
    torch.testing.assert_close(residual, before, atol=0, rtol=0)
    key, value = state.att_state_manager.get_state().get_layer_view(3)
    assert key.shape == value.shape

    old_batch = state.qwen4_exp_batch

    used = set(old_batch.mem_indices.tolist())
    slots = torch.tensor([slot for slot in range(48) if slot not in used][:2], device="cuda", dtype=torch.int64)
    angle = torch.randn(3, 2, 2, device="cuda")
    pc, ps = angle.cos().to(residual.dtype), angle.sin().to(residual.dtype)
    cos, sin = combine_mrope(pc, ps, [1, 1, 0])
    view = state.qwen4_exp_qsa[3]
    view.key_cos.index_copy_(0, slots, cos)
    view.key_sin.index_copy_(0, slots, sin)
    for row, count, slot in zip([5, 2], lengths, slots):
        old_batch.slot_table[row, count] = slot.int()
    metadata = PackedBatchMetadata.prepare(
        torch.tensor([5, 2], device="cuda", dtype=torch.int32),
        torch.tensor([14, 6], device="cuda", dtype=torch.int32),
        torch.ones(2, device="cuda", dtype=torch.int32),
        2,
        7,
        4,
    )
    state.qwen4_exp_batch = QSABatch.from_metadata(metadata, old_batch.slot_table, slots, cos, sin, 4)
    state.position_cos, state.position_sin, state.mem_index, state.is_prefill = pc, ps, slots.int(), False
    new_residual = torch.randn(2, 64, device="cuda", dtype=residual.dtype)
    actual = infer.token_forward(new_residual, state, weights)
    all_residual = torch.cat([residual[:13], new_residual[:1], residual[13:], new_residual[1:]])
    all_cos = torch.cat([old_batch.cos[:13], cos[:1], old_batch.cos[13:], cos[1:]])
    all_sin = torch.cat([old_batch.sin[:13], sin[:1], old_batch.sin[13:], sin[1:]])
    full_reference = reference_layer(all_residual, source, config, all_cos, all_sin, [14, 6])
    torch.testing.assert_close(actual, full_reference[[13, 19]], atol=0.016, rtol=0.016)


@pytest.mark.parametrize("physical_cache", [False, True])
def test_native_qsa_decoder_cuda_graph_replay(config, physical_cache):
    torch.manual_seed(937)
    lengths = [13, 5]
    infer, weights, source, state = setup(config, lengths, physical_cache)
    residual = torch.randn(sum(lengths), 64, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            infer.context_forward(residual, state, weights)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = infer.context_forward(residual, state, weights)
    batch = state.qwen4_exp_batch
    view = state.qwen4_exp_qsa[3]
    for _ in range(2):
        residual.normal_()
        angle = torch.randn_like(state.position_cos.float())
        state.position_cos.copy_(angle.cos())
        state.position_sin.copy_(angle.sin())
        cos, sin = combine_mrope(state.position_cos, state.position_sin, [1, 1, 0])
        batch.cos.copy_(cos)
        batch.sin.copy_(sin)
        view.key_cos.index_copy_(0, batch.mem_indices, cos)
        view.key_sin.index_copy_(0, batch.mem_indices, sin)
        graph.replay()
        expected = reference_layer(residual, source, config, cos, sin, lengths)
        torch.testing.assert_close(actual, expected, atol=0.016, rtol=0.016)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("prefill", [False, True])
def test_long_decode_attention_matches_single_queries(dtype, prefill):
    from lightllm.common.qsa.indexer import QSAIndexer

    torch.manual_seed(701)
    slots, rows, dim = 4096, 3, 128

    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=dtype)

    indexer = QSAIndexer(rand(5 * dim, dim), rand(dim), rand(dim))
    raw = rand(slots, dim)
    pooled = rand(1, slots // 4, dim)
    angles = torch.empty(slots, 0, device="cuda", dtype=dtype)
    view = QSAExecutionState(raw, pooled, angles, angles)
    query, key, value = rand(rows, 6, 256), rand(slots, 1, 256), rand(slots, 1, 256)
    hidden = rand(rows, dim)
    table = torch.arange(slots, device="cuda", dtype=torch.int32)[None, :]

    def execute(start, end, decode=True, attention_splits=None):
        count = end - start
        batch = QSABatch(
            table,
            torch.tensor([0], device="cuda", dtype=torch.int32),
            torch.tensor([slots], device="cuda", dtype=torch.int32),
            torch.tensor([slots // 4], device="cuda", dtype=torch.int32),
            torch.zeros(count, device="cuda", dtype=torch.int32),
            torch.arange(2050 + start, 2050 + end, device="cuda", dtype=torch.int32),
            torch.arange(slots - rows + start, slots - rows + end, device="cuda"),
            angles[:count],
            angles[:count],
            0,
        )
        return view.attend(
            indexer,
            hidden[start:end],
            query[start:end],
            key,
            value,
            batch,
            decode=decode,
            attention_splits=attention_splits,
        )

    options = {"decode": not prefill, "attention_splits": 32 if prefill else None}
    expected = torch.cat([execute(row, row + 1, **options) for row in range(rows)])
    torch.testing.assert_close(execute(0, rows, **options), expected, atol=0, rtol=0)
    torch.testing.assert_close(execute(0, 1, decode=False), expected[:1], atol=0, rtol=0)
