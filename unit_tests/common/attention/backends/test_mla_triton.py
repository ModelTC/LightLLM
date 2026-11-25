from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention.backends.mla import MLADecodeLayerConfig, MLADecodeLayerInputs
from lightllm.common.attention.backends.mla import MLAPrefillLayerConfig, MLAPrefillLayerInputs
from lightllm.common.attention.backends.mla.triton import TritonMLADecodeBackend, TritonMLAPrefillBackend
from lightllm.common.basemodel.attention_runtime import AttentionStepConfig, build_attention_metadata
from lightllm.common.basemodel.infer_struct import GraphMode, InferStateInfo


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# The model's step config: graph steps cover 192 kv tokens.
STEP_CONFIG = AttentionStepConfig(cuda_graph_max_kv_seq_len=192)
DECODE_SCALE = 576 ** -0.5


def decode_config(scale=DECODE_SCALE):
    return MLADecodeLayerConfig(
        num_heads=16, kv_lora_rank=512, qk_rope_head_dim=64, softmax_scale=scale, dtype=torch.bfloat16
    )


@pytest.fixture
def decode_builder():
    # One decode step builder serves the step's MLA layers.
    return TritonMLADecodeBackend.create_build_context(TritonMLADecodeBackend.build_context_key(decode_config()), None)


def decode(att_state, q_nope, q_rope, kv):
    impl = TritonMLADecodeBackend(decode_config())
    out = torch.empty(q_nope.shape, dtype=q_nope.dtype, device=q_nope.device)
    return impl.forward(att_state, MLADecodeLayerInputs(q_nope=q_nope, q_rope=q_rope, kv_cache=kv), out=out)


def _decode_state(lengths, *, capture=False, replay=False):
    state = InferStateInfo()
    state.graph_mode = GraphMode.CAPTURE if capture else GraphMode.REPLAY if replay else GraphMode.EAGER
    state.batch_size, state.total_token_num = len(lengths), sum(lengths)
    state.b_req_idx = torch.arange(len(lengths), device="cuda", dtype=torch.int32)
    state.b_seq_len = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    return state


def _reference_decode(q, kv, slots, lengths, scale):
    rows = [kv[slots[i, :length].long(), 0].float() for i, length in enumerate(lengths)]
    return torch.stack([((q[i].float() @ r.T) * scale).softmax(-1) @ r[:, :512] for i, r in enumerate(rows)])


# 192 nope + 64 rope (GLM-4 MoE Lite) runs the expanded-key kernel, whose own default scale is 1/16.
@pytest.mark.parametrize("nope_dim,v_dim", [(128, 128), (192, 256)])
def test_cached_prefill(nope_dim, v_dim):
    state = InferStateInfo()
    state.b_q_start_loc = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    state.b1_cu_kv_seq_len = torch.tensor([0, 5, 8], dtype=torch.int32, device="cuda")
    state.b_seq_len = torch.tensor([5, 3], dtype=torch.int32, device="cuda")
    state.b_ready_cache_len = torch.tensor([3, 0], dtype=torch.int32, device="cuda")
    state.max_q_seq_len = 3
    state.req_manager = SimpleNamespace(req_to_token_indexs=torch.zeros((2, 8), dtype=torch.int32, device="cuda"))
    state.is_prefill = True
    q = torch.randn((5, 8, nope_dim + 64), device="cuda", dtype=torch.bfloat16)
    k_nope = torch.randn((8, 8, nope_dim), device="cuda", dtype=q.dtype)
    k_rope = torch.randn((8, 1, 64), device="cuda", dtype=q.dtype)
    v = torch.randn((8, 8, v_dim), device="cuda", dtype=q.dtype)
    scale = 0.047
    out = torch.empty((5, 8, v_dim), device="cuda", dtype=q.dtype)
    layer_config = MLAPrefillLayerConfig(
        num_heads=8,
        qk_nope_head_dim=nope_dim,
        qk_rope_head_dim=64,
        v_head_dim=v_dim,
        softmax_scale=scale,
        dtype=q.dtype,
    )
    att_state = TritonMLAPrefillBackend.build(build_attention_metadata(state, STEP_CONFIG), None)
    actual = TritonMLAPrefillBackend(layer_config).forward(
        att_state,
        MLAPrefillLayerInputs(q=q, k_nope=k_nope, k_rope=k_rope, v=v),
        out=out,
        # The expanded-key path also allocates scratch keys.
        alloc_func=lambda shape, **kw: torch.empty(shape, **kw),
    )
    assert actual is out
    keys = torch.cat((k_nope, k_rope.expand(-1, 8, -1)), dim=-1).float()
    for i, (start, end) in enumerate([(0, 4), (0, 5), (5, 6), (5, 7), (5, 8)]):
        scores = torch.einsum("hd,khd->hk", q[i].float(), keys[start:end]) * scale
        expected = torch.einsum("hk,khd->hd", scores.softmax(-1), v[start:end].float())
        torch.testing.assert_close(out[i], expected.to(q.dtype), atol=0.012, rtol=0.02)


def test_decode_plans_once_per_step_independent_of_query_layout(decode_builder):
    lengths = (7, 300)
    slots = torch.randperm(768, device="cuda", dtype=torch.int32).view(2, 384)
    state = _decode_state(lengths)
    state.req_manager = SimpleNamespace(req_to_token_indexs=slots)
    att_state = TritonMLADecodeBackend.build(build_attention_metadata(state, STEP_CONFIG), decode_builder)
    kv = torch.randn((768, 1, 576), dtype=torch.bfloat16, device="cuda")
    scale = DECODE_SCALE
    # Layers may present the absorbed query with different strides; both reuse the step's args.
    q_transposed = torch.randn((16, 2, 512), dtype=kv.dtype, device="cuda").transpose(0, 1)
    q_contiguous = q_transposed.contiguous()
    q_rope = torch.randn((2, 16, 192), dtype=kv.dtype, device="cuda")[..., -64:]
    expected = _reference_decode(torch.cat((q_transposed, q_rope), dim=-1), kv, slots, lengths, scale)
    for q_nope in (q_transposed, q_contiguous):
        actual = decode(att_state, q_nope, q_rope, kv)
        torch.testing.assert_close(actual, expected.to(kv.dtype), rtol=0.02, atol=0.012)


def test_decode_replay_rebuilds_captured_schedule(decode_builder):
    slots = torch.randperm(384, device="cuda", dtype=torch.int32).view(2, 192)
    req_manager = SimpleNamespace(req_to_token_indexs=slots)
    state = _decode_state((7, 5), capture=True)
    state.req_manager = req_manager
    att_state = TritonMLADecodeBackend.build(build_attention_metadata(state, STEP_CONFIG), decode_builder)
    captured_schedule = att_state.schedule
    kv = torch.randn((384, 1, 576), dtype=torch.bfloat16, device="cuda")
    # The model produces the absorbed query by transposing the head-major BMM output.
    q_nope = torch.randn((16, 2, 512), dtype=kv.dtype, device=kv.device).transpose(0, 1)
    q_rope = torch.randn((2, 16, 192), dtype=kv.dtype, device=kv.device)[..., -64:]
    q = torch.cat((q_nope, q_rope), dim=-1)
    scale = DECODE_SCALE

    def run():
        return decode(att_state, q_nope, q_rope, kv)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
        second_layer = run()
    for lengths in [(7, 5), (150, 83)]:
        replay_state = _decode_state(lengths, replay=True)
        replay_state.req_manager = req_manager
        replay_att_state = TritonMLADecodeBackend.build(
            build_attention_metadata(replay_state, STEP_CONFIG), decode_builder
        )
        # Building the replay step's args rewrites the schedule the graph captured instead of making a new one.
        assert replay_att_state.schedule is captured_schedule
        state.copy_for_cuda_graph(replay_state)
        graph.replay()
        expected = _reference_decode(q, kv, slots, lengths, scale)
        torch.testing.assert_close(actual, expected.to(kv.dtype), rtol=0.02, atol=0.012)
        torch.testing.assert_close(second_layer, actual, rtol=0, atol=0)


def test_decode_step_builder_keeps_one_graph_schedule_per_microbatch_and_batch_size(decode_builder):
    from lightllm.common.attention.backend import AttentionMetadata

    def make_batch(lengths, microbatch_index=0, graph=True):
        b_seq_len = torch.tensor(lengths, device="cuda", dtype=torch.int32)
        return AttentionMetadata(
            req_to_token_indexs=torch.zeros((len(lengths), 192), device="cuda", dtype=torch.int32),
            b_req_idx=torch.arange(len(lengths), device="cuda", dtype=torch.int32),
            b_seq_len=b_seq_len,
            batch_size=len(lengths),
            max_kv_seq_len=max(lengths),
            total_token_num=sum(lengths),
            cuda_graph_max_kv_seq_len=192 if graph else None,
            microbatch_index=microbatch_index,
        )

    step_builder = decode_builder
    first = TritonMLADecodeBackend.build(make_batch((7, 5)), step_builder).schedule
    # Concurrent microbatches of microbatch overlap must not share captured buffers.
    other_microbatch = TritonMLADecodeBackend.build(make_batch((7, 5), microbatch_index=1), step_builder).schedule
    other_batch_size = TritonMLADecodeBackend.build(make_batch((7, 5, 3)), step_builder).schedule
    assert first is not other_microbatch and first is not other_batch_size
    starts = first.batch_start_index.clone()
    assert TritonMLADecodeBackend.build(make_batch((150, 83)), step_builder).schedule is first
    # Both length sets use 32-token blocks, so (7, 5) starts at [0, 1] and (150, 83) at [0, 5].
    assert starts.tolist() == [0, 1] and first.batch_start_index.tolist() == [0, 5]
    eager = [TritonMLADecodeBackend.build(make_batch((7, 5), graph=False), step_builder).schedule for _ in range(2)]
    assert eager[0] is not eager[1] and all(first is not schedule for schedule in eager)


@pytest.mark.parametrize(
    "spec_cls,fields",
    [
        ("MLAPrefillLayerConfig", dict(num_heads=8, qk_nope_head_dim=0, qk_rope_head_dim=64, v_head_dim=128)),
        ("MLADecodeLayerConfig", dict(num_heads=16, kv_lora_rank=512, qk_rope_head_dim=-64)),
    ],
)
def test_specs_reject_non_positive_dims(spec_cls, fields):
    from lightllm.common.attention.backends import mla

    with pytest.raises(ValueError, match="must be positive"):
        getattr(mla, spec_cls)(softmax_scale=0.1, dtype=torch.bfloat16, **fields)
