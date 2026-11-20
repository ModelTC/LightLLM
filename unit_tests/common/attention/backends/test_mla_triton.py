from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.attention.base_att import AttControl, BaseAttBackend
from lightllm.common.basemodel.attention.triton.mla import MlaTritonAttBackend
from lightllm.common.basemodel.infer_struct import InferStateInfo


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class _Model:
    # BaseAttBackend keys its per-model singletons by the model object, so the stub must be hashable.
    config = {"num_attention_heads": 16}
    tp_world_size_ = 1
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    data_type = torch.bfloat16
    graph_max_len_in_batch = 192


@pytest.fixture
def backend():
    model = _Model()
    backend = MlaTritonAttBackend(model=model)
    yield backend
    BaseAttBackend._instances.pop((MlaTritonAttBackend, model), None)


def _decode_state(lengths, *, capture=False, replay=False):
    state = InferStateInfo()
    state.is_cuda_graph_capture, state.is_cuda_graph_replay = capture, replay
    state.batch_size, state.total_token_num = len(lengths), sum(lengths)
    state.b_req_idx = torch.arange(len(lengths), device="cuda", dtype=torch.int32)
    state.b_seq_len = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    return state


def _reference_decode(q, kv, slots, lengths, scale):
    rows = [kv[slots[i, :length].long(), 0].float() for i, length in enumerate(lengths)]
    return torch.stack([((q[i].float() @ r.T) * scale).softmax(-1) @ r[:, :512] for i, r in enumerate(rows)])


def test_cached_prefill(backend):
    state = InferStateInfo()
    state.b_q_start_loc = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    state.b1_cu_kv_seq_len = torch.tensor([0, 5, 8], dtype=torch.int32, device="cuda")
    state.b_seq_len = torch.tensor([5, 3], dtype=torch.int32, device="cuda")
    state.b_ready_cache_len = torch.tensor([3, 0], dtype=torch.int32, device="cuda")
    state.max_q_seq_len = 3
    att = backend.create_att_prefill_state(infer_state=state)
    q = torch.randn((5, 8, 192), device="cuda", dtype=torch.bfloat16)
    k_nope = torch.randn((8, 8, 128), device="cuda", dtype=q.dtype)
    k_rope = torch.randn((8, 1, 64), device="cuda", dtype=q.dtype)
    v = torch.randn_like(k_nope)
    scale = 0.047
    out = torch.empty_like(q[..., :128])
    actual = att.prefill_att(
        q=q,
        k=(k_nope, k_rope),
        v=v,
        att_control=AttControl(mla_prefill=True, mla_prefill_dict={"softmax_scale": scale}),
        alloc_func=lambda *a, **kw: out,
    )
    assert actual is out
    keys = torch.cat((k_nope, k_rope.expand(-1, 8, -1)), dim=-1).float()
    for i, (start, end) in enumerate([(0, 4), (0, 5), (5, 6), (5, 7), (5, 8)]):
        scores = torch.einsum("hd,khd->hk", q[i].float(), keys[start:end]) * scale
        expected = torch.einsum("hk,khd->hd", scores.softmax(-1), v[start:end].float())
        torch.testing.assert_close(out[i], expected.to(q.dtype), atol=0.012, rtol=0.02)


def test_decode_plans_in_init_state_independent_of_query_layout(backend):
    lengths = (7, 300)
    slots = torch.randperm(768, device="cuda", dtype=torch.int32).view(2, 384)
    state = _decode_state(lengths)
    state.req_manager = SimpleNamespace(req_to_token_indexs=slots)
    att = backend.create_att_decode_state(infer_state=state)
    att.init_state()
    assert att.batch is not None
    kv = torch.randn((768, 1, 576), dtype=torch.bfloat16, device="cuda")
    scale = 576 ** -0.5
    control = AttControl(mla_decode=True, mla_decode_dict={"softmax_scale": scale})
    # Layers may present the absorbed query with different strides; both reuse the init_state plan.
    q_transposed = torch.randn((16, 2, 512), dtype=kv.dtype, device="cuda").transpose(0, 1)
    q_contiguous = q_transposed.contiguous()
    q_rope = torch.randn((2, 16, 192), dtype=kv.dtype, device="cuda")[..., -64:]
    expected = _reference_decode(torch.cat((q_transposed, q_rope), dim=-1), kv, slots, lengths, scale)
    for q_nope in (q_transposed, q_contiguous):
        actual = att.decode_att(q=(q_nope, q_rope), k=kv, v=None, att_control=control)
        torch.testing.assert_close(actual, expected.to(kv.dtype), rtol=0.02, atol=0.012)


def test_decode_replay_refreshes_captured_plan(backend):
    slots = torch.randperm(384, device="cuda", dtype=torch.int32).view(2, 192)
    req_manager = SimpleNamespace(req_to_token_indexs=slots)
    state = _decode_state((7, 5), capture=True)
    state.req_manager = req_manager
    state.decode_att_state = att = backend.create_att_decode_state(infer_state=state)
    att.init_state()
    captured_schedule = att.batch.schedule
    kv = torch.randn((384, 1, 576), dtype=torch.bfloat16, device="cuda")
    # The model produces the absorbed query by transposing the head-major BMM output.
    q_nope = torch.randn((16, 2, 512), dtype=kv.dtype, device=kv.device).transpose(0, 1)
    q_rope = torch.randn((2, 16, 192), dtype=kv.dtype, device=kv.device)[..., -64:]
    q = torch.cat((q_nope, q_rope), dim=-1)
    scale = 576 ** -0.5
    control = AttControl(mla_decode=True, mla_decode_dict={"softmax_scale": scale})

    def decode():
        return att.decode_att(q=(q_nope, q_rope), k=kv, v=None, att_control=control)

    decode()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = decode()
        second_layer = decode()
    for lengths in [(7, 5), (150, 83)]:
        replay_state = _decode_state(lengths, replay=True)
        replay_state.req_manager = req_manager
        replay_state.decode_att_state = replay_att = backend.create_att_decode_state(infer_state=replay_state)
        replay_att.init_state()
        assert replay_att.batch is None
        state.copy_for_cuda_graph(replay_state)
        graph.replay()
        assert att.batch.schedule is captured_schedule
        expected = _reference_decode(q, kv, slots, lengths, scale)
        torch.testing.assert_close(actual, expected.to(kv.dtype), rtol=0.02, atol=0.012)
        torch.testing.assert_close(second_layer, actual, rtol=0, atol=0)
