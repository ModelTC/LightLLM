from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.attention.base_att import AttControl
from lightllm.common.basemodel.attention.triton.mla import MlaTritonAttBackend
from lightllm.common.basemodel.infer_struct import InferStateInfo


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_cached_prefill():
    state = InferStateInfo()
    state.b_q_start_loc = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    state.b1_cu_kv_seq_len = torch.tensor([0, 5, 8], dtype=torch.int32, device="cuda")
    state.b_seq_len = torch.tensor([5, 3], dtype=torch.int32, device="cuda")
    state.b_ready_cache_len = torch.tensor([3, 0], dtype=torch.int32, device="cuda")
    state.max_q_seq_len = 3
    att = MlaTritonAttBackend(model=None).create_att_prefill_state(infer_state=state)
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


def test_decode_replay_recomputes_splits():
    state = InferStateInfo()
    state.batch_size, state.max_kv_seq_len, state.total_token_num = 2, 192, 12
    slots = torch.randperm(384, device="cuda", dtype=torch.int32).view(2, 192)
    state.req_manager = SimpleNamespace(req_to_token_indexs=slots)
    state.b_req_idx = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    state.b_seq_len = torch.tensor([7, 5], device="cuda", dtype=torch.int32)
    att = MlaTritonAttBackend(model=None).create_att_decode_state(infer_state=state)
    att.init_state()
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
        state.b_seq_len.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        graph.replay()
        for i, length in enumerate(lengths):
            rows = kv[slots[i, :length].long(), 0].float()
            expected = ((q[i].float() @ rows.T) * scale).softmax(-1) @ rows[:, :512]
            torch.testing.assert_close(actual[i], expected.to(kv.dtype), rtol=0.02, atol=0.012)
        torch.testing.assert_close(second_layer, actual, rtol=0, atol=0)
