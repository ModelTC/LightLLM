from types import SimpleNamespace
import weakref

import pytest
import torch

from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.basemodel.attention.triton.fp import TritonAttBackend
from unit_tests.common.basemodel.attention.test_binding import bind_test_model, make_state


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("heads,dim", [(2, 128), (16, 256), (4, 512)])
def test_bound_gqa_prefill_decode_and_replay_with_new_request_contents(heads, dim):
    torch.manual_seed(42)
    state = make_state(((3,),), heads=heads, dim=dim, device="cuda")
    kv = torch.randn(8, 2 * heads, dim, dtype=torch.bfloat16, device="cuda")
    state.write_full_att_kv(3, torch.arange(8, dtype=torch.int32, device="cuda"), kv)
    model = bind_test_model([state])
    model.mtp_manager = SimpleNamespace(get_decode_draft_step=lambda _: 0)
    model.is_mtp_draft_model = False
    backend = TritonAttBackend(model)
    layer = model.layers_infer[0]
    layer.alloc_tensor = lambda shape, dtype, **kwargs: torch.empty(shape, dtype=dtype, device="cuda")
    infer = InferStateInfo()
    infer.batch_size, infer.total_token_num = 2, 7
    infer.max_q_seq_len, infer.max_kv_seq_len = 1, 4
    infer.b_req_idx = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    infer.b_seq_len = torch.tensor([3, 4], dtype=torch.int32, device="cuda")
    infer.b_ready_cache_len = infer.b_seq_len - 1
    infer.b_q_start_loc = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    infer.b1_cu_q_seq_len = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
    infer.req_manager = SimpleNamespace(
        req_to_token_indexs=torch.tensor([[0, 3, 6, 5], [2, 4, 7, 1]], dtype=torch.int32, device="cuda")
    )
    q = torch.randn(2, 2 * heads, dim, dtype=torch.bfloat16, device="cuda")

    def reference():
        results = []
        for batch, length in enumerate((3, 4)):
            ids = infer.req_manager.req_to_token_indexs[batch, :length].long()
            k, v = state.k_storage[0, ids].float(), state.v_storage[0, ids].float()
            k, v = k.repeat_interleave(2, dim=1), v.repeat_interleave(2, dim=1)
            scores = torch.einsum("hd,thd->ht", q[batch].float(), k) / dim ** 0.5
            results.append(torch.einsum("ht,thd->hd", scores.softmax(-1), v))
        return torch.stack(results).to(q.dtype)

    prefill = backend.create_att_prefill_state(infer)
    prefill.init_state()
    infer.attention_metadata = (prefill, None)
    torch.testing.assert_close(layer._context_attention_kernel(q, None, infer, None), reference(), atol=0.03, rtol=0.03)
    decode = backend.create_att_decode_state(infer)
    decode.init_state()
    infer.attention_metadata = (decode, None)
    for _ in range(3):
        layer._token_attention_kernel(q, infer, None)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = layer._token_attention_kernel(q, infer, None)
    for mapping in ([[1, 2, 7, 4], [0, 3, 5, 6]], [[6, 5, 4, 3], [7, 2, 1, 0]]):
        infer.req_manager.req_to_token_indexs.copy_(torch.tensor(mapping, dtype=torch.int32, device="cuda"))
        q.copy_(torch.randn_like(q))
        graph.replay()
        torch.testing.assert_close(out, reference(), atol=0.03, rtol=0.03)


def test_flashinfer_heterogeneous_layers_share_pages_and_replay_changed_lengths(monkeypatch):
    from lightllm.common.basemodel.attention.flashinfer.fp import FlashInferAttBackend
    from lightllm.common.basemodel.attention.legacy import prepare_model_attention, bind_model_attention
    from lightllm.server.core.objs.start_args_type import StartArgs

    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    monkeypatch.setenv("LIGHTLLM_DP_WORLD_SIZE", "1")
    torch.manual_seed(36)
    states = [make_state(((i,),), heads=h, dim=d, device="cuda") for i, (h, d) in enumerate(((2, 64), (4, 128)))]
    for state in states:
        state.page_size = 2
        state.allocate_storage(8)
        state.k_storage.normal_()
        state.v_storage.normal_()
    model = bind_test_model(states)
    model.args = StartArgs(page_size=2)
    model.config = dict(num_attention_heads=4, num_key_value_heads=2, hidden_size=256)
    model.max_seq_length = model.graph_max_len_in_batch = 8
    model.graph_max_batch_size = 2
    model.prefill_att_backend = model.decode_att_backend = FlashInferAttBackend(model)
    prepare_model_attention(model)
    for backend in model.attention_decode_backends:
        if backend is not None:
            backend.reserve_runtime_resources()
    bind_model_attention(model)
    alloc = lambda shape, dtype, **kwargs: torch.empty(shape, dtype=dtype, device="cuda")
    for layer in model.layers_infer:
        layer.alloc_tensor = alloc
    qs = [torch.randn(2, 2 * s.config.head_num, s.config.head_dim, device="cuda", dtype=torch.bfloat16) for s in states]

    def batch(lengths, request_ids, mapping, *, prefill=False, replay=False):
        infer = InferStateInfo()
        infer.batch_size, infer.max_q_seq_len, infer.max_kv_seq_len = 2, 1, max(lengths)
        infer.total_token_num, infer.is_prefill = sum(lengths), prefill
        infer.is_cuda_graph_replay = replay
        tensor = lambda x: torch.tensor(x, dtype=torch.int32, device="cuda")
        infer.input_ids = tensor([1, 2])
        infer.b_req_idx = tensor(request_ids)
        infer.b_seq_len = tensor(lengths)
        infer.b1_cu_q_seq_len = tensor([0, 1, 2])
        infer.req_manager = SimpleNamespace(req_to_token_indexs=tensor(mapping))
        backends = model.attention_prefill_backends if prefill else model.attention_decode_backends
        infer.attention_metadata = tuple(
            None if b is None else (b.create_att_prefill_state(infer) if prefill else b.create_att_decode_state(infer))
            for b in backends
        )
        infer.init_att_state()
        return infer

    mapping = [[0, 1, 6, 7, 4, 5], [4, 5, 2, 3, 0, 1]]

    def reference(state, q, infer):
        outputs = []
        for row, (req, length) in enumerate(zip(infer.b_req_idx.tolist(), infer.b_seq_len.tolist())):
            ids = infer.req_manager.req_to_token_indexs[req, :length].long()
            k = state.k_storage[0, ids].float().repeat_interleave(2, dim=1)
            v = state.v_storage[0, ids].float().repeat_interleave(2, dim=1)
            scores = torch.einsum("hd,thd->ht", q[row].float(), k) / state.config.head_dim ** 0.5
            outputs.append(torch.einsum("ht,thd->hd", scores.softmax(-1), v))
        return torch.stack(outputs).to(q.dtype)

    prefill = batch([3, 5], [0, 1], mapping, prefill=True)
    for layer, state, q in zip(model.layers_infer, states, qs):
        out = layer._context_attention_kernel(q, None, prefill, None)
        torch.testing.assert_close(out, reference(state, q, prefill), atol=0.02, rtol=0.02)
    capture = batch([3, 5], [0, 1], mapping)
    selected = [capture.attention_metadata[layer.attention.decode_slot] for layer in model.layers_infer]
    assert selected[0] is not selected[1]
    assert selected[0].kv_indices is selected[1].kv_indices
    for _ in range(2):
        for layer, q in zip(model.layers_infer, qs):
            layer._token_attention_kernel(q, capture, None)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = [layer._token_attention_kernel(q, capture, None) for layer, q in zip(model.layers_infer, qs)]
    for lengths, reqs in (([5, 2], [1, 0]), ([1, 6], [0, 1])):
        live = batch(lengths, reqs, list(reversed(mapping)), replay=True)
        for old, new in zip(capture.attention_metadata, live.attention_metadata):
            if old is not None:
                old.copy_for_decode_cuda_graph(new)
        for q in qs:
            q.normal_()
        graph.replay()
        for out, state, q in zip(outputs, states, qs):
            torch.testing.assert_close(out, reference(state, q, live), atol=0.02, rtol=0.02)
