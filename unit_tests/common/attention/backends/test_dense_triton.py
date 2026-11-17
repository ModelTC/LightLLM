import weakref
from types import SimpleNamespace
from typing import Iterator

import pytest
import torch

from lightllm.common.attention.backend import TensorAllocator
from lightllm.common.attention.backends.dense import DenseInputs, DensePrefillBatch, DenseDecodeBatch
from lightllm.common.attention.backends.dense.triton import TritonDensePrefillBackend, TritonDenseDecodeBackend
from lightllm.common.basemodel.attention.base_att import AttControl
from lightllm.common.basemodel.attention.triton.fp import TritonAttBackend
from lightllm.common.basemodel.infer_struct import InferStateInfo


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.fixture(params=["torch", "serving"])
def allocator(request: pytest.FixtureRequest) -> Iterator[TensorAllocator]:
    if request.param == "torch":
        yield torch.empty
        return

    from lightllm.common.basemodel.layer_infer.base_layer_infer import BaseLayerInfer
    from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager

    g_cache_manager.cache_env_in()
    try:
        yield BaseLayerInfer.__new__(BaseLayerInfer).alloc_tensor
    finally:
        g_cache_manager.cache_env_out()


def reference(q, k, v, token_map, req_ids, lengths, ready, window):
    outputs = []
    offset = 0
    for req_id, length, cached in zip(req_ids.tolist(), lengths.tolist(), ready):
        count = length - cached
        indices = token_map[req_id, :length].long()
        keys = k[indices].float().repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        values = v[indices].float().repeat_interleave(q.shape[1] // v.shape[1], dim=1)
        scores = torch.einsum("qhd,khd->hqk", q[offset : offset + count].float(), keys) * q.shape[-1] ** -0.5
        query_pos = torch.arange(cached, length, device=q.device)[:, None]
        key_pos = torch.arange(length, device=q.device)[None, :]
        mask = key_pos <= query_pos
        if window != (-1, -1):
            mask &= key_pos >= query_pos - window[0]
        scores.masked_fill_(~mask[None, :, :], float("-inf"))
        outputs.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), values))
        offset += count
    return torch.cat(outputs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("q_heads,kv_heads,window", [(4, 4, (-1, -1)), (8, 2, (-1, -1)), (8, 2, (15, 0))])
@pytest.mark.parametrize("cached", [False, True])
def test_prefill_decode(dtype, q_heads, kv_heads, window, cached):
    torch.manual_seed(42)
    lengths = torch.tensor([33, 9, 259], device="cuda", dtype=torch.int32)
    ready = [16, 5, 128] if cached else [0, 0, 0]
    token_map = torch.randperm(3 * 272, device="cuda", dtype=torch.int32).view(3, 272)
    req_ids = torch.tensor([2, 0, 1], device="cuda", dtype=torch.int32)
    kv = torch.randn(3 * 272, 2 * kv_heads, 64, device="cuda", dtype=dtype)
    before = kv.clone()
    k, v = kv[:, :kv_heads], kv[:, kv_heads:]
    query_lengths = lengths - torch.tensor(ready, device="cuda", dtype=torch.int32)
    q = torch.randn(int(query_lengths.sum()), q_heads + 1, 64, device="cuda", dtype=dtype)[:, :q_heads]
    q_before = q.clone()
    batch = DensePrefillBatch(
        req_to_token_indexs=token_map,
        b_req_idx=req_ids,
        b_seq_len=lengths,
        b_q_start_loc=query_lengths.cumsum(0, dtype=torch.int32) - query_lengths,
        b_ready_cache_len=torch.tensor(ready, device="cuda", dtype=torch.int32),
        max_q_seq_len=int(query_lengths.max()),
    )
    out = torch.empty(q.shape[0], q_heads + 1, 64, device=q.device, dtype=q.dtype)[:, :q_heads]
    inputs = DenseInputs(q=q, k_cache=k, v_cache=v, sliding_window=window)
    assert TritonDensePrefillBackend().forward(batch=batch, inputs=inputs, out=out) is out
    torch.testing.assert_close(
        out.float(), reference(q, k, v, token_map, req_ids, lengths, ready, window), atol=0.02, rtol=0.02
    )
    torch.testing.assert_close(q, q_before, atol=0, rtol=0)

    def allocate(shape: tuple[int, ...] | list[int], *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        assert device == q.device
        return torch.empty(shape, dtype=dtype, device=device)

    for _ in range(3):
        lengths += 1
        q = torch.randn(3, q_heads + 1, 64, device="cuda", dtype=dtype)[:, :q_heads]
        q_before = q.clone()
        batch = DenseDecodeBatch(
            req_to_token_indexs=token_map,
            b_req_idx=req_ids,
            b_seq_len=lengths,
            kv_seq_len_upper_bound=int(lengths.max()),
            tuning_kv_seq_len=int(lengths.max()),
        )
        out = torch.empty(3, q_heads + 1, 64, device=q.device, dtype=q.dtype)[:, :q_heads]
        inputs = DenseInputs(q=q, k_cache=k, v_cache=v, sliding_window=window)
        assert TritonDenseDecodeBackend().forward(batch=batch, inputs=inputs, out=out, alloc_func=allocate) is out
        torch.testing.assert_close(
            out.float(),
            reference(q, k, v, token_map, req_ids, lengths, (lengths - 1).tolist(), window),
            atol=0.02,
            rtol=0.02,
        )
        torch.testing.assert_close(q, q_before, atol=0, rtol=0)
    torch.testing.assert_close(kv, before, atol=0, rtol=0)


@pytest.mark.parametrize("q_heads,kv_heads", [(4, 4), (8, 2)])
def test_adapter_graph(q_heads, kv_heads, allocator):
    torch.manual_seed(43)
    owner = InferStateInfo()
    owner.batch_size = 2
    owner.b_req_idx = torch.tensor([1, 0], device="cuda", dtype=torch.int32)
    owner.b_seq_len = torch.tensor([33, 17], device="cuda", dtype=torch.int32)
    owner.max_kv_seq_len = 33
    token_map = torch.randperm(2 * 512, device="cuda", dtype=torch.int32).view(2, 512)
    owner.req_manager = SimpleNamespace(req_to_token_indexs=token_map)
    model = type("Model", (), {})()
    model.is_mtp_draft_model = False
    model.mtp_manager = SimpleNamespace(get_decode_draft_step=lambda _: 0)
    backend = TritonAttBackend(model)
    state = backend.create_att_decode_state(owner)
    state.init_state()
    owner.decode_att_state = state
    owner.is_cuda_graph_capture = True
    owner.max_kv_seq_len = 512
    q = torch.randn(2, q_heads, 64, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(1024, 2 * kv_heads, 64, device="cuda", dtype=q.dtype)
    k, v = kv[:, :kv_heads], kv[:, kv_heads:]

    def forward() -> torch.Tensor:
        return state.decode_att(q=q, k=k, v=v, att_control=AttControl(), alloc_func=allocator)

    forward()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = forward()
    for lengths in ([129, 65], [300, 257]):
        owner.b_seq_len.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        q.normal_()
        graph.replay()
        expected = reference(q, k, v, token_map, owner.b_req_idx, owner.b_seq_len, [x - 1 for x in lengths], (-1, -1))
        torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)
    assert state.decode_max_kv_seq_len == 33
    owner_ref = weakref.ref(owner)
    del owner
    assert owner_ref() is None
    with pytest.raises(ReferenceError):
        _ = state.infer_state
    del backend._instances[(type(backend), model)]
