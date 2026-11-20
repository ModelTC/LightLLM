import weakref
from types import SimpleNamespace
from typing import Iterator

import pytest
import torch
import torch.nn.functional as F

from lightllm.common.attention.backend import TensorAllocator
from lightllm.common.attention.backends.gqa import GQAInputs
from lightllm.common.attention.backends.gqa.flashinfer import (
    FlashInferPrefillBatch,
    FlashInferDecodeBatch,
    FlashInferGQAPrefillBackend,
    FlashInferGQADecodeBackend,
)
from lightllm.common.basemodel.attention.flashinfer.fp import FlashInferAttBackend
from lightllm.common.basemodel.infer_struct import InferStateInfo


flashinfer = pytest.importorskip("flashinfer")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    token_map: torch.Tensor,
    req_ids: list[int],
    lengths: list[int],
    ready: list[int],
) -> torch.Tensor:
    outputs = []
    offset = 0
    for req_id, length, cached in zip(req_ids, lengths, ready):
        count = length - cached
        slots = token_map[req_id, :length].long()
        keys = k[slots].float().repeat_interleave(q.shape[1] // k.shape[1], dim=1).transpose(0, 1)
        values = v[slots].float().repeat_interleave(q.shape[1] // v.shape[1], dim=1).transpose(0, 1)
        mask = torch.arange(length, device=q.device)[None, :] <= torch.arange(cached, length, device=q.device)[:, None]
        outputs.append(
            F.scaled_dot_product_attention(
                q[offset : offset + count].float().transpose(0, 1), keys, values, attn_mask=mask
            ).transpose(0, 1)
        )
        offset += count
    return torch.cat(outputs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("q_heads,kv_heads", [(4, 4), (8, 2)])
@pytest.mark.parametrize("page_size", [1, 16, 64])
@pytest.mark.parametrize("cached", [False, True])
def test_prefill_decode(dtype: torch.dtype, q_heads: int, kv_heads: int, page_size: int, cached: bool) -> None:
    torch.manual_seed(44)
    lengths = [33, 9, 259]
    ready = [16, 5, 128] if cached else [0, 0, 0]
    req_ids = [2, 0, 1]
    page_map = torch.randperm(3 * 512 // page_size, device="cuda", dtype=torch.int32).view(3, -1)
    token_map = (page_map[:, :, None] * page_size + torch.arange(page_size, device="cuda")).reshape(3, 512)
    page_counts = [(length + page_size - 1) // page_size for length in lengths]
    indptr = torch.tensor([0, *page_counts], device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    indices = torch.cat([page_map[req, :count] for req, count in zip(req_ids, page_counts)])
    last_page_len = torch.tensor([(length - 1) % page_size + 1 for length in lengths], device="cuda", dtype=torch.int32)
    query_lengths = [length - cached for length, cached in zip(lengths, ready)]
    q_starts = torch.tensor([0, *query_lengths], device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    q = torch.randn(sum(query_lengths), q_heads, 64, device="cuda", dtype=dtype)
    kv = torch.randn(3 * 512, 2 * kv_heads, 64, device="cuda", dtype=dtype)
    k, v = kv[:, :kv_heads], kv[:, kv_heads:]
    before = kv.clone()
    q_before = q.clone()
    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    prefill = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
    prefill.plan(
        qo_indptr=q_starts,
        paged_kv_indptr=indptr,
        paged_kv_indices=indices,
        paged_kv_last_page_len=last_page_len,
        num_qo_heads=q_heads,
        num_kv_heads=kv_heads,
        head_dim_qk=64,
        page_size=page_size,
        causal=True,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    out = torch.empty_like(q)
    assert (
        FlashInferGQAPrefillBackend().forward(
            batch=FlashInferPrefillBatch(wrapper=prefill, page_size=page_size),
            inputs=GQAInputs(q=q, k_cache=k, v_cache=v),
            out=out,
        )
        is out
    )
    torch.testing.assert_close(
        out.float(), reference(q, k, v, token_map, req_ids, lengths, ready), atol=0.02, rtol=0.02
    )
    torch.testing.assert_close(q, q_before, atol=0, rtol=0)

    decode = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD", use_tensor_cores=True)
    decode.plan(
        indptr=indptr,
        indices=indices,
        last_page_len=last_page_len,
        num_qo_heads=q_heads,
        num_kv_heads=kv_heads,
        head_dim=64,
        page_size=page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    q = torch.randn(3, q_heads, 64, device="cuda", dtype=dtype)
    q_before = q.clone()
    out = torch.empty_like(q)
    assert (
        FlashInferGQADecodeBackend().forward(
            batch=FlashInferDecodeBatch(wrapper=decode, page_size=page_size),
            inputs=GQAInputs(q=q, k_cache=k, v_cache=v),
            out=out,
        )
        is out
    )
    torch.testing.assert_close(
        out.float(), reference(q, k, v, token_map, req_ids, lengths, [x - 1 for x in lengths]), atol=0.02, rtol=0.02
    )
    torch.testing.assert_close(q, q_before, atol=0, rtol=0)
    torch.testing.assert_close(kv, before, atol=0, rtol=0)


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


@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_adapter_graph(page_size: int, dtype: torch.dtype, allocator: TensorAllocator) -> None:
    torch.manual_seed(45)
    page_map = torch.randperm(2 * 512 // page_size, device="cuda", dtype=torch.int32).view(2, -1)
    token_map = (page_map[:, :, None] * page_size + torch.arange(page_size, device="cuda")).reshape(2, 512)
    model = type("Model", (), {})()
    model.args = SimpleNamespace(page_size=page_size)
    model.config = {"num_attention_heads": 8, "num_key_value_heads": 2, "hidden_size": 512}
    model.max_seq_length = 512
    model.graph_max_batch_size = 2
    model.graph_max_len_in_batch = 512
    model.data_type = dtype
    backend = FlashInferAttBackend(model=model)

    def owner(lengths: list[int], replay: bool = False) -> InferStateInfo:
        state = InferStateInfo()
        state.input_ids = torch.zeros(2, device="cuda", dtype=torch.int64)
        state.batch_size = 2
        state.b_req_idx = torch.tensor([1, 0], device="cuda", dtype=torch.int32)
        state.b_seq_len = torch.tensor(lengths, device="cuda", dtype=torch.int32)
        state.max_kv_seq_len = max(lengths)
        state.microbatch_index = 0
        state.req_manager = SimpleNamespace(req_to_token_indexs=token_map)
        state.is_cuda_graph_capture = not replay
        state.is_cuda_graph_replay = replay
        return state

    captured_owner = owner([33, 17])
    captured = backend.create_att_decode_state(infer_state=captured_owner)
    captured.init_state()
    q = torch.randn(2, 8, 64, device="cuda", dtype=dtype)
    kv = torch.randn(1024, 4, 64, device="cuda", dtype=dtype)
    k, v = kv[:, :2], kv[:, 2:]
    kv_before = kv.clone()
    wrapper = captured.decode_wrapper
    pointers = {
        name: getattr(captured, name).data_ptr() for name in ("kv_starts", "kv_indices", "kv_last_page_len_buffer")
    }

    def forward() -> torch.Tensor:
        return captured.decode_att(q=q, k=k, v=v, alloc_func=allocator)

    forward()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = forward()
    for lengths in ([129, 65], [300, 257]):
        runtime_owner = owner(lengths, replay=True)
        runtime = backend.create_att_decode_state(infer_state=runtime_owner)
        runtime.init_state()
        assert runtime.decode_wrapper is None
        captured.copy_for_decode_cuda_graph(new_state=runtime)
        assert captured.decode_wrapper is wrapper
        for name, pointer in pointers.items():
            assert getattr(captured, name).data_ptr() == pointer
        q.normal_()
        q_before = q.clone()
        graph.replay()
        expected = reference(q, k, v, token_map, [1, 0], lengths, [x - 1 for x in lengths])
        torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)
        torch.testing.assert_close(q, q_before, atol=0, rtol=0)
    torch.testing.assert_close(kv, kv_before, atol=0, rtol=0)
    owner_ref = weakref.ref(captured_owner)
    del captured_owner
    assert owner_ref() is None
    with pytest.raises(ReferenceError):
        _ = captured.infer_state
    del backend._instances[(type(backend), model)]
