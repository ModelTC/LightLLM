from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.attention.contracts import ResourceId
from lightllm.common.basemodel.attention.mla import BoundMla, MlaDecodeInput, MlaPrefillInput
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.server.core.objs.start_args_type import StartArgs


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("provider", ["triton", "fa3"])
def test_mla_bound_inputs_prefill_decode_and_graph(provider, monkeypatch):
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    if provider == "triton":
        from lightllm.common.basemodel.attention.triton.mla import MlaTritonAttBackend as Backend
    else:
        from lightllm.common.basemodel.attention.fa3.mla import MlaFa3AttBackend as Backend

    torch.manual_seed(5)
    device, dtype, heads = "cuda", torch.bfloat16, 16
    alloc = lambda shape, dtype, device=device: torch.empty(shape, dtype=dtype, device=device)
    infer = InferStateInfo()
    infer.batch_size, infer.total_token_num = 2, 8
    infer.microbatch_index, infer.max_q_seq_len, infer.max_kv_seq_len = 0, 3, 5
    infer.b_req_idx = torch.tensor([0, 1], dtype=torch.int32, device=device)
    infer.b_seq_len = torch.tensor([3, 5], dtype=torch.int32, device=device)
    infer.b_ready_cache_len = torch.tensor([1, 2], dtype=torch.int32, device=device)
    infer.b_q_start_loc = torch.tensor([0, 2], dtype=torch.int32, device=device)
    infer.b1_cu_q_seq_len = torch.tensor([0, 2, 5], dtype=torch.int32, device=device)
    infer.b1_cu_kv_seq_len = torch.tensor([0, 3, 8], dtype=torch.int32, device=device)
    infer.req_manager = SimpleNamespace(
        req_to_token_indexs=torch.tensor([[0, 4, 8, 1, 2], [2, 6, 10, 3, 7]], dtype=torch.int32, device=device)
    )

    class Model(SimpleNamespace):
        __hash__ = object.__hash__
        __eq__ = object.__eq__

    model = Model(
        args=StartArgs(page_size=1, running_max_req_size=2),
        graph_max_batch_size=4,
        max_seq_length=1024,
        tp_world_size_=1,
        graph=None,
        req_manager=infer.req_manager,
        is_mtp_draft_model=False,
        mtp_manager=SimpleNamespace(get_decode_draft_step=lambda _: 0),
    )
    backend = Backend(model)
    cache = torch.randn(1030, 1, 576, device=device, dtype=dtype)
    binding = BoundMla(ResourceId("main", 3), cache, 192 ** -0.5, 64)
    q = torch.randn(5, heads, 192, device=device, dtype=dtype)
    k_nope = torch.randn(8, heads, 128, device=device, dtype=dtype)
    k_rope = torch.randn(8, 1, 64, device=device, dtype=dtype)
    v = torch.randn(8, heads, 128, device=device, dtype=dtype)
    metadata = backend.create_att_prefill_state(infer)
    metadata.init_state()
    actual = binding.prefill(MlaPrefillInput(q, k_nope, k_rope, v), (metadata,), alloc)
    expected, q_start, kv_start = [], 0, 0
    for length, ready in ((3, 1), (5, 2)):
        key = torch.cat(
            (k_nope[kv_start : kv_start + length], k_rope[kv_start : kv_start + length].expand(-1, heads, -1)), -1
        ).float()
        for offset in range(length - ready):
            scores = (
                torch.einsum("hd,thd->ht", q[q_start + offset].float(), key[: ready + offset + 1])
                * binding.softmax_scale
            )
            expected.append(
                torch.einsum("ht,thd->hd", scores.softmax(-1), v[kv_start : kv_start + ready + offset + 1].float())
            )
        q_start += length - ready
        kv_start += length
    torch.testing.assert_close(actual, torch.stack(expected).to(dtype), atol=0.03, rtol=0.03)

    infer.max_q_seq_len = 1
    infer.max_kv_seq_len = 1024
    infer.total_token_num = 770
    infer.b_seq_len.copy_(torch.tensor([257, 513], device=device))
    infer.req_manager.req_to_token_indexs = torch.stack(
        (torch.randperm(1030, device=device)[:1024], torch.randperm(1030, device=device)[:1024])
    ).to(torch.int32)
    infer.b1_cu_q_seq_len = torch.arange(3, dtype=torch.int32, device=device)
    inputs = MlaDecodeInput(
        torch.randn(2, heads, 512, device=device, dtype=dtype), torch.randn(2, heads, 64, device=device, dtype=dtype)
    )
    metadata = backend.create_att_decode_state(infer)
    metadata.init_state()
    for _ in range(3):
        binding.decode(inputs, (metadata,), alloc)
    graph = torch.cuda.CUDAGraph()
    metadata.prepare_for_capture()
    with torch.cuda.graph(graph):
        actual = binding.decode(inputs, (metadata,), alloc)
    for lengths in ((1, 769), (513, 257)):
        infer.b_seq_len.copy_(torch.tensor(lengths, device=device))
        inputs.q_nope.copy_(torch.randn_like(inputs.q_nope))
        inputs.q_rope.copy_(torch.randn_like(inputs.q_rope))
        cache.copy_(torch.randn_like(cache))
        graph.replay()
        expected = []
        for batch, length in enumerate(lengths):
            kv = cache[infer.req_manager.req_to_token_indexs[batch, :length].long(), 0].float()
            query = torch.cat((inputs.q_nope[batch], inputs.q_rope[batch]), -1).float()
            scores = query @ kv.T * binding.softmax_scale
            expected.append(scores.softmax(-1) @ kv[:, :512])
        torch.testing.assert_close(actual, torch.stack(expected).to(dtype), atol=0.03, rtol=0.03)
