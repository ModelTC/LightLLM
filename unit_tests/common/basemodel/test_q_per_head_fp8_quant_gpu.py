import pytest
import torch

from lightllm.common.basemodel.triton_kernel.quantization.q_per_head_fp8_quant import (
    q_per_head_fp8_quant,
    ref_q_per_head_fp8_quant,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def test_grouped_q_fp8_quant_matches_reference_and_cuda_graph():
    # DSpark-style flattened query rows: one empty logical request and two
    # independent multi-token requests. The Q scale remains [request, KV head].
    torch.manual_seed(0)
    seq_lens = torch.tensor([0, 2, 3], device="cuda", dtype=torch.int32)
    q = torch.randn((5, 1, 16), device="cuda", dtype=torch.bfloat16)
    starts = torch.tensor([0, 0, 2, 5], device="cuda", dtype=torch.int32)
    batch_ids = torch.tensor([1, 1, 2, 2, 2], device="cuda", dtype=torch.int64)

    actual_q, actual_scale = q_per_head_fp8_quant(q, seq_lens, starts, token_batch_ids=batch_ids)
    ref_q, ref_scale = ref_q_per_head_fp8_quant(q, seq_lens)
    torch.cuda.synchronize()
    assert torch.equal(actual_q.view(torch.int8), ref_q.view(torch.int8))
    torch.testing.assert_close(actual_scale, ref_scale, atol=5e-6, rtol=1e-3)

    static_q = torch.empty_like(q)
    static_out = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    static_scales = torch.empty((3, 1), device="cuda", dtype=torch.float32)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, scales = q_per_head_fp8_quant(static_q, seq_lens, starts, static_scales, batch_ids)
        static_out.copy_(out)
    static_q.copy_(q)
    graph.replay()
    torch.cuda.synchronize()
    assert static_scales.shape == (3, 1)
    assert torch.equal(static_out.view(torch.int8), ref_q.view(torch.int8))
    torch.testing.assert_close(scales, ref_scale, atol=5e-6, rtol=1e-3)
