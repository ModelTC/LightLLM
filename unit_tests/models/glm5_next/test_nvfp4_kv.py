"""GLM NVFP4 backend keeps fresh/current KV precision and physical sparse indices."""

from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.attention.base_att import AttControl
from lightllm.common.basemodel.attention.nsa.nvfp4_glm5_next import NVFP4Glm5NextSparseAttBackend
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.basemodel.triton_kernel.kv_copy.nvfp4_mla import quantize_nvfp4_mla
from unit_tests.common.att_resources.test_nvfp4_mla import dequantize, reference


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("rows", [5, 17])
def test_fresh_and_cached_prefill_use_physical_indices_and_current_bf16(cached, rows):
    torch.manual_seed(53)
    q_counts = [rows // 2, rows - rows // 2]
    prefix = int(cached)
    kv_counts = [count + prefix for count in q_counts]
    total = sum(kv_counts)
    q = torch.randn(rows, 8, 512, device="cuda", dtype=torch.bfloat16)
    raw = torch.randn(total, 1, 512, device="cuda", dtype=q.dtype)
    packed = torch.zeros(64, 1, 432, device="cuda", dtype=torch.uint8)
    mem = torch.randperm(64, device="cuda", dtype=torch.int32)[:total]
    quantize_nvfp4_mla(raw, mem, packed[..., :288])
    table = torch.zeros(5, 64, device="cuda", dtype=torch.int32)
    table[3, : kv_counts[0]] = mem[: kv_counts[0]]
    table[1, : kv_counts[1]] = mem[kv_counts[0] :]
    state = InferStateInfo()
    state.max_cache_len = prefix
    state.total_token_num = total
    state.position_ids = torch.cat(
        [torch.arange(prefix, count, device="cuda", dtype=torch.int32) for count in kv_counts]
    )
    state.b_req_idx = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    state.b1_cu_q_seq_len = torch.tensor([0, q_counts[0], rows], device="cuda", dtype=torch.int32)
    state.b1_cu_kv_seq_len = torch.tensor([0, kv_counts[0], total], device="cuda", dtype=torch.int32)
    state.req_manager = SimpleNamespace(req_to_token_indexs=table)
    raw_rows = torch.cat(
        [
            torch.arange(prefix, kv_counts[0], device="cuda"),
            torch.arange(kv_counts[0] + prefix, total, device="cuda"),
        ]
    )
    state.mem_index = mem[raw_rows]
    model = type("Model", (), {"config": {"index_topk": 2048}})()
    backend = NVFP4Glm5NextSparseAttBackend(model)
    backend_state = backend.create_att_prefill_state(state)
    backend_state.init_state()
    if cached and rows > 8:
        assert torch.equal(backend_state.ragged_mem_index, mem)
    else:
        assert backend_state.ragged_mem_index is None
    relative = torch.full((rows, 2176), -1, device="cuda", dtype=torch.int32)
    physical = relative.clone()
    query = 0
    for request, count in enumerate(q_counts):
        start = 0 if request == 0 else kv_counts[0]
        for offset in range(count):
            length = prefix + offset + 1
            relative[query, :length] = torch.arange(length, device="cuda", dtype=torch.int32)
            physical[query, :length] = mem[start : start + length]
            query += 1
    current = raw[raw_rows].clone().add_(0.25) if cached else raw[raw_rows]
    control = AttControl(
        nsa_prefill=True,
        nsa_prefill_dict=dict(
            topk_mem_indices=physical,
            topk_indices=relative,
            prefill_cache_kv=current,
            softmax_scale=0.0625,
            kv_lora_rank=512,
        ),
    )
    out = backend_state.prefill_att(q, packed[..., :288], None, control)
    values = dequantize(packed)
    values[state.mem_index.long()] = current
    expected = reference(q, values, physical, 0.0625)
    torch.testing.assert_close(out, expected, atol=0.016, rtol=0.02)
    if cached:
        backend_state.current_row_table = None
        torch.cuda.synchronize()
        # Match the layer template's shape-only graph, which is never replayed.
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            backend_state.prefill_att(q, packed[..., :288], None, control)
        assert backend_state.current_row_table is None
        out = backend_state.prefill_att(q, packed[..., :288], None, control)
        torch.testing.assert_close(out, expected, atol=0.016, rtol=0.02)


@pytest.mark.parametrize("kv_type", ["nvfp4kv", "fp8kv_dsa"])
@pytest.mark.parametrize("dtype,latent", [(torch.float16, 512), (torch.bfloat16, 256)])
def test_quantized_glm_rejects_incompatible_compute_or_latent(kv_type, dtype, latent):
    from lightllm.models.glm5_next.model import Glm5NextTpPartModel
    from lightllm.common.basemodel.basemodel import TpPartBaseModel

    model = object.__new__(Glm5NextTpPartModel)
    model.config = dict(qk_rope_head_dim=0, kv_lora_rank=latent)
    model.data_type = dtype
    model.args = SimpleNamespace(llm_kv_type=kv_type, enable_tpsp_mix_mode=False)
    # The base validation is unrelated to this format; exercise the model's
    # additional constraints without constructing GPU weights or a service.
    from unittest.mock import patch

    with patch.object(TpPartBaseModel, "_verify_params", return_value=None):
        with pytest.raises(ValueError, match="BF16 computation and kv_lora_rank=512"):
            Glm5NextTpPartModel._verify_params(model)
