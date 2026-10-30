from __future__ import annotations

from unit_tests.common.att_resources.model_helpers import full_state_test_buffers

from unit_tests.common.att_resources.model_helpers import read_kv_storage

from unit_tests.common.att_resources.model_helpers import rebuild_full_state

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.att_resources import AttResourceManager
from lightllm.common.att_resources.allocators import FullAttKVAllocator
from lightllm.common.basemodel.attention.base_att import BaseAttBackend
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.basemodel.layer_infer.template.transformer_layer_infer_template import TransformerLayerInferTpl
from lightllm.common.basemodel.triton_kernel.kv_copy.prefill_compact_kv_flashmla_fp8 import (
    get_prefill_kv_cache_and_remap_indices_triton,
)
from unit_tests.common.att_resources.test_capacity import definition


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("kind", ["gqa", "static_head", "packed_glm"])
@pytest.mark.parametrize("role", ["main", "draft0"])
@pytest.mark.parametrize("capture", [False, True])
def test_transformer_writes_logical_layer_through_manager(kind: str, role: str, capture: bool):
    template = definition(kind)
    state = rebuild_full_state(template, config=replace(template.config, logical_layer_groups=((3,), (8,))), token_capacity=8, device="cuda")
    for storage_buffer in full_state_test_buffers(state).values():
        storage_buffer.zero_()
    manager = AttResourceManager(request_capacity=2)
    manager.set_att_state(role, state)
    manager.full_att_allocator = FullAttKVAllocator(state.capacity, state.page_size, pin_memory=False)
    manager.finish_state_creation()
    state = getattr(manager, f"{role}_{template.get_state_type()}_state")
    infer = InferStateInfo()
    infer.att_resource_manager = manager
    infer.is_draft_model = role == "draft0"
    infer.att_resource_role = role
    infer.mem_index = torch.tensor([1, 5], dtype=torch.int32, device="cuda")
    heads = 1 if kind == "packed_glm" else 2 * state.config.head_num
    dim = state.config.kv_dim if kind == "packed_glm" else state.config.head_dim
    kv = torch.full((2, heads, dim), 0.5, dtype=torch.bfloat16, device="cuda")
    layer = SimpleNamespace(layer_num_=8)
    TransformerLayerInferTpl._post_cache_kv(layer, kv, infer, None)
    if capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            TransformerLayerInferTpl._post_cache_kv(layer, kv, infer, None)
        for storage_buffer in full_state_test_buffers(state).values():
            storage_buffer.zero_()
        graph.replay()
    assert not hasattr(infer, "mem_manager")
    assert not hasattr(infer, "full_att_state")
    assert not read_kv_storage(state)[0].view(torch.uint8).any()
    view = state.get_layer_view(8)
    if kind == "packed_glm":
        assert torch.equal(view.kv[infer.mem_index], kv)
        assert not view.indexer_k.any()
    else:
        k, v = view
        actual = torch.cat((k[infer.mem_index], v[infer.mem_index]), dim=1)
        if kind == "static_head":
            actual = actual.view(torch.float8_e4m3fn).float() * state.scales[1][None, :, None]
            # Backends index fixed scales by physical row, even when model layer
            # IDs are sparse or include appended draft layers.
            assert BaseAttBackend._find_storage_layer_index(None, k, v, state) == 1
            assert state.cpu_scales.shape == (2, 4)
        assert torch.equal(actual, kv)


def test_packed_fp8_prefill_combines_history_and_fresh_kv_with_sparse_remapping():
    template = definition("packed_fp8")
    state = rebuild_full_state(template, config=template.config, token_capacity=8, device="cuda")
    for storage_buffer in full_state_test_buffers(state).values():
        storage_buffer.zero_()
    history_ids = torch.tensor([3, 7], dtype=torch.int32, device="cuda")
    history = torch.randn(2, 1, 576, dtype=torch.bfloat16, device="cuda")
    state.write_full_att_kv(0, history_ids, history)
    fresh = torch.full((1, 1, 576), 123, dtype=torch.bfloat16, device="cuda")
    topk = torch.tensor([[7, 5, 3, 7, -1], [5, -1, -1, -1, -1]], dtype=torch.int32, device="cuda")
    compact, mapped = get_prefill_kv_cache_and_remap_indices_triton(
        packed_kv=state.get_layer_view(0).kv,
        topk_mem_indices=topk,
        prefill_mem_index=torch.tensor([5], dtype=torch.int32, device="cuda"),
        prefill_cache_kv=fresh,
        prefill_dtype=fresh.dtype,
    )
    assert compact.shape == (3, 1, 576)
    assert torch.equal(mapped == -1, topk == -1)
    assert mapped[0, 0] == mapped[0, 3]
    assert torch.equal(compact[mapped[0, 1]], fresh[0])
    assert torch.equal(compact[mapped[1, 0]], fresh[0])
    torch.testing.assert_close(compact[mapped[0, 0], :, :512], history[1, :, :512], atol=0.15, rtol=0.08)
    torch.testing.assert_close(compact[mapped[0, 2], :, :512], history[0, :, :512], atol=0.15, rtol=0.08)
    assert torch.equal(compact[mapped[0, 0], :, 512:], history[1, :, 512:])
    assert torch.equal(compact[mapped[0, 2], :, 512:], history[0, :, 512:])
