"""FP8 backends obtain calibration from the role's matching storage scheme."""

from __future__ import annotations

from unit_tests.common.att_resources.model_helpers import rebuild_full_state

from typing import TYPE_CHECKING
from dataclasses import replace
from types import SimpleNamespace
import weakref

import pytest
import torch

from lightllm.common.att_resources import AttResourceManager
from lightllm.common.att_resources.allocators import FullAttKVAllocator
from lightllm.common.basemodel.attention.base_att import BaseAttBackend
from lightllm.common.basemodel.attention.fa3 import fp8 as fa3
from lightllm.common.basemodel.attention.flashinfer import fp8 as flashinfer
from lightllm.common.basemodel.infer_struct import InferStateInfo
from unit_tests.common.att_resources.test_capacity import definition

if TYPE_CHECKING:
    from collections.abc import Sequence


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("kind", ["static_head", "static_tensor"])
@pytest.mark.parametrize("role", ["main", "draft0"])
@pytest.mark.parametrize("prefill", [False, True])
def test_fp8_kernel_receives_calibration_for_role_and_physical_layer(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    role: str,
    prefill: bool,
):
    template = definition(kind)
    states = []
    for value in (2, 5):
        scales = template.scales + value
        scales[1] += 10
        q_scales = None if kind == "static_tensor" else template.q_scales + value
        states.append(rebuild_full_state(
            template, config=replace(template.config, logical_layer_groups=((3,), (8,))), token_capacity=8, device="cuda",
            scales=scales, q_scales=q_scales,
        ))
    manager = AttResourceManager(request_capacity=2)
    manager.set_att_state("main", states[0])
    manager.set_att_state("draft0", states[1])
    manager.full_att_allocator = FullAttKVAllocator(8, 4, pin_memory=False)
    # Another encoding in the same role must not supply the FP8 calibration.
    plain = definition("gqa")
    plain = rebuild_full_state(plain, config=plain.config, token_capacity=8, device="cuda")
    manager.set_att_state(role, plain)
    manager.finish_state_creation()
    selected = getattr(manager, f"{role}_{template.get_state_type()}_state")
    k, v = selected.get_layer_view(8)
    infer = InferStateInfo()
    infer.att_resource_manager = manager
    infer.is_draft_model = role == "draft0"
    infer.att_resource_role = role
    infer.batch_size, infer.max_q_seq_len = 2, 1
    infer.b_seq_len = torch.ones(2, dtype=torch.int32, device="cuda")
    backend = object.__new__(BaseAttBackend)
    observed = {}

    if kind == "static_head":
        state_cls = fa3.Fp8Fa3PrefillAttState if prefill else fa3.Fp8Fa3DecodeAttState
        parent_cls = fa3.Fa3PrefillAttState if prefill else fa3.Fa3DecodeAttState
        monkeypatch.setattr(parent_cls, "init_state", lambda self: None)
        att = state_cls(backend=backend, _infer_state_weak_ref=weakref.ref(infer))
        att.b_att_seq_len = infer.b_seq_len
        att.init_state()

        def quantize(q: torch.Tensor, scales: torch.Tensor):
            observed["q_scales"] = scales
            return q

        def attention(**kwargs: object):
            observed.update(kwargs)
            return kwargs["q"]

        monkeypatch.setattr(fa3, "q_per_head_static_fp8_quant", quantize)
        monkeypatch.setattr(fa3, "flash_attn_with_kvcache", attention)
    else:
        state_cls = flashinfer.Fp8FlashInferPrefillAttState if prefill else flashinfer.Fp8FlashInferDecodeAttState
        att = state_cls(backend=backend, _infer_state_weak_ref=weakref.ref(infer))
        wrapper = SimpleNamespace(run=lambda *args, **kwargs: observed.update(kwargs))
        att.prefill_wrapper = att.decode_wrapper = wrapper

    q = torch.zeros((2, selected.config.head_num, selected.config.head_dim), dtype=torch.bfloat16, device="cuda")

    def allocate(shape: Sequence[int], dtype: torch.dtype, device: str | torch.device | int):
        return torch.empty(shape, dtype=dtype, device=device)

    if prefill:
        att.prefill_att(q, k, v, alloc_func=allocate)
    else:
        att.decode_att(q, k, v, alloc_func=allocate)
    if kind == "static_head":
        heads = selected.config.head_num
        torch.testing.assert_close(observed["q_scales"], selected.q_scales[1])
        torch.testing.assert_close(observed["k_descale"], selected.scales[1, :heads].expand(2, heads))
        torch.testing.assert_close(observed["v_descale"], selected.scales[1, heads:].expand(2, heads))
    else:
        assert observed["k_scale"] == selected.cpu_scales[1, 0]
        assert observed["v_scale"] == selected.cpu_scales[1, 1]
    assert not hasattr(infer, "full_att_state")


@pytest.mark.parametrize("head_num", [1, 3])
@pytest.mark.parametrize("scale_mode", ["per_tensor", "per_head"])
def test_fp8_split_storage_matches_calibrated_values_and_combined_kernel(head_num: int, scale_mode: str):
    from lightllm.common.att_resources.full_att import StaticFP8AttentionConfig, StaticFP8FullAttentionState
    from lightllm.common.basemodel.triton_kernel.destindex_copy_kv_fp8 import destindex_copy_kv_fp8

    scale_count = 2 if scale_mode == "per_tensor" else 2 * head_num
    scales = torch.arange(1, scale_count + 1, dtype=torch.float32, device="cuda").unsqueeze(0) / 4
    state = StaticFP8FullAttentionState(
        layout_config=StaticFP8AttentionConfig(
            logical_layer_groups=((5,),), global_head_num=head_num, head_dim=128, dtype=torch.bfloat16,
            scale_mode=scale_mode, calibration_id="split-storage",
        ),
        page_size=4, device="cuda", scales=scales,
        q_scales=torch.ones((1, head_num), dtype=torch.float32, device="cuda") if scale_mode == "per_head" else None,
    )
    state.initialize_storage(8)
    state.k_storage.zero_()
    state.v_storage.zero_()
    indices = torch.tensor([1, 6], dtype=torch.int32, device="cuda")
    kv = torch.randn((2, 2 * head_num, 128), dtype=torch.bfloat16, device="cuda")
    state.write_full_att_kv(logical_layer_id=5, token_indices=indices, kv=kv)
    k, v = state.get_layer_view(5)
    assert k.untyped_storage().data_ptr() != v.untyped_storage().data_ptr()
    actual = torch.cat((k[indices], v[indices]), dim=1)
    head_scales = scales[0].repeat_interleave(head_num) if scale_mode == "per_tensor" else scales[0]
    expected = (kv.float() / head_scales[None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    assert torch.equal(actual, expected.view(torch.uint8))
    assert not k[[0, 2, 3, 4, 5, 7, 8, 9, 10, 11]].any()
    assert not v[[0, 2, 3, 4, 5, 7, 8, 9, 10, 11]].any()

    # 算子的原合并输出入口仍应保持相同的量化结果。
    combined = torch.zeros((12, 2 * head_num, 128), dtype=torch.uint8, device="cuda")
    destindex_copy_kv_fp8(
        KV=kv, DestLoc=indices, scales=scales[0], Out=combined.view(torch.float8_e4m3fn),
        is_per_tensor_quant=scale_mode == "per_tensor",
    )
    assert torch.equal(combined[indices], actual)
