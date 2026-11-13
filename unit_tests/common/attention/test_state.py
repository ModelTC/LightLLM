from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention_state.bounded_att import ConvSSMStateConfig, GDNState, KDAState
from lightllm.common.attention_state.commit import RequestGenerationStamp
from lightllm.common.attention_state.manager import AttentionStateManager


def test_commit_owns_mapping_and_detects_recycled_requests():
    requests = torch.tensor([2, 2, 4], dtype=torch.int32)
    steps = torch.tensor([0, 1, 0], dtype=torch.int32)
    generations = torch.zeros(8, dtype=torch.int64)
    output = RequestGenerationStamp.stamp("main", requests, steps, generations)
    requests.fill_(6)
    steps.fill_(5)
    assert output.request_indices.tolist() == [2, 2, 4]
    assert output.mtp_indices.tolist() == [0, 1, 0]
    output.validate_generation(generations)
    generations[2] += 1
    with pytest.raises(RuntimeError, match="reused request slot"):
        output.validate_generation(generations)


def test_overlap_commit_preserves_microbatch_order_and_role():
    generation = torch.zeros(8, dtype=torch.int64)
    a = RequestGenerationStamp.stamp("main", torch.tensor([4, 4]), torch.tensor([0, 1]), generation)
    b = RequestGenerationStamp.stamp("main", torch.tensor([2]), torch.tensor([0]), generation)
    merged = RequestGenerationStamp.concatenate((a, b))
    assert merged.request_indices.tolist() == [4, 4, 2]
    assert merged.mtp_indices.tolist() == [0, 1, 0]
    with pytest.raises(ValueError, match="same model role"):
        RequestGenerationStamp.concatenate((a, replace(b, role="draft0")))


@pytest.mark.parametrize("state_cls", [GDNState, KDAState])
def test_capture_snapshot_restores_only_participating_role_and_requests(state_cls):
    config = ConvSSMStateConfig(
        logical_layer_ids=(0,),
        global_k_heads=1,
        global_v_heads=1,
        key_dim=2,
        value_dim=2,
        conv_width=2,
        conv_dtype=torch.bfloat16,
        ssm_dtype=torch.float32,
    )
    manager = AttentionStateManager(request_capacity=3)
    main = state_cls(config, request_capacity=3, mtp_step=2, device="cpu")
    draft = state_cls(config, request_capacity=3, mtp_step=2, device="cpu")
    manager.register_state("main", main)
    manager.register_state("draft0", draft)
    manager.finish_state_registration()
    main.conv_state.fill_(2)
    main.ssm_state.fill_(3)
    main.mtp_state_index.fill_(1)
    snapshot = manager.snapshot_capture_state(torch.tensor([1, 1]), role="main")
    main.conv_state[:, 1].fill_(8)
    main.ssm_state[:, 3:6].fill_(9)
    main.mtp_state_index[1] = 2
    main.conv_state[:, 2].fill_(6)
    draft.conv_state.fill_(7)
    manager.restore_capture_state(snapshot)
    assert torch.all(main.conv_state[:, 1] == 2)
    assert torch.all(main.ssm_state[:, 3:6] == 3)
    assert main.mtp_state_index[1] == 1
    assert torch.all(main.conv_state[:, 2] == 6)
    assert torch.all(draft.conv_state == 7)


@pytest.mark.parametrize("kind", ["mla", "packed_bf16", "packed_group128_fp8", "packed_group64_e8m0"])
def test_mla_adapter_binds_state_layout(kind):
    from lightllm.common.attention.api import AttentionFamily, AttentionPhase
    from lightllm.common.attention_state.full_att import (
        MLAConfig,
        MLAFullAttentionState,
        PackedMLALogicalConfig,
        create_packed_mla_state,
    )
    from lightllm.common.basemodel.attention_adapters import ADAPTERS, AttentionDescription
    from unit_tests.common.attention.helpers import model_stub

    nope = kind == "packed_group64_e8m0"
    rope_dim = 0 if nope else 64
    latent_dim = 512 + rope_dim
    if kind == "mla":
        state = MLAFullAttentionState(MLAConfig(((2, 5),), latent_dim, torch.bfloat16), device="cpu")
    else:
        state = create_packed_mla_state(PackedMLALogicalConfig(((2, 5),), latent_dim, kind), device="cpu")
    manager = AttentionStateManager(request_capacity=2)
    manager.register_state("main", state)
    manager.finish_state_registration()
    model = model_stub(llm_kv_type="fp8kv_dsa" if "fp8" in kind or nope else "None")
    model.att_state_manager, model.data_type = manager, torch.bfloat16
    layer = SimpleNamespace(
        layer_num_=5,
        tp_q_head_num_=16,
        qk_nope_head_dim=512,
        kv_lora_rank=512,
        softmax_scale=latent_dim ** -0.5,
        network_config_={} if nope else {"qk_rope_head_dim": rope_dim},
    )
    if not nope:
        layer.qk_rope_head_dim = rope_dim
    family = AttentionFamily.MLA if kind == "mla" else AttentionFamily.SPARSE_MLA
    adapter = ADAPTERS[family]
    spec = adapter.describe(model, layer, AttentionPhase.PREFILL, AttentionDescription(family))
    assert state.storage is None
    state.allocate_storage(8)
    resources = adapter.bind(model, layer, None)
    assert spec.cache.dtype == resources.kv.dtype == state.storage_dtype
    assert resources.qk_rope_head_dim == rope_dim
    assert resources.kv.data_ptr() == state.layer_resources(2).data_ptr()
    assert resources.kv.shape[-1] == (584 if nope else 656 if kind == "packed_group128_fp8" else latent_dim)
