from unit_tests.common.att_resources.model_helpers import build_hybrid_resources
import dataclasses
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.att_resources.allocators import FullAttKVAllocator

from lightllm.common.att_resources.full_att import (
    PackedMLAFullAttentionState,
    PackedMLAAttentionConfig,
)
from lightllm.common.req_manager import ReqManager
from lightllm.models.glm5_next.config import Glm5NextResourceConfig
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.envs_utils import get_env_start_args, set_env_start_args, set_unique_server_name


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.usefixtures("kv_allocator_environment"),
]


def test_nope_cache_config_uses_native_mla_width():
    config = Glm5NextResourceConfig.from_model_config(
        {
            "kv_lora_rank": 512,
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention"] * 3 + ["deepseek_sparse_attention"],
            "index_kpool": 4,
            "index_head_dim": 128,
            "linear_attn_config": {
                "num_heads": 8,
                "head_dim": 128,
                "short_conv_kernel_size": 4,
                "kda_layers": [0, 1, 2],
            },
        },
        StartArgs(tp=4, data_type="bfloat16"),
    )
    assert config.full_att_head_dim == 584
    assert config.full_att_head_dim * config.full_att_dtype.itemsize == 512 * 2 + 144


@pytest.mark.parametrize("small_page", [False, True])
@pytest.mark.parametrize("tp_world_size", [1, 4])
@pytest.mark.parametrize("mtp_step", [0, 2])
def test_hybrid_checkpoint_restore_and_packed_kv_copy(monkeypatch, small_page, tp_world_size, mtp_step):
    monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_NODE", "0")
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    monkeypatch.setattr("lightllm.common.req_manager.req_sampling_params.get_vocab_size", lambda _: 128)
    args = StartArgs(
        tp=tp_world_size,
        data_type="bfloat16",
        linear_att_hash_page_size=4,
        linear_att_page_block_num=2,
        cpu_cache_token_page_size=8,
        mtp_step=mtp_step,
        mtp_mode="eagle_with_att" if mtp_step else None,
    )
    set_unique_server_name(args)
    get_env_start_args.cache_clear()
    set_env_start_args(dataclasses.asdict(args))
    config = Glm5NextResourceConfig(
        tp_world_size=tp_world_size,
        full_att_all_num_kv_heads=1,
        full_att_dtype=torch.bfloat16,
        full_att_num_kv_heads=1,
        full_att_head_dim=584,
        global_linear_k_heads=2 * tp_world_size,
        global_linear_v_heads=2 * tp_world_size,
        num_linear_k_heads=2,
        num_linear_v_heads=2,
        head_linear_k_dim=128,
        head_linear_v_dim=128,
        conv_kernel_size=4,
        linear_layer_num=3,
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=torch.float32,
        full_attention_interval=4,
        all_layer_num=4,
        draft_full_att_kv_layer_num=int(mtp_step > 0),
    )
    full = PackedMLAFullAttentionState(
        PackedMLAAttentionConfig(
            tuple((layer,) for layer in config.get_full_att_logical_layer_ids()),
            1,
            584,
            torch.bfloat16,
            tp_world_size=tp_world_size,
            kv_dim=512,
        ),
         page_size=args.page_size,
    )
    full.initialize_storage(16)
    req = ReqManager(3, 16)
    from unit_tests.common.att_resources.model_helpers import create_full_attention_roles

    main_full, draft_full = create_full_attention_roles(full, config.all_layer_num)
    resources = build_hybrid_resources(
        config=config, request_capacity=4, mtp_step=mtp_step,
        full_state=main_full,
        full_allocator=FullAttKVAllocator(full.capacity, full.page_size),
        draft_full_state=draft_full,
        big_page_capacity=4,
        small_page_capacity=2,
    )
    full = main_full
    linear = resources.main_kda_conv_bf16_ssm_fp32_state
    tail = resources.main_kpool_c4_bf16_state
    assert not hasattr(req, "req_to_conv_state")
    att_kv = full.get_layer_view(3).kv
    assert att_kv.shape == (17, 1, 512)
    assert att_kv.stride(0) == 584
    index_bytes = full.get_layer_view(3).indexer_k
    index_bytes.random_(0, 256)
    expected_index_bytes = index_bytes.clone()
    new_kv = torch.randn(2, 1, 512, dtype=torch.bfloat16, device="cuda")
    destinations = torch.tensor([5, 9], dtype=torch.int32, device="cuda")
    full.write_full_att_kv(3, destinations, new_kv)
    assert torch.equal(att_kv[destinations], new_kv)
    assert torch.equal(index_bytes, expected_index_bytes)
    assert tail.get_layer_view(3).shape == (4, 4 + mtp_step, 256)
    if mtp_step:
        assert resources.draft0_kpool_c4_bf16_state.get_layer_view(4).shape == (4, 4 + mtp_step, 256)
        assert draft_full.get_layer_view(4).kv.shape == att_kv.shape
    kind = "small" if small_page else "big"
    allocator = resources.bounded_att_small_page_allocator if small_page else resources.bounded_att_big_page_allocator
    slot = allocator.alloc(1)[0]
    resources.reset_bounded_att_state([0])
    resources.reset_attention_tail_state([0])
    linear.conv_state[:, 0].normal_()
    linear.ssm_state[:, 0].normal_()
    conv = linear.conv_state[:, 0, ..., :3].clone()
    ssm = linear.ssm_state[:, 0].clone()
    resources.save_bounded_att_pages([0], page_kind=kind, slot_ids=[slot])
    torch.cuda.synchronize()
    linear.conv_state[:, 0].zero_()
    linear.ssm_state[:, 0].zero_()
    tail.tail_buffer.normal_()
    unchanged = tail.tail_buffer[:, [0, 1, 3]].clone()
    resources.restore_bounded_att_pages([2], page_kind=kind, slot_ids=[slot])
    resources.reset_attention_tail_state([2])
    torch.cuda.synchronize()
    assert torch.equal(linear.conv_state[:, 2, ..., :3], conv)
    assert torch.equal(linear.ssm_state[:, 2 * (mtp_step + 1)], ssm)
    # Both page sizes are aligned to complete pools. Restoring a prefix must
    # discard stale tail values from a previously allocated request slot.
    assert not tail.tail_buffer[:, 2].any()
    assert torch.equal(tail.tail_buffer[:, [0, 1, 3]], unchanged)
    tail.tail_buffer[:, 2].normal_()
    preserved_tail = tail.tail_buffer.clone()
    resources.reset_bounded_att_state([2])
    assert not linear.conv_state[:, 2].any()
    assert not linear.ssm_state[:, 2 * (mtp_step + 1) : 3 * (mtp_step + 1)].any()
    assert torch.equal(tail.tail_buffer, preserved_tail)
    resources.reset_attention_tail_state([2])
    assert not tail.tail_buffer[:, 2].any()
    assert torch.equal(tail.tail_buffer[:, [0, 1, 3]], unchanged)
    # KV moves carry MLA latents and pooled FP8 bytes; raw keys/gates only
    # belong to live requests. Bytewise equality checks FP8 scale integrity.
    packed_bytes = full.storage.view(torch.uint8)
    packed_bytes[:, 0].random_(0, 256)
    full.copy_full_att_tokens(torch.tensor([0], device="cuda"), torch.tensor([7], device="cuda"))
    assert torch.equal(packed_bytes[:, 0], packed_bytes[:, 7])


@pytest.mark.parametrize("gate_type", ["silu", "sigmoid"])
def test_gated_norm_gate_type_and_strided_gate(gate_type):
    from lightllm.common.basemodel.triton_kernel.norm.gated_rmsnorm import gated_rmsnorm_forward

    x = torch.randn(12, 128, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(3, 8, 128, device="cuda", dtype=torch.bfloat16)[:, :4]
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    actual = gated_rmsnorm_forward(x, weight, None, 1e-5, gate, gate_type=gate_type)
    expected = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5) * weight.float()
    z = gate.reshape(12, 128).float()
    expected *= z.sigmoid() if gate_type == "sigmoid" else torch.nn.functional.silu(z)
    torch.testing.assert_close(actual, expected.bfloat16(), atol=0.008, rtol=0.008)


@pytest.mark.parametrize("add_one", [False, True])
def test_clamped_swiglu_preserves_gpt_oss_default(add_one):
    from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import silu_and_mul_fwd

    x = torch.linspace(-25, 25, 2048, device="cuda", dtype=torch.bfloat16).view(4, 512)
    out = torch.empty(4, 256, device="cuda", dtype=torch.bfloat16)
    kwargs = {} if add_one else {"clamp_up_add_one": False}
    alpha, limit = (1.702, 7.0) if add_one else (1.0, 10.0)
    silu_and_mul_fwd(x, out, limit=limit, alpha=alpha, **kwargs)
    gate, up = x.float().chunk(2, -1)
    gate = gate.clamp(max=limit)
    gate = (gate * torch.sigmoid(alpha * gate)).bfloat16().float()
    expected = gate * (up.clamp(-limit, limit) + int(add_one))
    torch.testing.assert_close(out, expected.bfloat16(), atol=0.008, rtol=0.008)
