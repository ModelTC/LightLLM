from __future__ import annotations

from unit_tests.common.att_resources.model_helpers import bind_test_fp8_calibration

from unit_tests.common.att_resources.model_helpers import read_kv_storage

from collections.abc import Callable

import pytest
import torch

from lightllm.common.attention_state.full_att import (
    GQAAttentionConfig, MLAAttentionConfig, FullAttentionState, GQAFullAttentionState, GroupQuantizedFullAttentionState,
    StaticFP8FullAttentionState, MLAFullAttentionState, PackedMLAFullAttentionState,
    create_gqa_state, create_mla_state,
)
from lightllm.common.attention_state.bounded_att import (
    BoundedAttentionState, ConvSSMStateConfig, create_gdn_state, create_kda_state,
)
from lightllm.common.attention_state.bounded_att import WindowCompressorStateConfig, create_window_compressor_state


@pytest.mark.parametrize(
    "kv_type, expected_type, storage_dtype, width",
    [
        ("None", "gqa_bf16", torch.bfloat16, 128),
        ("int4kv", "gqa_int4_group16_bf16_scale", torch.int8, 64),
        ("int8kv", "gqa_int8_group16_bf16_scale", torch.int8, 128),
        ("fp8kv_spt", "gqa_fp8_static_per_tensor", torch.uint8, 128),
        ("fp8kv_sph", "gqa_fp8_static_per_head", torch.uint8, 128),
    ],
)
def test_gqa_creation_preserves_logical_tp_config_and_deferred_capacity(
    kv_type: str,
    expected_type: str,
    storage_dtype: torch.dtype,
    width: int,
):
    config = GQAAttentionConfig(
        logical_layer_groups=((3,), (7,)),
        global_head_num=8,
        head_dim=128,
        dtype=torch.bfloat16,
        tp_rank=2,
        tp_world_size=4,
    )
    scales = torch.full((2, 4 if kv_type == "fp8kv_sph" else 2), 0.5, dtype=torch.float32)
    q_scales = torch.ones((2, 2), dtype=torch.float32) if kv_type == "fp8kv_sph" else None
    state = create_gqa_state(
        config=config,
        kv_type=kv_type,
        page_size=4,
        device="meta",
        group_size=16,
    )
    assert state.get_state_type() == expected_type
    assert state.state_name is None
    assert state.config.logical_layer_groups == ((3,), (7,))
    assert state.config.global_head_num == 8 and state.config.head_num == 2
    assert state.config.tp_rank == 2 and state.config.tp_world_size == 4
    assert state.config.head_dim == config.head_dim and config.dtype == torch.bfloat16
    assert state.capacity == 0 and read_kv_storage(state) is None
    if kv_type.startswith("fp8"):
        assert state.scales is None and state.q_scales is None
        bind_test_fp8_calibration(state=state, scales=scales, q_scales=q_scales)
        assert torch.equal(state.scales, scales)
        if q_scales is not None:
            assert torch.equal(state.q_scales, q_scales)
    assert type(state.config).__bases__ == (object,)
    if kv_type in ("int4kv", "int8kv"):
        assert state.config.scale_dtype == config.dtype
    elif kv_type.startswith("fp8"):
        assert not hasattr(state.config, "dtype")
    state.device = torch.device("cpu")
    state.allocate_storage(8)
    assert state.k_storage.shape == state.v_storage.shape == (2, 12, 2, width)
    assert state.k_storage.untyped_storage().data_ptr() != state.v_storage.untyped_storage().data_ptr()
    assert state.k_storage.dtype == state.v_storage.dtype == storage_dtype
    if kv_type in ("int4kv", "int8kv"):
        assert state.k_scales.shape == state.v_scales.shape == (2, 12, 2, 8)
        assert state.k_scales.untyped_storage().data_ptr() != state.v_scales.untyped_storage().data_ptr()


@pytest.mark.parametrize(
    "packed, kv_type, kv_dim, row_width, expected_type",
    [
        (False, "None", 576, 576, "mla_bf16"),
        (True, "None", 576, 648, "mla_packed_bf16"),
        (True, "None", 512, 584, "mla_packed_bf16"),
        (True, "fp8kv_dsa", 576, 800, "mla_packed_fp8_group128"),
    ],
)
def test_mla_creation_computes_packed_width_from_logical_dimensions(
    packed: bool,
    kv_type: str,
    kv_dim: int,
    row_width: int,
    expected_type: str,
):
    config = MLAAttentionConfig(
        logical_layer_groups=((2,), (6,)),
        latent_dim=kv_dim,
        dtype=torch.bfloat16,
        tp_rank=1,
        tp_world_size=4,
    )
    state = create_mla_state(
        config=config,
        packed=packed,
        kv_type=kv_type,
        page_size=4,
        device="meta",
    )
    assert state.get_state_type() == expected_type
    assert state.config.latent_dim == config.latent_dim == kv_dim
    assert state.config.dtype == (torch.float8_e4m3fn if kv_type == "fp8kv_dsa" else config.dtype)
    assert not hasattr(state.config, "storage_dtype")
    assert not hasattr(state.config, "replicated")
    assert state.capacity == 0 and read_kv_storage(state) is None
    assert type(state.config).__bases__ == (object,)
    assert not hasattr(state.config, "head_num") and not hasattr(state.config, "global_head_num")
    state.device = torch.device("cpu")
    state.allocate_storage(8)
    assert read_kv_storage(state).shape == (2, 12, 1, row_width)
    assert read_kv_storage(state).dtype == (torch.uint8 if kv_type == "fp8kv_dsa" else config.dtype)
    if packed:
        # 保持旧算子读取的物理布局：FP8 indexer 紧接 KV，浮点格式保留中间的对齐间隙。
        expected_indexer_offset = 656 if kv_type == "fp8kv_dsa" else kv_dim * 2 + 12
        assert state.config.indexer_offset == expected_indexer_offset
        assert state.get_layer_view(2).indexer_k.shape[0] == 12


@pytest.mark.parametrize("create_state", [create_kda_state, create_gdn_state])
def test_bounded_creation_allocates_runtime_but_defers_pools(create_state: Callable[..., BoundedAttentionState]):
    config = ConvSSMStateConfig(
        logical_layer_ids=(0, 2),
        global_k_heads=2,
        global_v_heads=2,
        key_dim=8,
        value_dim=8,
        conv_width=3,
        conv_dtype=torch.bfloat16,
        ssm_dtype=torch.float32,
    )
    state = create_state(
        config=config,
        request_capacity=3,
        mtp_step=2,
        device="cpu",
    )
    assert state.config is config and state.state_name is None
    assert state.conv_state.shape == (2, 3, 48, 5)
    assert state.ssm_state.shape == (2, 9, 2, 8, 8)
    assert state.mtp_state_index.shape == (3,)
    assert state.big_pages_pool is None and state.small_pages_pool is None


def test_kpool_creation_preserves_pool_and_mtp_dimensions():
    config = WindowCompressorStateConfig(
        logical_layer_ids=(3, 7),
        window_size=4, compress_stride=4, target="indexer",
        candidate_dim=128,
        dtype=torch.bfloat16,
    )
    state = create_window_compressor_state(
        config=config,
        request_capacity=3,
        mtp_step=2,
        device="cpu",
    )
    assert state.config is config and state.state_name is None
    assert state.buffer.shape == (2, 3, 6, 256)
    assert state.get_state_type() == "compressor_indexer_w4_s4_bf16"


def test_create_rejects_encoding_from_another_logical_family():
    config = GQAAttentionConfig(
        logical_layer_groups=((0,),),
        global_head_num=1,
        head_dim=576,
        dtype=torch.bfloat16,
    )
    with pytest.raises(AssertionError, match="unsupported GQA encoding"):
        create_gqa_state(config=config, kv_type="fp8kv_dsa", device="meta")
    config = MLAAttentionConfig(
        logical_layer_groups=((0,),), latent_dim=576, dtype=torch.bfloat16,
    )
    with pytest.raises(AssertionError, match="unsupported packed MLA encoding"):
        create_mla_state(
            config=config,
            packed=True,
            kv_type="int4kv",
            device="meta",
        )
    with pytest.raises(AssertionError, match="unsupported MLA encoding"):
        create_mla_state(config=config, kv_type="fp8kv_dsa", device="meta")


@pytest.mark.parametrize(
    "state_class",
    [GQAFullAttentionState, GroupQuantizedFullAttentionState, StaticFP8FullAttentionState,
     MLAFullAttentionState, PackedMLAFullAttentionState],
)
def test_full_attention_implementations_are_independent(state_class: type[FullAttentionState]):
    # 具体方案只共享抽象契约，不继承其他具体方案的存储或操作实现。
    assert state_class.__bases__ == (FullAttentionState,)
    assert not state_class.__abstractmethods__
    assert FullAttentionState.__abstractmethods__ <= state_class.__dict__.keys()
