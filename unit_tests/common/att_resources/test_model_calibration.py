"""Model-owned calibration slices must not depend on State initialization order."""

from __future__ import annotations

from unit_tests.common.att_resources.model_helpers import read_kv_storage

from typing import TYPE_CHECKING
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.att_resources import AttResourceManager
from lightllm.common.att_resources.full_att import FullAttentionState
from unit_tests.common.att_resources.model_helpers import owned_states
from lightllm.common.basemodel.basemodel_utils import init_model_full_attention
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.qwen3_5_mtp.model import Qwen3_5MTPModel
from lightllm.models.qwen3next.model import Qwen3NextTpPartModel
from lightllm.server.core.objs.start_args_type import StartArgs

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("encoding", ["fp8kv_sph", "fp8kv_spt"])
def test_calibration_slices_follow_model_configs_before_other_states_exist(tmp_path: Path, encoding: str):
    config = dict(
        architectures=["Qwen3NextForCausalLM"],
        num_hidden_layers=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        hidden_size=32,
        head_dim=16,
        full_attention_interval=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=3,
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    scales = torch.arange(1, 9, dtype=torch.float32) * 0.125
    calibration = dict(
        architectures="Qwen3NextForCausalLM",
        num_head=1,
        num_layers=8,
        scales_shape=[8, 2],
        scales=scales[:, None].expand(-1, 2).tolist(),
        q_calibration=dict(num_head=1, scales_shape=[8, 1], scales=(scales[:, None] * 2).tolist()),
        quant_type="per_head" if encoding == "fp8kv_sph" else "per_tensor",
    )
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration))
    args = StartArgs(
        model_dir=str(tmp_path),
        data_type="bfloat16",
        llm_kv_type=encoding,
        kv_quant_calibration_config_path=str(calibration_path),
        page_size=4,
        mtp_mode="vanilla_with_att",
    )
    definitions = (
        (Qwen3NextTpPartModel, config, 8, args, (3, 7), (0, 2)),
        (
            Qwen3NextTpPartModel,
            dict(config, num_hidden_layers=6, full_attention_interval=2),
            6,
            args,
            (9, 11, 13),
            (2, 5),
        ),
        (Qwen3NextTpPartModel, dict(config, num_hidden_layers=3), 3, args, (), (5, 5)),
        (
            LlamaTpPartModel,
            dict(config, num_hidden_layers=3),
            3,
            replace(args, mtp_mode="vanilla_no_att"),
            (),
            (5, 5),
        ),
        # Native MTP keeps the main config but its resource hook describes one full-attention layer.
        (Qwen3_5MTPModel, config, 1, args, (20,), (5, 6)),
        (LlamaTpPartModel, dict(config, num_hidden_layers=2), 2, args, (21, 22), (6, 8)),
    )
    models, layer_start = [], 0
    for cls, model_config, layer_count, model_args, _, _ in definitions:
        model = cls.__new__(cls)
        model.is_mtp_draft_model = bool(models)
        model.config, model.args = model_config.copy(), model_args
        model.tp_rank_, model.tp_world_size_ = 0, 1
        model.layers_infer = [SimpleNamespace(layer_num_=i) for i in range(layer_start, layer_start + layer_count)]
        if models:
            model.main_model, model.mtp_previous_draft_models = models[0], models[1:].copy()
        models.append(model)
        layer_start += layer_count

    # Prepare KV definitions in reverse: no preceding model has a manager or State yet.
    for index in reversed(range(len(models))):
        model = models[index]
        assert all(not hasattr(previous, "att_resource_manager") for previous in models[:index])
        model.att_resource_manager = AttResourceManager(request_capacity=1)
        init_model_full_attention(model)
        states = owned_states(model.att_resource_manager.__dict__, FullAttentionState)
        state = states[0] if states else None
        expected_layers, (start, end) = definitions[index][-2:]
        if not expected_layers:
            assert state is None
            continue
        assert state.config.logical_layer_groups == tuple((layer,) for layer in expected_layers)
        assert read_kv_storage(state) is None
        assert torch.equal(state.scales[:, 0], scales[start:end])
        if encoding == "fp8kv_sph":
            assert torch.equal(state.q_scales[:, 0], scales[start:end] * 2)
