"""统一标定阶段不依赖 State 注册顺序，不在布局定义阶段加载参数。"""

from __future__ import annotations

from unit_tests.common.att_resources.model_helpers import read_kv_storage

from typing import TYPE_CHECKING
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.att_resources import AttResourceManager
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.qwen3_5_mtp.model import Qwen3_5MTPModel
from lightllm.models.qwen3next.model import Qwen3NextTpPartModel
from lightllm.server.core.objs.start_args_type import StartArgs

if TYPE_CHECKING:
    from pathlib import Path
    from easydict import EasyDict


@pytest.mark.parametrize("encoding", ["fp8kv_sph", "fp8kv_spt"])
def test_calibration_binds_all_model_states_after_creation(tmp_path: Path, encoding: str):
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

    # 反向创建、注册，标定仍按 main 和 draft 的数值顺序绑定。
    manager = AttResourceManager(request_capacity=1)
    for index in reversed(range(len(models))):
        model = models[index]
        model.att_resource_manager = manager
        states = []
        if not (model.is_mtp_draft_model and model.args.mtp_mode in ("vanilla_no_att", "eagle_no_att")):
            states = model._create_full_att_states(
                model_config=model.config,
                args=model.args,
                model_logical_layer_ids=tuple(layer.layer_num_ for layer in model.layers_infer),
                tp_rank=model.tp_rank_,
                tp_world_size=model.tp_world_size_,
            )
        for state in states:
            assert state.resource_id is None
            assert state.scales is None and state.q_scales is None
            manager.set_att_state(model.att_resource_role, state)
    manager.finish_state_creation()
    manager.load_full_att_calibration(args=args)
    for index, model in enumerate(models):
        states = manager.get_full_attention_states(model.att_resource_role)
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


@pytest.mark.parametrize("tp_world_size", [1, 4])
def test_calibration_reads_once_and_handles_shared_layers_numeric_roles_and_tp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tp_world_size: int,
):
    from builtins import open as original_open
    from lightllm.common.att_resources.full_att import FullAttentionConfig, create_gqa_state

    # 第 0 行属于普通精度的共享层；其余行分别属于 main、draft2、draft10 的 FP8 State。
    scales = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
    q_scales = scales[:, :2] + 20
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(dict(
        num_layers=4,
        num_head=2,
        quant_type="per_head",
        scales_shape=[4, 4],
        scales=scales.tolist(),
        q_calibration=dict(num_head=2, scales_shape=[4, 2], scales=q_scales.tolist()),
    )))
    args = StartArgs(kv_quant_calibration_config_path=str(calibration_path))
    reads = []

    def count_open(file: str, *args: object, **kwargs: object):
        if file == str(calibration_path):
            reads.append(file)
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", count_open)
    for tp_rank in range(tp_world_size):
        manager = AttResourceManager(request_capacity=1)
        expected = []
        # 特意打乱角色及同角色 State 的注册顺序。
        for role, groups, row, encoding in (
            ("draft10", ((5,),), 3, "fp8kv_sph"),
            ("main", ((2, 3),), 1, "fp8kv_sph"),
            ("draft2", ((4,),), 2, "fp8kv_sph"),
            ("main", ((0, 1),), 0, "None"),
        ):
            state = create_gqa_state(
                config=FullAttentionConfig(
                    logical_layer_groups=groups,
                    global_head_num=4,
                    head_dim=16,
                    dtype=torch.bfloat16,
                    tp_rank=tp_rank,
                    tp_world_size=tp_world_size,
                ), kv_type=encoding, device="meta",
            )
            manager.set_att_state(role, state)
            if encoding != "None":
                expected.append((state, row))
        with pytest.raises(AssertionError, match="State creation"):
            manager.load_full_att_calibration(args=args)
        manager.finish_state_creation()
        manager.load_full_att_calibration(args=args)
        heads = 4 // tp_world_size
        local_heads = slice(tp_rank * heads, (tp_rank + 1) * heads)
        for state, row in expected:
            expected_k = scales[row:row + 1, :2].repeat_interleave(2, dim=-1)[:, local_heads]
            expected_v = scales[row:row + 1, 2:].repeat_interleave(2, dim=-1)[:, local_heads]
            expected_q = q_scales[row:row + 1].repeat_interleave(2, dim=-1)[:, local_heads]
            assert torch.equal(state.scales, torch.cat((expected_k, expected_v), dim=-1))
            assert torch.equal(state.q_scales, expected_q)
            assert state.k_storage is None
    assert len(reads) == tp_world_size  # 每个管理器只读取一次，不随 State 数量增加。


@pytest.mark.parametrize("encoding", ["fp8kv_spt", "fp8kv_sph"])
def test_cpu_layout_does_not_require_calibration_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cpu_args: EasyDict,
    encoding: str,
):
    from lightllm.utils.cpu_cache_layout import build_cpu_cache_layouts

    config = dict(
        model_type="llama",
        num_attention_heads=2,
        hidden_size=32,
        num_hidden_layers=2,
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    cpu_args.model_dir = str(tmp_path)
    cpu_args.llm_kv_type = encoding
    cpu_args.data_type = "bfloat16"
    cpu_args.kv_quant_calibration_config_path = str(tmp_path / "does-not-exist.json")
    monkeypatch.setattr("lightllm.utils.llm_utils.get_llm_model_class", lambda: LlamaTpPartModel)
    layouts = build_cpu_cache_layouts()
    assert len(layouts) == 1
    (layout,) = layouts.values()
    assert layout.byte_end > layout.byte_start
    assert {region.field for region in layout.regions} == {"k", "v"}


@pytest.mark.parametrize("encoding", ["fp8kv_spt", "fp8kv_sph"])
def test_deferred_calibration_keeps_unit_scale_policy(encoding: str):
    from lightllm.common.att_resources.full_att import FullAttentionConfig, create_gqa_state

    state = create_gqa_state(
        config=FullAttentionConfig(
            logical_layer_groups=((0, 1), (2,)),
            global_head_num=2,
            head_dim=16,
            dtype=torch.bfloat16,
        ), kv_type=encoding, device="cpu",
    )
    manager = AttResourceManager(request_capacity=1)
    manager.set_att_state("main", state)
    manager.finish_state_creation()
    with pytest.raises(AssertionError, match="calibration must be bound"):
        state.initialize_storage(4)
    if encoding == "fp8kv_sph":
        with pytest.raises(AssertionError, match="requires a calibration file"):
            manager.load_full_att_calibration(args=StartArgs())
    else:
        manager.load_full_att_calibration(args=StartArgs())
        assert torch.equal(state.scales, torch.ones((2, 2)))
        state.initialize_storage(4)
        assert state.k_storage.shape[0] == 2
