"""Exercise shared-manager startup with independently prepared hybrid drafts."""

from __future__ import annotations

from unit_tests.common.att_resources.model_helpers import full_state_test_buffers

from unit_tests.common.att_resources.model_helpers import read_kv_storage

from typing import TYPE_CHECKING
from contextlib import nullcontext
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch

from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.qwen3next.model import Qwen3NextTpPartModel
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.server.router.model_infer.mode_backend import base_backend
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.utils.att_resource_config import build_cpu_cache_layouts
from lightllm.utils.envs_utils import get_added_mtp_kv_layer_num, get_env_start_args

if TYPE_CHECKING:
    from lightllm.common.basemodel.basemodel import TpPartBaseModel
    from pathlib import Path


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_backend_rejects_more_than_36_drafts_before_loading():
    backend = ModeBackend.__new__(ModeBackend)
    backend.args = StartArgs(mtp_mode="vanilla_with_att", mtp_step=37)
    with pytest.raises(AssertionError, match="at most 36"):
        backend.init_mtp_draft_model({})
    assert backend.draft_models == []


@pytest.mark.parametrize("main_linear", [False, True])
@pytest.mark.parametrize("draft_full", [False, True])
@pytest.mark.parametrize("draft_count", [1, 2])
@pytest.mark.parametrize("encoding", ["None", "fp8kv_sph", "fp8kv_spt"])
@pytest.mark.parametrize("via_backend", [False, True])
def test_backend_allocates_all_roles_before_warmup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    main_linear: bool,
    draft_full: bool,
    draft_count: int,
    encoding: str,
    via_backend: bool,
):
    args = StartArgs(
        model_dir=str(tmp_path),
        data_type="bfloat16",
        llm_kv_type=encoding,
        page_size=4,
        linear_att_hash_page_size=4,
        linear_att_page_block_num=2,
        linear_att_cache_size=2,
        cpu_cache_token_page_size=8,
        mtp_mode="vanilla_with_att",
        mtp_step=draft_count,
    )
    for key in ("CURRENT_RANK_IN_DP", "CURRENT_RANK_IN_NODE", "CURRENT_DEVICE_ID"):
        monkeypatch.setenv("LIGHTLLM_" + key, "0")
    monkeypatch.setenv("LIGHTLLM_DP_WORLD_SIZE", "1")
    config = dict(
        model_type="qwen3_5_text",
        architectures=["Qwen3NextForCausalLM"],
        num_hidden_layers=4,
        n_layer=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=64,
        head_dim=16,
        full_attention_interval=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=3,
    )
    draft_config = dict(
        config,
        num_hidden_layers=2 if draft_full else 1,
        n_layer=2 if draft_full else 1,
        num_key_value_heads=1,
        head_dim=8,
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    draft_dir = tmp_path / "draft"
    draft_dir.mkdir()
    (draft_dir / "config.json").write_text(json.dumps(draft_config))
    args.mtp_draft_model_dir = [str(draft_dir)] * draft_count
    main_full_layers = 2 if main_linear else 4
    if encoding != "None":
        calibration_layers = main_full_layers + int(draft_full) * draft_count
        scales = torch.arange(1, calibration_layers + 1, dtype=torch.float32) * 0.125
        calibration = {
            "architectures": "Qwen3NextForCausalLM",
            "num_head": 1,
            "num_layers": calibration_layers,
            "scales_shape": [calibration_layers, 2],
            "scales": scales[:, None].expand(-1, 2).tolist(),
            "q_calibration": {
                "num_head": 1,
                "scales_shape": [calibration_layers, 1],
                "scales": (scales[:, None] * 2).tolist(),
            },
            "quant_type": "per_head" if encoding == "fp8kv_sph" else "per_tensor",
        }
        calibration_path = tmp_path / "calibration.json"
        calibration_path.write_text(json.dumps(calibration))
        args.kv_quant_calibration_config_path = str(calibration_path)
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(asdict(args)))
    get_env_start_args.cache_clear()
    get_added_mtp_kv_layer_num.cache_clear()
    main_cls = Qwen3NextTpPartModel if main_linear else LlamaTpPartModel
    monkeypatch.setattr("lightllm.utils.llm_utils.get_llm_model_class", lambda: main_cls)
    monkeypatch.setattr("lightllm.models.get_draft_model_class", lambda *_: Qwen3NextTpPartModel)
    models, warmed, padded, captured, prefill_captured, ready = [], set(), set(), set(), set(), []
    checked = []
    mapping = SimpleNamespace(bind_hold_page=lambda indices: None)
    request_allocations = []

    def create_req_manager(max_request_num: int, max_sequence_length: int):
        assert len(models) == 1 + draft_count
        assert all(not hasattr(item, "req_manager") for item in models)
        request_allocations.append((max_request_num, max_sequence_length))
        return mapping

    monkeypatch.setattr("lightllm.common.req_manager.ReqManager", create_req_manager)

    def mark_backends(model: TpPartBaseModel):
        # 第一个模型初始化 backend 前，全部 State 已创建完成，可通过缓存查询取得资源。
        for item in models:
            for state in item.att_resource_manager.get_full_attention_states(item.att_resource_role):
                logical_layer_ids = {layer_id for group in state.config.logical_layer_groups for layer_id in group}
                for layer in item.layers_infer:
                    if layer.layer_num_ in logical_layer_ids:
                        assert item.att_resource_manager.get_att_state(
                            model_name=item.att_resource_role,
                            att_type="",
                            logical_layer_id=layer.layer_num_,
                            state_class=type(state),
                        ) is state
        model.prefill_att_backend = model.decode_att_backend = object()
        model.prefill_att_backend1 = model.decode_att_backend1 = None
        model.backends_bound = True

    def warmup(model: TpPartBaseModel):
        assert model.backends_bound
        assert all(item.req_manager is mapping for item in models)
        assert all(item.att_resource_manager is models[0].att_resource_manager for item in models)
        assert model.att_resource_manager.full_att_allocator.size == 32
        assert all(item.max_total_token_num == 32 for item in models[: models.index(model) + 1])
        assert all(item.max_total_token_num is None for item in models[models.index(model) + 1 :])
        assert captured == prefill_captured == {id(item) for item in models[: models.index(model)]}
        assert not ready
        warmed.add(id(model))

    def capture(model: TpPartBaseModel):
        assert warmed == padded == {id(item) for item in models[: models.index(model) + 1]}
        assert not ready
        captured.add(id(model))

    def check_max_len(model: TpPartBaseModel):
        assert model is models[0]
        assert captured == prefill_captured == {id(model)}
        assert not ready
        checked.append(model)

    offset = 0
    for index in range(1 + draft_count):
        cls = Qwen3NextTpPartModel if index else main_cls
        model = cls.__new__(cls)
        model.is_mtp_draft_model = bool(index)
        model.args, model.config = args, draft_config.copy() if index else config.copy()
        model.max_req_num, model.max_total_token_num, model.mem_fraction = 2, None, 0.9
        model.max_seq_length, model.batch_max_tokens = (32 if index else 24), 16
        model.tp_world_size_, model.tp_rank_ = 1, 0
        model.layers_infer = [SimpleNamespace(layer_num_=i) for i in range(offset, offset + model.config["n_layer"])]
        offset += len(model.layers_infer)
        model.torch_memory_saver = SimpleNamespace(region=lambda **_: nullcontext())
        if index:
            model.main_model, model.mtp_previous_draft_models = models[0], models[1:].copy()
        model._init_att_backend = lambda model=model: mark_backends(model)
        model._init_att_backend1 = lambda: None
        model._init_hidden_collector = lambda: None
        model._autotune_warmup = lambda model=model: warmup(model)
        model._init_padded_req = lambda model=model: padded.add(id(model))
        model._init_cudagraph = lambda model=model: capture(model)
        model._init_prefill_cuda_graph = lambda model=model: prefill_captured.add(id(model))
        model._check_max_len_infer = lambda model=model: check_max_len(model)
        models.append(model)

    def memory_info(device: str | torch.device | int):
        assert all(item.req_manager is mapping for item in models)
        manager = models[0].att_resource_manager
        states = manager.get_full_attention_states()
        assert all(read_kv_storage(state) is None for state in states)
        assert manager.draft0_gdn_conv_bf16_ssm_fp32_state.conv_state.is_cuda
        assert manager.draft0_gdn_conv_bf16_ssm_fp32_state.big_pages_pool is None
        assert manager.full_att_allocator is None
        assert not warmed
        return manager.get_full_att_token_bytes() * (32 + args.page_size), 0

    monkeypatch.setattr(torch.cuda, "mem_get_info", memory_info)
    monkeypatch.setattr("lightllm.utils.envs_utils.set_model_init_status", ready.append)
    assert all(not hasattr(item, "att_resource_manager") for item in models)

    class RuntimeReady(Exception):
        """Stop after model readiness, before unrelated queue and thread setup."""

    if via_backend:
        backend = ModeBackend.__new__(ModeBackend)
        backend.init_rank_infos = lambda: None
        backend.init_mtp_draft_model = lambda _: backend.draft_models.extend(models[1:])
        monkeypatch.setattr(base_backend, "init_distributed_env", lambda _: None)
        monkeypatch.setattr(base_backend.dist_group_manager, "create_groups", lambda **_: None)
        monkeypatch.setattr(base_backend, "TokenLoad", lambda *_: None)
        monkeypatch.setattr(base_backend, "get_model", lambda *_: (models[0], False))

        def mark_ready(value: bool):
            ready.append(value)
            raise RuntimeReady

        monkeypatch.setattr("lightllm.utils.envs_utils.set_model_init_status", mark_ready)

    def start_runtime():
        if via_backend:
            with pytest.raises(RuntimeReady):
                backend.init_model(
                    dict(args=args, world_size=1, load_way="HF", weight_dir=str(tmp_path), max_total_token_num=None)
                )
        else:
            assert all(getattr(item, "att_resource_manager", None) is None for item in models), "resources already initialized"
            req_manager = create_req_manager(
                max_request_num=models[0].max_req_num,
                max_sequence_length=max(item.get_req_manager_max_seq_len() for item in models),
            )
            for item in models:
                item.bind_req_manager(req_manager=req_manager)
            for index, item in enumerate(models):
                item.prepare_att_resources()
                assert all(other.att_resource_manager is models[0].att_resource_manager for other in models[: index + 1])
                assert all(not hasattr(other, "att_resource_manager") for other in models[index + 1 :])
            main = models[0]
            main.att_resource_manager.initialize_resources(
                args, token_capacity=main.max_total_token_num, mem_fraction=main.mem_fraction
            )
            assert all(item.max_total_token_num is None for item in models)
            for index, item in enumerate(models):
                item.init_runtime()
                assert all(not getattr(other, "backends_bound", False) for other in models[index + 1 :])
                assert warmed == padded == captured == prefill_captured == {
                    id(other) for other in models[: index + 1]
                }

    start_runtime()
    assert request_allocations == [(2, 32)]
    assert captured == {id(model) for model in models}
    assert checked == [models[0]]
    assert ready == ([True] if via_backend else [])
    manager = models[0].att_resource_manager
    assert len(manager.get_bounded_attention_states()) == draft_count + int(main_linear)
    assert all(not hasattr(model, "mtp_state_cursors") for model in models)
    assert all(not hasattr(model, "full_att_state") for model in models)
    draft_states = [manager.get_bounded_attention_states(f"draft{i}")[0] for i in range(draft_count)]
    assert len({state.conv_state.data_ptr() for state in draft_states}) == draft_count
    manager.commit_bounded_att_state(
        torch.tensor([0], dtype=torch.int32, device="cuda"),
        torch.tensor([draft_count + 1], dtype=torch.int32, device="cuda"), role="main",
    )
    assert all(not state.mtp_state_index.any() for state in draft_states)
    if main_linear:
        assert manager.main_gdn_conv_bf16_ssm_fp32_state.mtp_state_index[0] == draft_count
    for i, state in enumerate(draft_states):
        assert len(state.config.logical_layer_ids) == 1
        manager.commit_bounded_att_state(
            torch.tensor([0], dtype=torch.int32, device="cuda"),
            torch.tensor([i + 2], dtype=torch.int32, device="cuda"), role=f"draft{i}",
        )
        assert state.mtp_state_index[0] == i + 1
        assert all(not other.mtp_state_index.any() for other in draft_states[i + 1 :])
        if draft_full:
            main_state = manager.get_full_attention_states("main")[0]
            draft_state = manager.get_full_attention_states(f"draft{i}")[0]
            assert main_state.config.head_dim == 16
            assert draft_state.config.head_dim == 8
            assert {buffer.data_ptr() for buffer in full_state_test_buffers(main_state).values()}.isdisjoint(
                buffer.data_ptr() for buffer in full_state_test_buffers(draft_state).values()
            )
            assert len(draft_state.config.logical_layer_groups) == 1
        else:
            assert not manager.get_full_attention_states(f"draft{i}")

    if encoding != "None":
        ranges = [("main", 0, main_full_layers)] + [
            (f"draft{i}", main_full_layers + i, main_full_layers + i + 1) for i in range(draft_count)
        ]
        for role, start, end in ranges:
            for state in manager.get_full_attention_states(role):
                assert torch.equal(state.scales[:, 0].cpu(), scales[start:end])
                if encoding == "fp8kv_sph":
                    assert torch.equal(state.q_scales[:, 0].cpu(), scales[start:end] * 2)

    layouts = build_cpu_cache_layouts()
    assert (max(layout.byte_end for layout in layouts.values()) + 15) // 16 * 16 == manager.get_cpu_page_bytes()
    assert layouts.keys() == manager.get_cpu_transfer_layouts().keys()
    for resource_id, layout in layouts.items():
        runtime_layout = manager.get_cpu_transfer_layouts()[resource_id]
        assert [(r.field, r.byte_start, r.byte_end, r.dtype, r.axes, r.shape) for r in layout.regions] == [
            (r.field, r.byte_start, r.byte_end, r.dtype, r.axes, r.shape) for r in runtime_layout.regions
        ]
    slot = manager.bounded_att_big_page_allocator.alloc(1)[0]
    for i, state in enumerate(manager.get_bounded_attention_states()):
        state.conv_state.fill_(i + 3)
        state.ssm_state.fill_(i + 7)
    manager.save_bounded_att_pages([0], page_kind="big", slot_ids=[slot])
    manager.reset_bounded_att_state([0])
    assert all(not state.conv_state[:, 0].any() for state in manager.get_bounded_attention_states())
    manager.restore_bounded_att_pages([0], page_kind="big", slot_ids=[slot])
    for i, state in enumerate(manager.get_bounded_attention_states()):
        assert (state.conv_state[:, 0, :, : state.config.conv_width] == i + 3).all()
        assert (state.ssm_state[:, 0] == i + 7).all()
    with pytest.raises(AssertionError, match="resources already initialized"):
        start_runtime()
    assert all(item.att_resource_manager is manager for item in models)
    get_env_start_args.cache_clear()
    get_added_mtp_kv_layer_num.cache_clear()
