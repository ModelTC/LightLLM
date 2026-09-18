import json
from types import SimpleNamespace
import pytest
import torch

import lightllm.common.basemodel.attention.fa3.fp as fa3_module
import lightllm.common.kv_cache_mem_manager.export_calibration_mem_manager as calibration_module
from lightllm.common.kv_cache_mem_manager.export_calibration_mem_manager import ExportCalibrationMemoryManager


def _manager(layer_num=3, head_num=2):
    manager = object.__new__(ExportCalibrationMemoryManager)
    manager.layer_num = layer_num
    manager.head_num = head_num
    manager.total_head_num = head_num
    manager.qmax = 448.0
    manager.qmin = -448.0
    manager.calibration_counts = [0] * layer_num
    manager.observed_token_rows = [0] * layer_num
    manager._calibration_active = False
    manager.abs_max = torch.zeros((layer_num, 2 * head_num), dtype=torch.float32)
    return manager


def _kv(value):
    return torch.tensor([[[value], [2 * value], [3 * value], [4 * value]]], dtype=torch.bfloat16)


def _q_manager(layer_num=3, head_num=2, target="q"):
    manager = _manager(layer_num, head_num)
    manager.calibration_target = target
    if target == "q":
        manager.abs_max = torch.zeros((layer_num, head_num), dtype=torch.float32)
    else:
        manager.q_abs_max = torch.zeros((layer_num, head_num), dtype=torch.float32)
        manager.q_calibration_counts = [0] * layer_num
        manager.q_observed_token_rows = [0] * layer_num
    return manager


def _fa3_state(state_class, manager, layer_index, request_ids, cu_seqlens_q, **infer_state):
    req_manager = SimpleNamespace(HOLD_REQUEST_ID=-1)
    model = SimpleNamespace(mem_manager=manager, req_manager=req_manager)
    backend = SimpleNamespace(model=model, _find_layer_index=lambda **_: layer_index)
    state = state_class(
        backend=backend,
        infer_state=SimpleNamespace(
            b_req_idx=torch.tensor(request_ids, dtype=torch.int32),
            b_seq_len=torch.tensor(infer_state.get("b_seq_len", [101] * len(request_ids)), dtype=torch.int32),
            max_q_seq_len=infer_state.get("max_q_seq_len", 2),
            is_prefill=infer_state.get("is_prefill", False),
        ),
    )
    state.cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32)
    state.cu_seqlens_k = torch.tensor(cu_seqlens_q, dtype=torch.int32)
    state.page_table = torch.empty((len(request_ids), 1), dtype=torch.int32)
    state.b_att_seq_len = torch.ones(len(request_ids), dtype=torch.int32)
    state.decode_max_q_seq_len = 1
    state.decode_max_kv_seq_len = 1
    state.causal = True
    return state


def test_explicit_lifecycle_collects_every_target_and_draft_layer(monkeypatch):
    monkeypatch.setattr(calibration_module, "get_model_init_status", lambda: True)
    monkeypatch.setattr(calibration_module, "get_global_rank", lambda: 0)
    monkeypatch.setattr(calibration_module, "get_added_mtp_kv_layer_num", lambda: 1)
    monkeypatch.setattr(
        calibration_module,
        "get_env_start_args",
        lambda: SimpleNamespace(llm_prefill_att_backend=["fa3"], model_dir="fake"),
    )
    monkeypatch.setattr(calibration_module, "get_model_architectures", lambda _: "Fake")
    manager = _manager()
    manager.update_calibration_data(_kv(99), 0)
    assert manager.calibration_counts == [0, 0, 0]
    manager.begin_calibration()
    for layer, value in enumerate((5, 6, 7)):
        manager.update_calibration_data(_kv(value), layer)
    snap = manager.snapshot_calibration()
    assert not snap["active"]
    assert snap["counts"] == [1, 1, 1]
    assert snap["observed_token_rows"] == [1, 1, 1]
    assert snap["abs_max"] == [[5, 10, 15, 20], [6, 12, 18, 24], [7, 14, 21, 28]]
    manager.update_calibration_data(_kv(99), 2)
    assert manager.calibration_counts == [1, 1, 1]


def test_empty_kv_does_not_consume_layer_sample(monkeypatch):
    monkeypatch.setattr(calibration_module, "get_model_init_status", lambda: True)
    monkeypatch.setattr(calibration_module, "get_global_rank", lambda: 0)
    monkeypatch.setattr(calibration_module, "get_added_mtp_kv_layer_num", lambda: 1)
    monkeypatch.setattr(
        calibration_module,
        "get_env_start_args",
        lambda: SimpleNamespace(llm_prefill_att_backend=["fa3"], model_dir="fake"),
    )
    monkeypatch.setattr(calibration_module, "get_model_architectures", lambda _: "Fake")
    manager = _manager(layer_num=1)
    manager.begin_calibration()
    manager.update_calibration_data(torch.empty((0, 4, 1), dtype=torch.bfloat16), 0)
    assert manager.calibration_counts == [0]


@pytest.mark.parametrize("target", ["q", "qkv"])
def test_fa3_prefill_and_decode_collect_q_with_new_query_rows_only(monkeypatch, target):
    """Prefill expands request validity by new Q rows, not total KV lengths."""
    monkeypatch.setattr(calibration_module, "get_model_init_status", lambda: True)
    monkeypatch.setattr(fa3_module, "flash_attn_with_kvcache", lambda **kwargs: kwargs["q"])
    monkeypatch.setattr(fa3_module, "flash_attn_with_kvcache_autotune", lambda **kwargs: kwargs["q"])
    manager = _q_manager(target=target)
    manager._calibration_active = True
    k = v = torch.zeros((1, 2, 1), dtype=torch.bfloat16)

    prefill = _fa3_state(
        fa3_module.Fa3PrefillAttState,
        manager,
        0,
        [7, -1, 8],
        [0, 2, 3, 5],
        b_seq_len=[101, 10_000, 202],
        is_prefill=True,
    )
    q_prefill = torch.tensor(
        [[1, 2, 3, 4], [5, 6, 7, 8], [999, 999, 999, 999], [9, 10, 11, 12], [13, 14, 15, 16]],
        dtype=torch.bfloat16,
    ).unsqueeze(-1)
    prefill._nomarl_prefill_att(q_prefill, k, v, fa3_module.AttControl())

    same_target_decode = _fa3_state(fa3_module.Fa3DecodeAttState, manager, 0, [7], [0, 1])
    same_target_decode._normal_decode_att(
        torch.tensor([[30, 1, 1, 15]], dtype=torch.bfloat16).unsqueeze(-1), k, v, fa3_module.AttControl()
    )
    target_decode = _fa3_state(fa3_module.Fa3DecodeAttState, manager, 1, [7, -1], [0, 1, 2])
    target_decode._normal_decode_att(
        torch.tensor([[17, 18, 19, 20], [999, 999, 999, 999]], dtype=torch.bfloat16).unsqueeze(-1),
        k,
        v,
        fa3_module.AttControl(),
    )
    draft_decode = _fa3_state(fa3_module.Fa3DecodeAttState, manager, 2, [8], [0, 1])
    draft_decode._normal_decode_att(
        torch.tensor([[21, 22, 23, 24]], dtype=torch.bfloat16).unsqueeze(-1), k, v, fa3_module.AttControl()
    )

    counts = manager.calibration_counts if target == "q" else manager.q_calibration_counts
    rows = manager.observed_token_rows if target == "q" else manager.q_observed_token_rows
    maxima = manager.abs_max if target == "q" else manager.q_abs_max
    assert counts == [2, 1, 1]
    assert rows == [5, 1, 1]
    assert maxima.tolist() == [[30, 16], [18, 20], [22, 24]]
    if target == "qkv":
        assert manager.calibration_counts == [0, 0, 0]
        assert manager.abs_max.tolist() == [[0, 0, 0, 0]] * 3


def test_fa3_prefill_q_collection_ignores_inactive_and_all_padding(monkeypatch):
    monkeypatch.setattr(calibration_module, "get_model_init_status", lambda: True)
    monkeypatch.setattr(fa3_module, "flash_attn_with_kvcache", lambda **kwargs: kwargs["q"])
    manager = _q_manager(layer_num=1)
    state = _fa3_state(fa3_module.Fa3PrefillAttState, manager, 0, [-1, -1], [0, 0, 2])
    q = torch.full((2, 4, 1), 10_000, dtype=torch.bfloat16)
    state._nomarl_prefill_att(q, torch.zeros((1, 2, 1)), torch.zeros((1, 2, 1)), fa3_module.AttControl())
    assert manager.calibration_counts == [0]
    manager._calibration_active = True
    state._nomarl_prefill_att(q, torch.zeros((1, 2, 1)), torch.zeros((1, 2, 1)), fa3_module.AttControl())
    assert manager.calibration_counts == [0]

    calls = []
    kv_only = SimpleNamespace(calibration_target="kv", update_q_calibration_data=lambda *args: calls.append(args))
    state = _fa3_state(fa3_module.Fa3PrefillAttState, kv_only, 0, [1], [0, 1], is_prefill=True)
    state._nomarl_prefill_att(q[:1], torch.zeros((1, 2, 1)), torch.zeros((1, 2, 1)), fa3_module.AttControl())
    assert calls == []


def test_joint_per_head_export_loads_for_draft_and_target_only(tmp_path, monkeypatch):
    import lightllm.common.kv_cache_mem_manager.fp8_static_per_head_quant_mem_manager as loader

    cfg = {
        "version": "1.0",
        "architectures": "Fake",
        "quant_type": "per_head",
        "qmin": -448.0,
        "qmax": 448.0,
        "num_layers": 3,
        "num_target_layers": 2,
        "num_draft_layers": 1,
        "num_head": 2,
        "scales_shape": [3, 4],
        "scales": [[1.0] * 4 for _ in range(3)],
        "q_calibration": {"num_head": 2, "scales_shape": [3, 2], "scales": [[1.0] * 2 for _ in range(3)]},
    }
    path = tmp_path / "joint.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(loader, "get_model_architectures", lambda _: "Fake")
    monkeypatch.setattr(
        loader,
        "get_env_start_args",
        lambda: SimpleNamespace(kv_quant_calibration_config_path=str(path), model_dir="fake"),
    )
    monkeypatch.setattr(loader, "get_dp_world_size", lambda: 1)
    for runtime_draft, layer_num in ((1, 3), (0, 2)):
        monkeypatch.setattr(loader, "get_added_mtp_kv_layer_num", lambda d=runtime_draft: d)
        obj = object.__new__(loader.FP8StaticPerHeadQuantMemManager)
        obj.head_num, obj.layer_num = 2, layer_num
        loaded_cfg, loaded_scales = obj._load_and_check_config()
        assert loaded_cfg["num_target_layers"] == 2
        assert loaded_scales.tolist() == cfg["scales"]


def test_q_calibration_loader_accepts_decode_and_prefill_decode_stages(tmp_path, monkeypatch):
    import lightllm.common.kv_cache_mem_manager.fp8_static_per_head_quant_mem_manager as loader

    kv_cfg = {
        "architectures": "Fake",
        "quant_type": "per_head",
        "qmin": -448.0,
        "qmax": 448.0,
        "num_layers": 3,
        "num_target_layers": 2,
        "num_draft_layers": 1,
        "num_head": 2,
        "scales_shape": [3, 4],
        "scales": [[1.0] * 4 for _ in range(3)],
    }
    q_cfg = {
        **kv_cfg,
        "tensor": "q",
        "scale_layout": "kv_head_group",
        "scales_shape": [3, 2],
        "scales": [[1.0, 2.0] for _ in range(3)],
    }
    monkeypatch.setattr(loader, "get_added_mtp_kv_layer_num", lambda: 1)
    monkeypatch.setattr(loader, "get_dp_world_size", lambda: 1)
    for stage in ("decode", "prefill_and_decode"):
        path = tmp_path / f"{stage}.json"
        path.write_text(json.dumps({**kv_cfg, "q_calibration": {**q_cfg, "calibration_stage": stage}}))
        monkeypatch.setattr(
            loader,
            "get_env_start_args",
            lambda p=path: SimpleNamespace(kv_quant_calibration_config_path=str(p)),
        )
        obj = object.__new__(loader.FP8StaticPerHeadQuantMemManager)
        obj.head_num, obj.layer_num = 2, 3
        loaded_cfg, _ = obj._load_and_check_config()
        assert loaded_cfg["q_calibration"]["calibration_stage"] == stage
