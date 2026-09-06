import json
from importlib import import_module
from types import SimpleNamespace

import pytest
import torch

import lightllm.common.basemodel.attention.base_att as base_att_module
import lightllm.common.basemodel.hidden_collector as hidden_collector_module
from lightllm.common.basemodel.attention.base_att import BaseAttBackend
from lightllm.common.basemodel.attention.fa3.fp import Fa3DecodeAttState, Fa3PrefillAttState
from lightllm.common.basemodel.attention.fa3.mla import MlaFa3DecodeAttState, MlaFa3PrefillAttState
from lightllm.common.basemodel.attention.triton.fp import TritonDecodeAttState
from lightllm.models import get_draft_model_class
from lightllm.models.qwen3_eagle.layer_weights.transformer_layer_weight import Qwen3EagleTransformerLayerWeight
from lightllm.utils import envs_utils


@pytest.mark.parametrize(
    "module_name, class_name",
    [
        ("lightllm.models.deepseek_mtp.model", "Deepseek3MTPModel"),
        ("lightllm.models.glm4_moe_lite_mtp.model", "Glm4MoeLiteMTPModel"),
        ("lightllm.models.mistral_mtp.model", "MistralMTPModel"),
        ("lightllm.models.qwen3_moe_mtp.model", "Qwen3MOEMTPModel"),
        ("lightllm.models.qwen3_5_mtp.model", "Qwen3_5MTPModel"),
        ("lightllm.models.qwen3_5_moe_mtp.model", "Qwen3_5MoeMTPModel"),
        ("lightllm.models.qwen3_eagle.model", "Qwen3EagleModel"),
        ("lightllm.models.qwen3_dflash.model", "Qwen3DFlashModel"),
        ("lightllm.models.qwen3_5_dflash.model", "Qwen3_5DFlashModel"),
        ("lightllm.models.qwen3_dspark.model", "Qwen3DSparkModel"),
        ("lightllm.models.qwen3_5_dspark.model", "Qwen3_5DSparkModel"),
    ],
)
def test_spec_draft_model_class_is_marked(module_name, class_name):
    model_class = getattr(import_module(module_name), class_name)
    assert model_class.is_mtp_draft_model is True


def test_qwen3_eagle_uses_layers_checkpoint_prefix():
    layer_weight = Qwen3EagleTransformerLayerWeight.__new__(Qwen3EagleTransformerLayerWeight)
    layer_weight.layer_num_ = 2

    layer_weight._init_weight_names()

    assert layer_weight._q_weight_name == "layers.2.self_attn.q_proj.weight"
    assert layer_weight._hidden_norm_weight_name == "layers.2.hidden_norm.weight"


@pytest.mark.parametrize(
    "mtp_mode, is_draft_model, draft_step, dynamic_spec, expected",
    [
        ("dspark", False, 7, True, True),
        ("dspark", True, 7, True, False),
        ("dflash", False, 7, True, True),
        ("dflash", True, 7, True, False),
        ("vanilla_with_att", True, 7, True, False),
        ("vanilla_with_att", True, 0, True, False),
        ("eagle3", True, 0, True, False),
        ("eagle3", False, 7, True, True),
        ("eagle3", False, 7, False, False),
    ],
)
def test_attention_backend_selects_dynamic_spec_layout(
    monkeypatch,
    mtp_mode,
    is_draft_model,
    draft_step,
    dynamic_spec,
    expected,
):
    monkeypatch.setattr(
        base_att_module,
        "get_env_start_args",
        lambda: SimpleNamespace(mtp_mode=mtp_mode, mtp_dynamic_verify=dynamic_spec),
    )
    backend = SimpleNamespace(
        model=SimpleNamespace(
            is_mtp_draft_model=is_draft_model,
            mtp_manager=SimpleNamespace(get_decode_draft_step=lambda _: draft_step),
        )
    )

    assert BaseAttBackend.uses_dynamic_spec_verify_layout(backend) is expected


@pytest.mark.parametrize(
    "mtp_mode, is_draft_model, expected",
    [
        (None, False, True),
        ("dflash", False, True),
        ("dflash", True, False),
        ("dspark", True, False),
        ("eagle3", True, True),
        ("vanilla_with_att", True, True),
    ],
)
def test_attention_backend_selects_causality(monkeypatch, mtp_mode, is_draft_model, expected):
    monkeypatch.setattr(
        base_att_module,
        "get_env_start_args",
        lambda: SimpleNamespace(mtp_mode=mtp_mode),
    )
    backend = SimpleNamespace(model=SimpleNamespace(is_mtp_draft_model=is_draft_model))

    assert BaseAttBackend.uses_causal_attention(backend) is expected


@pytest.mark.parametrize("state_class", [Fa3PrefillAttState, MlaFa3PrefillAttState])
def test_fa3_prefill_state_owns_causality(state_class):
    infer_state = SimpleNamespace(
        b1_cu_q_seq_len=torch.tensor([0, 1, 2], dtype=torch.int32),
        b1_cu_kv_seq_len=torch.tensor([0, 3, 7], dtype=torch.int32),
        b_req_idx=torch.tensor([0, 1], dtype=torch.int32),
        batch_size=2,
        max_kv_seq_len=4,
        input_ids=torch.empty(2, dtype=torch.int64),
        req_manager=SimpleNamespace(req_to_token_indexs=torch.arange(8, dtype=torch.int32).reshape(2, 4)),
    )
    state = state_class(
        backend=SimpleNamespace(uses_causal_attention=lambda: False),
        infer_state=infer_state,
    )

    state.init_state()

    assert state.causal is False


@pytest.mark.parametrize("state_class", [Fa3DecodeAttState, MlaFa3DecodeAttState])
def test_fa3_decode_state_owns_causality(state_class):
    infer_state = SimpleNamespace(
        b1_cu_q_seq_len=torch.tensor([0, 1, 2], dtype=torch.int32),
        b1_cu_kv_seq_len=torch.tensor([0, 3, 7], dtype=torch.int32),
        b_req_idx=torch.tensor([0, 1], dtype=torch.int32),
        b_seq_len=torch.tensor([3, 4], dtype=torch.int32),
    )
    model = SimpleNamespace(
        is_mtp_draft_model=False,
        mtp_manager=SimpleNamespace(get_decode_draft_step=lambda _: 0),
    )
    state = state_class(
        backend=SimpleNamespace(
            model=model,
            uses_causal_attention=lambda: False,
            uses_dynamic_spec_verify_layout=lambda: False,
        ),
        infer_state=infer_state,
    )
    state._init_page_table = lambda _: None

    state.init_state()

    assert state.causal is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_mtp_decode_state_builds_group_markers():
    model = SimpleNamespace(
        is_mtp_draft_model=False,
        mtp_manager=SimpleNamespace(get_decode_draft_step=lambda _: 2),
        req_manager=SimpleNamespace(HOLD_REQUEST_ID=-1),
    )
    state = TritonDecodeAttState(
        backend=SimpleNamespace(model=model),
        infer_state=SimpleNamespace(
            b_req_idx=torch.tensor([7, 7, -1, -1], dtype=torch.int32, device="cuda"),
        ),
    )

    state.init_state()

    assert state.b_mark_mtp_shared_group.cpu().tolist() == [0, 2, 1, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("state_class", [Fa3DecodeAttState, MlaFa3DecodeAttState])
def test_fa3_dynamic_decode_state_builds_group_markers(state_class):
    model = SimpleNamespace(req_manager=SimpleNamespace(HOLD_REQUEST_ID=-1))
    state = state_class(
        backend=SimpleNamespace(model=model),
        infer_state=SimpleNamespace(
            b_req_idx=torch.tensor([7, 7, -1, -1], dtype=torch.int32, device="cuda"),
            b_seq_len=torch.tensor([3, 4, 2, 2], dtype=torch.int32, device="cuda"),
            batch_size=4,
        ),
    )

    b_att_req_idx = state._init_dynamic_spec_verify_state(draft_step=2)
    torch.cuda.synchronize()

    assert b_att_req_idx.cpu().tolist() == [7, -1, -1, -1]
    assert state.b_att_seq_len.cpu().tolist() == [4, 2, 2, 0]
    assert state.cu_seqlens_q.cpu().tolist() == [0, 2, 3, 4, 4]


@pytest.mark.parametrize(
    "model_type, spec_mode, expected_class_name",
    [
        ("deepseek_v3", "vanilla_with_att", "Deepseek3MTPModel"),
        ("deepseek_v3", "eagle_with_att", "Deepseek3MTPModel"),
        ("glm4_moe_lite", "vanilla_with_att", "Glm4MoeLiteMTPModel"),
        ("glm4_moe_lite", "eagle_with_att", "Glm4MoeLiteMTPModel"),
        ("mistral", "vanilla_no_att", "MistralMTPModel"),
        ("mistral", "eagle_no_att", "MistralMTPModel"),
        ("qwen3_moe", "vanilla_no_att", "Qwen3MOEMTPModel"),
        ("qwen3_moe", "eagle_no_att", "Qwen3MOEMTPModel"),
        ("qwen3_5", "vanilla_with_att", "Qwen3_5MTPModel"),
        ("qwen3_5_text", "eagle_with_att", "Qwen3_5MTPModel"),
        ("qwen3_5_moe", "vanilla_with_att", "Qwen3_5MoeMTPModel"),
        ("qwen3_5_moe_text", "eagle_with_att", "Qwen3_5MoeMTPModel"),
        ("qwen3", "dflash", "Qwen3DFlashModel"),
        ("qwen3_5", "dflash", "Qwen3_5DFlashModel"),
        ("qwen3_5_text", "dflash", "Qwen3_5DFlashModel"),
        ("qwen3", "dspark", "Qwen3DSparkModel"),
        ("qwen3_5", "dspark", "Qwen3_5DSparkModel"),
        ("qwen3_5_text", "dspark", "Qwen3_5DSparkModel"),
        ("qwen3", "eagle3", "Qwen3EagleModel"),
    ],
)
def test_draft_model_registry(model_type, spec_mode, expected_class_name):
    model_class = get_draft_model_class(
        model_cfg={"model_type": model_type},
        spec_mode=spec_mode,
    )

    assert model_class.__name__ == expected_class_name


def test_draft_model_registry_rejects_unsupported_model_type():
    with pytest.raises(ValueError, match="Unsupported speculative draft model"):
        get_draft_model_class(
            model_cfg={"model_type": "gemma4"},
            spec_mode="dspark",
        )


@pytest.mark.parametrize(
    "model_type, spec_mode",
    [
        ("deepseek_v3", "eagle3"),
        ("deepseek_v3", "dspark"),
        ("deepseek_v3", "dflash"),
        ("glm4_moe_lite", "eagle3"),
        ("glm4_moe_lite", "dspark"),
        ("glm4_moe_lite", "dflash"),
        ("qwen3_5", "eagle3"),
        ("qwen3_5_moe", "eagle3"),
        ("qwen3_5_moe", "dspark"),
        ("qwen3_5_moe", "dflash"),
        ("qwen3", "eagle_no_att"),
    ],
)
def test_draft_model_registry_rejects_unsupported_mode(model_type, spec_mode):
    with pytest.raises(ValueError, match="Unsupported speculative draft model"):
        get_draft_model_class(
            model_cfg={"model_type": model_type},
            spec_mode=spec_mode,
        )


def test_hidden_collector_reads_target_layer_ids(monkeypatch):
    config_reads = []

    def get_config_dict(path):
        config_reads.append(path)
        return {"target_layer_ids": [1, 20, 36]}, {}

    monkeypatch.setattr(
        hidden_collector_module.PretrainedConfig,
        "get_config_dict",
        get_config_dict,
    )
    monkeypatch.setattr(
        hidden_collector_module,
        "get_env_start_args",
        lambda: SimpleNamespace(mtp_draft_model_dir=["/models/dspark"]),
    )
    model = SimpleNamespace(is_mtp_draft_model=False, layers_num=40)

    collector = hidden_collector_module.LayerHiddenCollector(model=model)
    new_collector = collector.new_instance()

    assert collector.layer_ids == frozenset((1, 20, 36))
    assert new_collector.layer_ids == collector.layer_ids
    assert config_reads == ["/models/dspark"]


@pytest.mark.parametrize(
    "mtp_mode, mtp_step, expected_layer_num",
    [
        (None, 0, 0),
        ("vanilla_no_att", 7, 0),
        ("eagle_no_att", 7, 0),
        ("vanilla_with_att", 7, 7),
        ("eagle_with_att", 7, 1),
    ],
)
def test_fixed_added_mtp_kv_layer_num_by_mode(monkeypatch, mtp_mode, mtp_step, expected_layer_num):
    monkeypatch.setattr(
        envs_utils,
        "get_env_start_args",
        lambda: SimpleNamespace(mtp_mode=mtp_mode, mtp_step=mtp_step),
    )
    envs_utils.get_added_mtp_kv_layer_num.cache_clear()

    assert envs_utils.get_added_mtp_kv_layer_num() == expected_layer_num


def test_dflash_added_kv_layers_come_from_draft_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"num_hidden_layers": 5}))

    envs_utils.get_env_start_args.cache_clear()
    envs_utils.get_added_mtp_kv_layer_num.cache_clear()
    envs_utils.set_env_start_args(
        {
            "mtp_mode": "dflash",
            "mtp_step": 7,
            "mtp_dynamic_verify": False,
            "mtp_draft_model_dir": [str(tmp_path)],
        }
    )

    assert envs_utils.get_added_mtp_kv_layer_num() == 5


def test_qwen35_dflash_added_kv_layers_come_from_nested_draft_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "num_hidden_layers": 48,
                "dflash_config": {"num_hidden_layers": 5},
            }
        )
    )

    envs_utils.get_env_start_args.cache_clear()
    envs_utils.get_added_mtp_kv_layer_num.cache_clear()
    envs_utils.set_env_start_args(
        {
            "mtp_mode": "dflash",
            "mtp_step": 7,
            "mtp_dynamic_verify": False,
            "mtp_draft_model_dir": [str(tmp_path)],
        }
    )

    assert envs_utils.get_added_mtp_kv_layer_num() == 5


def test_dspark_added_kv_layers_come_from_draft_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"num_hidden_layers": 6}))

    envs_utils.get_env_start_args.cache_clear()
    envs_utils.get_added_mtp_kv_layer_num.cache_clear()
    envs_utils.set_env_start_args(
        {
            "mtp_mode": "dspark",
            "mtp_step": 7,
            "mtp_dynamic_verify": True,
            "mtp_draft_model_dir": [str(tmp_path)],
        }
    )

    assert envs_utils.get_added_mtp_kv_layer_num() == 6


def test_eagle3_added_kv_layers_come_from_draft_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"num_hidden_layers": 2}))

    envs_utils.get_env_start_args.cache_clear()
    envs_utils.get_added_mtp_kv_layer_num.cache_clear()
    envs_utils.set_env_start_args(
        {
            "mtp_mode": "eagle3",
            "mtp_step": 7,
            "mtp_dynamic_verify": False,
            "mtp_draft_model_dir": [str(tmp_path)],
        }
    )

    assert envs_utils.get_added_mtp_kv_layer_num() == 2
