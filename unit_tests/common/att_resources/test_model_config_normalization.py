"""模型启动和 CPU cache 必须使用同一份有效配置。"""

from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel import TpPartBaseModel
from lightllm.models.glm4_moe_lite_mtp.model import Glm4MoeLiteMTPModel
from lightllm.models.glm5_next_mtp.model import Glm5NextMTPModel
from lightllm.models.qwen3_5_dflash.model import Qwen3_5DFlashModel
from lightllm.models.qwen3_5_dspark.model import Qwen3_5DSparkModel
from lightllm.models.qwen3_5_mtp.model import Qwen3_5MTPModel
from lightllm.models.qwen3_dflash2.model import Qwen3DFlash2Model
from lightllm.server.core.objs.start_args_type import StartArgs


@pytest.mark.parametrize("model_class", [Qwen3DFlash2Model, Qwen3_5DFlashModel, Qwen3_5DSparkModel])
@pytest.mark.parametrize("nested", [False, True])
def test_dflash_runtime_and_cpu_state_use_overridden_config(
    model_class: type[TpPartBaseModel], tmp_path: Path, nested: bool,
) -> None:
    text_config = {
        "num_hidden_layers": 32,
        "n_layer": 32,
        "num_attention_heads": 8,
        "n_head": 8,
        "hidden_size": 1024,
        "num_key_value_heads": 4,
        "rope_parameters": {"rope_theta": 10000, "partial_rotary_factor": 0.5},
        "dflash_config": {"num_hidden_layers": 4, "num_attention_heads": 4, "hidden_size": 512},
    }
    raw_config = {"text_config": text_config} if nested else text_config
    original = deepcopy(raw_config)
    (tmp_path / "config.json").write_text(json.dumps(raw_config))

    # CPU 服务只调用类接口；模型启动从同一文件走真实 _init_config 调用链。
    cpu_config = model_class.normalize_model_config(raw_config)
    model = model_class.__new__(model_class)
    model.weight_dir_ = str(tmp_path)
    model.finetune_config = None
    model._init_config()
    assert model.config == cpu_config
    assert cpu_config["num_hidden_layers"] == cpu_config["n_layer"] == 4
    assert cpu_config["num_attention_heads"] == cpu_config["n_head"] == 4
    assert cpu_config["hidden_size"] == cpu_config["n_embd"] == cpu_config["n_embed"] == 512
    assert model_class.normalize_model_config(cpu_config) == cpu_config
    assert raw_config == original

    args = StartArgs(data_type="bfloat16", llm_kv_type="None", page_size=4)
    (state,) = model_class.create_full_att_states(
        model_config=cpu_config,
        args=args,
        model_logical_layer_ids=tuple(range(cpu_config["num_hidden_layers"])),
    )
    assert state.config.logical_layer_groups == ((0,), (1,), (2,), (3,))
    assert state.config.head_dim == 128


@pytest.mark.parametrize("model_class", [Glm4MoeLiteMTPModel, Glm5NextMTPModel, Qwen3_5MTPModel])
def test_native_mtp_runtime_matches_normalized_config(
    model_class: type[TpPartBaseModel], tmp_path: Path,
) -> None:
    text_config = {
        "num_hidden_layers": 32,
        "num_nextn_predict_layers": 1,
        "hidden_size": 1024,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "full_attention_interval": 4,
        "mhc": True,
        "scoring_func": "sigmoid",
    }
    original = deepcopy(text_config)
    raw_config = {"text_config": text_config, "vision_config": {}}
    (tmp_path / "config.json").write_text(json.dumps(raw_config))
    config = model_class.normalize_model_config(raw_config)
    model = model_class.__new__(model_class)
    model.weight_dir_ = str(tmp_path)
    model.finetune_config = None
    model.tp_world_size_ = 1
    model.main_model = SimpleNamespace(config=text_config)
    model._init_config()

    assert model.config == config
    assert config["num_hidden_layers"] == config["n_layer"] == 1
    assert model_class.normalize_model_config(config) == config
    assert text_config == original
    if model_class is Qwen3_5MTPModel:
        assert config["full_attention_interval"] == 1
    else:
        assert config["mtp_checkpoint_layer_start"] == 32


@pytest.mark.parametrize("model_class", [Glm4MoeLiteMTPModel, Glm5NextMTPModel])
def test_native_mtp_weights_keep_checkpoint_layer_start(
    model_class: type[TpPartBaseModel], monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = model_class.normalize_model_config({
        "num_hidden_layers": 32, "num_nextn_predict_layers": 1, "hidden_size": 1024,
    })
    module = importlib.import_module(model_class.pre_and_post_weight_class.__module__)
    names = []

    def init_base(self: object, data_type: torch.dtype, network_config: dict[str, object]) -> None:
        self.data_type_ = data_type

    def record_weight(**kwargs: object) -> SimpleNamespace:
        names.append(kwargs.get("weight_names", kwargs.get("weight_name")))
        return SimpleNamespace()

    monkeypatch.setattr(module.PreAndPostLayerWeight, "__init__", init_base)
    monkeypatch.setattr(module, "ROWMMWeight", record_weight)
    monkeypatch.setattr(module, "RMSNormWeight", record_weight)
    quant_cfg = SimpleNamespace(get_quant_method=lambda layer, name: None)
    pre_post_weight = model_class.pre_and_post_weight_class(
        data_type=torch.bfloat16, network_config=config, quant_cfg=quant_cfg,
    )
    assert len(names) == 4
    assert all(name.startswith("model.layers.32.") for name in names)

    model = model_class.__new__(model_class)
    model.config = config
    model.data_type = torch.bfloat16
    model.quant_cfg = quant_cfg
    model.main_model = SimpleNamespace(pre_post_weight=SimpleNamespace(
        wte_weight_=object(), lm_head_weight_=object(), final_norm_weight_=object(),
    ))
    model.pre_and_post_weight_class = lambda *args, **kwargs: pre_post_weight
    layers = []

    def record_layer(
        layer_num: int,
        data_type: torch.dtype,
        network_config: dict[str, object],
        quant_cfg: object,
    ) -> object:
        layers.append(layer_num)
        return SimpleNamespace(verify_load=lambda: None)

    model.transformer_weight_class = record_layer
    if model_class is Glm4MoeLiteMTPModel:
        model.weight_dir_ = "unused"
        model.weight_dict = None
        pre_post_weight.verify_load = lambda: None
        monkeypatch.setattr(importlib.import_module(model_class.__module__), "load_hf_weights", lambda *a, **kw: None)
    model._init_weights()
    assert layers == [32]
