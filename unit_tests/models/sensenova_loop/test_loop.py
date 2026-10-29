import dataclasses
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from lightllm.models.registry import ModelRegistry
from lightllm.models.sensenova_loop.config import get_loop_layer_indices, get_loop_layer_types
from lightllm.common.state_cache_manager.linear_att import LinearAttCacheConfig
from lightllm.server.core.objs.start_args_type import StartArgs


def _config(layers=32, start=4, end=28, count=3):
    return {
        "model_type": "sensenova_loop",
        "loop_settings": {"start": start, "end": end, "eval_count": count, "use_gate": False},
        "text_config": {
            "num_hidden_layers": layers,
            "full_attention_interval": 4,
            "layer_types": ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(layers)],
            "num_key_value_heads": 4,
            "head_dim": 256,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
        },
    }


@pytest.mark.parametrize("start,end,count", [(4, 28, 3), (4, 28, 2), (1, 6, 2)])
def test_cache_slots_follow_execution_depth_including_unaligned_loop(monkeypatch, start, end, count):
    config = _config(start=start, end=end, count=count)
    args = StartArgs(model_dir="unused", tp=4, dp=1, data_type="bfloat16", mtp_mode="eagle_with_att", mtp_step=3)
    monkeypatch.setattr("lightllm.common.state_cache_manager.linear_att.get_env_start_args", lambda: args)
    monkeypatch.setattr("lightllm.common.state_cache_manager.linear_att.get_added_mtp_kv_layer_num", lambda: 1)
    monkeypatch.setattr("transformers.configuration_utils.PretrainedConfig.get_config_dict", lambda _: (config, {}))
    cache = LinearAttCacheConfig.load_from_args()
    indices = get_loop_layer_indices(config["loop_settings"], 32)
    kinds = get_loop_layer_types(config["loop_settings"], config["text_config"])
    assert len(kinds) == 32 + (end - start) * (count - 1)
    full = linear = 0
    slots = {}
    for execution, (physical, kind) in enumerate(zip(indices, kinds)):
        if kind == "full_attention":
            slot = cache.get_full_att_kv_layer_index(execution)
            assert slot == full
            full += 1
        else:
            slot = cache.get_linear_state_layer_index(execution)
            assert slot == linear
            linear += 1
        assert slot not in slots.setdefault(physical, set())
        slots[physical].add(slot)
    assert cache.get_main_model_full_att_layer_num() == full
    assert cache.linear_layer_num == linear
    assert cache.get_full_att_kv_layer_index(len(indices)) == full  # Independent MTP draft.
    if (start, end, count) == (4, 28, 3):
        assert (cache.all_layer_num, full, linear) == (80, 20, 60)
    if (start, end, count) == (4, 28, 2):
        assert (cache.all_layer_num, full, linear) == (56, 14, 42)


def test_physical_weights_are_created_and_loaded_once(monkeypatch):
    from lightllm.models.sensenova_loop.model import SenseNovaLoopTpPartModel
    from lightllm.models.qwen3_5.model import Qwen3_5TpPartModel

    model = object.__new__(SenseNovaLoopTpPartModel)
    model.loop_layer_indices = get_loop_layer_indices(_config()["loop_settings"], 32)
    model.num_physical_layers = 32
    model.config = {"n_layer": 80}
    model.data_type, model.quant_cfg = torch.bfloat16, None
    loaded = []

    def create_weight(index, dtype, network_config, quant_cfg):
        assert network_config is model.config
        return object()

    def load_weights(self, weights):
        loaded.extend(self.trans_layers_weight)

    model.pre_and_post_weight_class = lambda dtype, network_config: object()
    model.transformer_weight_class = create_weight
    monkeypatch.setattr(Qwen3_5TpPartModel, "load_weights", load_weights)
    model._init_weights()
    model.load_weights({})
    assert len(loaded) == len(set(loaded)) == 32
    assert len(model.trans_layers_weight) == 80
    for execution, physical in enumerate(model.loop_layer_indices):
        assert model.trans_layers_weight[execution] is loaded[physical]
    assert model.config["n_layer"] == 80


@pytest.mark.parametrize("streams,count", [(2, 2), (4, 3)])
def test_mhc_registry_and_embedded_mtp_reuse_existing_qwen_draft(tmp_path, streams, count):
    from lightllm.models.draft_registry import get_draft_model_class
    from lightllm.models.sensenova_loop.model import SenseNovaLoopTpPartModel
    from lightllm.models.qwen3_5_mtp.model import Qwen3_5MTPModel

    config = _config(count=count)
    config["mhc"] = dict(start=4, end=28, streams=streams, iterations=20)
    config["text_config"].update(num_attention_heads=16, hidden_size=4096, vocab_size=128)
    config["text_config"]["layer_types"] = ["linear_attention"] * 3 + ["full_attention"]
    config["text_config"]["layer_types"] *= 8
    (tmp_path / "config.json").write_text(json.dumps(config))
    registration = ModelRegistry.get_model_config(config)
    assert registration.model_class.endswith(":SenseNovaLoopTpPartModel")
    assert registration.is_multimodal
    assert get_draft_model_class(config, "eagle_with_att") is Qwen3_5MTPModel

    target = object.__new__(SenseNovaLoopTpPartModel)
    target.weight_dir_, target.finetune_config, target.tp_world_size_ = str(tmp_path), None, 4
    target.args = SimpleNamespace(**dataclasses.asdict(StartArgs()))
    target._init_config()
    assert target.config["n_layer"] == 32 + 24 * (count - 1)
    assert target.num_physical_layers == 32

    draft = object.__new__(Qwen3_5MTPModel)
    draft.weight_dir_, draft.finetune_config, draft.tp_world_size_ = str(tmp_path), None, 4
    draft._init_config()
    assert draft.config["n_layer"] == draft.config["full_attention_interval"] == 1
    assert draft.config["layer_types"] == config["text_config"]["layer_types"]


def test_loop_tokenizer_loads_without_hf_loop_autoconfig(tmp_path):
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessorFast
    from lightllm.server.tokenizer import get_tokenizer

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"hello": 0, "[UNK]": 1}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(tmp_path)
    Qwen2VLImageProcessorFast().save_pretrained(tmp_path)
    config = _config()
    config.update(
        architectures=["SenseNovaLoopForConditionalGeneration"],
        vision_config={},
        vision_start_token_id=2,
        vision_end_token_id=3,
        image_token_id=4,
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    wrapped = get_tokenizer(str(tmp_path))
    assert wrapped.encode("hello") == [0]
    assert isinstance(wrapped.image_processor, Qwen2VLImageProcessorFast)


@pytest.mark.parametrize("custom_layout", [False, True])
def test_model_and_worker_use_the_same_loop_cache_layout(monkeypatch, custom_layout):
    from lightllm.models.sensenova_loop.model import SenseNovaLoopTpPartModel

    config = _config(start=1, end=6, count=2)
    if custom_layout:
        config["text_config"]["layer_types"] = ["full_attention", "linear_attention"] * 16
    original = deepcopy(config)
    args = StartArgs(tp=4, dp=1, data_type="bfloat16")
    monkeypatch.setattr("lightllm.common.state_cache_manager.linear_att.get_added_mtp_kv_layer_num", lambda: 1)
    monkeypatch.setattr("lightllm.common.state_cache_manager.linear_att.get_env_start_args", lambda: args)
    monkeypatch.setattr("transformers.configuration_utils.PretrainedConfig.get_config_dict", lambda _: (config, {}))
    monkeypatch.setattr(
        "lightllm.models.qwen3next.model.ReqManagerForMamba",
        lambda *args, linear_config: SimpleNamespace(linear_config=linear_config),
    )
    raw_cache = LinearAttCacheConfig.load_from_args()
    model = object.__new__(SenseNovaLoopTpPartModel)
    model.args = args
    model.max_req_num, model.batch_max_tokens, model.max_seq_length = 2, 8, 16
    model._init_req_manager()
    assert dataclasses.asdict(raw_cache) == dataclasses.asdict(model.linear_config)
    assert model.req_manager.linear_config is model.linear_config
    assert config == original


def test_plain_loop_registry_does_not_enable_mhc():
    registration = ModelRegistry.get_model_config(_config())
    assert registration.model_class == "lightllm.models.sensenova_loop.model:SenseNovaLoopTpPartModel"
    assert registration.is_multimodal


def test_weight_load_failure_restores_execution_weights(monkeypatch):
    from lightllm.models.sensenova_loop.model import SenseNovaLoopTpPartModel
    from lightllm.models.qwen3_5.model import Qwen3_5TpPartModel

    model = object.__new__(SenseNovaLoopTpPartModel)
    model.config = {"n_layer": 80}
    model.physical_layer_weights = [object()]
    model.trans_layers_weight = [object()]
    execution_config, execution_weights = model.config, model.trans_layers_weight

    def fail(*args):
        raise RuntimeError("weight load failed")

    monkeypatch.setattr(Qwen3_5TpPartModel, "load_weights", fail)
    with pytest.raises(RuntimeError, match="weight load failed"):
        model.load_weights({})
    assert model.config is execution_config
    assert model.trans_layers_weight is execution_weights


@pytest.fixture
def layer_runtime(monkeypatch):
    from lightllm.utils.envs_utils import get_env_start_args

    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(dataclasses.asdict(StartArgs())))
    for key, value in {
        "DP_WORLD_SIZE": 1,
        "CURRENT_RANK_IN_DP": 0,
        "CURRENT_DEVICE_ID": 0,
        "CURRENT_RANK_IN_NODE": 0,
    }.items():
        monkeypatch.setenv("LIGHTLLM_" + key, str(value))
    get_env_start_args.cache_clear()
    yield
    get_env_start_args.cache_clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mhc_enabled", [False, True])
def test_loop_layer_types_are_local_to_sensenova(layer_runtime, tmp_path, mhc_enabled):
    from lightllm.common.quantization import Quantcfg
    from lightllm.models.qwen3_5.layer_infer.transformer_layer_infer import Qwen35TransformerLayerInfer
    from lightllm.models.qwen3_5.layer_weights.transformer_layer_weight import Qwen35TransformerLayerWeight
    from lightllm.models.qwen3_5_mtp.model import Qwen3_5MTPModel
    from lightllm.models.sensenova_loop.layer_infer.transformer_layer_infer import SenseNovaLoopTransformerLayerInfer
    from lightllm.models.sensenova_loop.layer_weights.transformer_layer_weight import (
        SenseNovaLoopTransformerLayerWeight,
    )

    config = _config(layers=8, start=1, end=6, count=2)
    text = config["text_config"]
    text.update(
        n_layer=8,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=128,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        rms_norm_eps=1e-6,
        vocab_size=128,
        layer_types=["full_attention", "linear_attention"] * 4,
    )
    if mhc_enabled:
        text["mhc"] = dict(start=1, end=6, streams=2, iterations=20)
    indices = get_loop_layer_indices(config["loop_settings"], 8)
    execution_config = {
        **text,
        "loop_physical_indices": indices,
        "layer_types": get_loop_layer_types(config["loop_settings"], text),
    }
    weights = [
        SenseNovaLoopTransformerLayerWeight(i, torch.bfloat16, execution_config, Quantcfg(execution_config))
        for i in range(8)
    ]
    for execution, physical in enumerate(indices):
        layer = SenseNovaLoopTransformerLayerInfer(execution, execution_config)
        expected_linear = text["layer_types"][physical] == "linear_attention"
        assert layer.is_linear_attention_layer == weights[physical].is_linear_attention_layer == expected_linear
        assert hasattr(weights[physical], "linear_in_proj") == expected_linear
        assert hasattr(weights[physical], "qkvo_gate_proj") != expected_linear

    # The base model still uses its original periodic layout, even with layer_types present.
    base_layer = Qwen35TransformerLayerInfer(0, text)
    base_weight = Qwen35TransformerLayerWeight(0, torch.bfloat16, text, Quantcfg(text))
    assert base_layer.is_linear_attention_layer and base_weight.is_linear_attention_layer

    (tmp_path / "config.json").write_text(json.dumps(config))
    draft = object.__new__(Qwen3_5MTPModel)
    draft.weight_dir_, draft.finetune_config, draft.tp_world_size_ = str(tmp_path), None, 1
    draft._init_config()
    for global_index in (len(indices), len(indices) + 1):
        assert not Qwen35TransformerLayerInfer(global_index, draft.config).is_linear_attention_layer
