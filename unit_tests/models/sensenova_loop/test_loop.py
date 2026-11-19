import dataclasses
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from lightllm.models.registry import ModelRegistry
from lightllm.models.sensenova_loop.config import get_loop_layer_indices, get_loop_layer_types
from lightllm.models.qwen3next.config import Qwen3NextResourceConfig
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
            "num_attention_heads": 16,
            "head_dim": 256,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
        },
    }


@pytest.mark.parametrize("start,end,count", [(4, 28, 1), (4, 28, 3), (4, 28, 2), (1, 6, 2), (0, 32, 3)])
@pytest.mark.parametrize("cache_mode", ["independent", "overwrite"])
def test_cache_slots_follow_selected_mode_including_unaligned_loop(monkeypatch, start, end, count, cache_mode):
    config = _config(start=start, end=end, count=count)
    args = StartArgs(
        model_dir="unused",
        tp=4,
        dp=1,
        data_type="bfloat16",
        mtp_mode="eagle_with_att",
        mtp_step=3,
        loop_cache_mode=cache_mode,
    )
    monkeypatch.setattr("lightllm.models.qwen3next.config.get_env_start_args", lambda: args)
    monkeypatch.setattr("transformers.configuration_utils.PretrainedConfig.get_config_dict", lambda _: (config, {}))
    from lightllm.models.sensenova_loop.model import SenseNovaLoopTpPartModel

    normalized = SenseNovaLoopTpPartModel.normalize_model_config(config)
    assert SenseNovaLoopTpPartModel.normalize_model_config(normalized) == normalized
    indices = get_loop_layer_indices(config["loop_settings"], 32)
    assert normalized["n_layer"] == normalized["num_hidden_layers"] == len(indices)
    logical_ids = tuple(range(len(indices)))
    (full,) = SenseNovaLoopTpPartModel.create_full_att_states(normalized, args, logical_ids)
    (bounded,) = SenseNovaLoopTpPartModel.create_bounded_att_states(normalized, args, logical_ids)
    full_layers = tuple(group[0] for group in full.config.logical_layer_groups)
    linear_layers = bounded.config.logical_layer_ids
    kinds = get_loop_layer_types(config["loop_settings"], config["text_config"])
    assert len(kinds) == 32 + (end - start) * (count - 1)
    slots = {}
    for execution, (physical, kind) in enumerate(zip(indices, kinds)):
        layer_id = physical if cache_mode == "overwrite" else execution
        if kind == "full_attention":
            slot = full_layers.index(layer_id)
        else:
            slot = linear_layers.index(layer_id)
        visits = slots.setdefault(physical, set())
        if cache_mode == "independent":
            assert slot not in visits
        elif visits:
            assert slot in visits
        visits.add(slot)
    if cache_mode == "overwrite":
        assert (len(full_layers), len(linear_layers)) == (8, 24)
    else:
        assert len(full_layers) + len(linear_layers) == len(indices)
        assert len(full_layers) == kinds.count("full_attention")
    assert len(slots) == 32


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


@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("cache_mode", ["independent", "overwrite"])
def test_forward_executes_every_loop_visit(monkeypatch, prefill, cache_mode):
    from lightllm.common.basemodel.basemodel import dist_group_manager, g_cache_manager
    from lightllm.common.basemodel.hidden_collector import NoopHiddenCollector
    from lightllm.models.sensenova_loop.model import SenseNovaLoopTpPartModel

    config = _config(layers=8, start=1, end=6, count=2)
    config["text_config"].update(hidden_size=128, num_attention_heads=1, vocab_size=128)
    model = object.__new__(SenseNovaLoopTpPartModel)
    model.args = StartArgs(loop_cache_mode=cache_mode)
    model.tp_world_size_ = 1
    model.config = model.normalize_model_config(config)
    model.loop_layer_indices = model.config["loop_physical_indices"]
    model.physical_layer_weights = [SimpleNamespace(layer_num_=i) for i in range(8)]
    model.trans_layers_weight = [model.physical_layer_weights[i] for i in model.loop_layer_indices]
    visits = []
    cache_ids = []
    writes = []

    class Layer:
        def __init__(self, layer_num, network_config):
            self.layer_num_ = layer_num

        def forward(self, hidden, infer_state, weight):
            visits.append(weight.layer_num_)
            cache_ids.append(self.layer_num_)
            writes.append(getattr(self, "linear_att_write_state", True))
            return hidden + 1

        context_forward = token_forward = forward

    model.transformer_layer_infer_class = Layer
    model.pre_layer_infer_class = lambda **_: SimpleNamespace(
        context_forward=lambda input_ids, *_: input_ids,
        token_forward=lambda input_ids, *_: input_ids,
        _tpsp_sp_split=lambda input, **_: input,
    )
    model.post_layer_infer_class = lambda **_: SimpleNamespace(
        _tpsp_allgather=lambda input, **_: input,
        token_forward=lambda hidden, *_, **__: hidden,
    )
    model._init_infer_layer()
    model._init_some_value()
    model.pre_post_weight, model.prefill_graph = None, None
    model._create_model_output = lambda output, _: output
    monkeypatch.setattr(g_cache_manager, "cache_env_in", lambda: None)
    monkeypatch.setattr(g_cache_manager, "cache_env_out", lambda: None)
    monkeypatch.setattr(dist_group_manager, "clear_deepep_buffer", lambda: None)
    infer = SimpleNamespace(
        input_ids=torch.zeros(1, 1),
        hidden_collector=NoopHiddenCollector(),
        need_dp_prefill_balance=False,
        is_cuda_graph_capture=False,
    )
    output = model._context_forward(infer) if prefill else model._token_forward(infer)
    assert visits == [0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7]
    assert output.item() == 13
    assert cache_ids == (visits if cache_mode == "overwrite" else list(range(13)))
    assert writes == ([True] + [False] * 5 + [True] * 7 if cache_mode == "overwrite" else [True] * 13)


@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("write_state", [False, True])
def test_linear_layer_forwards_state_write_policy(monkeypatch, prefill, write_state):
    from lightllm.models.sensenova_loop.layer_infer.transformer_layer_infer import SenseNovaLoopTransformerLayerInfer

    layer = object.__new__(SenseNovaLoopTransformerLayerInfer)
    layer.is_linear_attention_layer = True
    layer.linear_att_write_state = write_state
    layer.layer_num_, layer.tp_world_size_ = 4, 1
    mixed = torch.ones(1, 4)
    layer._linear_in_proj = lambda *args: mixed
    layer._linear_post = lambda output, *args: output
    seen = []

    def attention(*args, att_control, **kwargs):
        params = att_control.linear_att_prefill_dict if prefill else att_control.linear_att_decode_dict
        seen.append(params)
        return mixed, None

    infer = SimpleNamespace(
        prefill_att_state1=SimpleNamespace(prefill_att=attention),
        decode_att_state1=SimpleNamespace(decode_att=attention),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    forward = layer.context_attention_forward if prefill else layer.token_attention_forward
    assert forward(mixed, infer, object()) is mixed
    assert len(seen) == 1
    assert seen[0]["write_state"] is write_state
    assert seen[0]["layer_num"] == 4


@pytest.mark.parametrize("factory", ["create_full_att_states", "create_bounded_att_states"])
def test_overwrite_requires_native_ssm(factory):
    from lightllm.models.sensenova_loop.model import SenseNovaLoopTpPartModel

    args = StartArgs(loop_cache_mode="overwrite", ssm_state_mode="replay")
    with pytest.raises(ValueError, match="requires ssm_state_mode=native"):
        getattr(SenseNovaLoopTpPartModel, factory)(_config(), args, tuple(range(80)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("cache_mode", ["independent", "overwrite"])
def test_loop_kv_writes_use_selected_cache_slots(cache_mode):
    from lightllm.models.sensenova_loop.model import SenseNovaLoopTpPartModel

    config = _config(layers=8, start=1, end=6, count=3)
    config["text_config"].update(num_key_value_heads=1, head_dim=32)
    config = SenseNovaLoopTpPartModel.normalize_model_config(config)
    (state,) = SenseNovaLoopTpPartModel.create_full_att_states(
        model_config=config,
        args=StartArgs(data_type="bfloat16", loop_cache_mode=cache_mode),
        model_logical_layer_ids=tuple(range(config["n_layer"])),
    )
    state.device = torch.device("cuda")
    state.allocate_storage(token_capacity=3)
    state.k_storage.zero_()
    state.v_storage.zero_()
    assert state.k_storage.shape[0] == state.v_storage.shape[0] == (2 if cache_mode == "overwrite" else 4)
    token_indices = torch.tensor([1], dtype=torch.int32, device="cuda")
    for execution, physical in enumerate(config["loop_physical_indices"]):
        if config["layer_types"][execution] == "linear_attention":
            continue
        kv = torch.full((1, 2, 32), execution + 1, dtype=torch.bfloat16, device="cuda")
        kv[:, 1].neg_()
        before_k, before_v = state.k_storage.clone(), state.v_storage.clone()
        layer_id = physical if cache_mode == "overwrite" else execution
        state.write_full_att_kv(layer_id, token_indices, kv)
        k, v = state.get_layer_view(layer_id)
        assert torch.equal(k[1], kv[0, :1])
        assert torch.equal(v[1], kv[0, 1:])
        other = [i for i in range(state.k_storage.shape[0]) if i != state.get_storage_layer_index(layer_id)]
        assert torch.equal(state.k_storage[other], before_k[other])
        assert torch.equal(state.v_storage[other], before_v[other])
    assert torch.count_nonzero(state.k_storage[:, [0, 2]]) == 0
    assert torch.count_nonzero(state.v_storage[:, [0, 2]]) == 0


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
    assert len(target.loop_layer_indices) == 32 + 24 * (count - 1)
    assert target.num_physical_layers == 32

    draft = object.__new__(Qwen3_5MTPModel)
    draft.weight_dir_, draft.finetune_config, draft.tp_world_size_ = str(tmp_path), None, 4
    draft._init_config()
    assert draft.config["n_layer"] == draft.config["full_attention_interval"] == 1
    assert draft.config["layer_types"] == ["full_attention"]


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
@pytest.mark.parametrize("cache_mode", ["independent", "overwrite"])
def test_model_and_worker_use_the_same_loop_cache_layout(monkeypatch, custom_layout, cache_mode):
    from lightllm.models.sensenova_loop.model import SenseNovaLoopTpPartModel

    config = _config(start=1, end=6, count=2)
    if custom_layout:
        config["text_config"]["layer_types"] = ["full_attention", "linear_attention"] * 16
    original = deepcopy(config)
    args = StartArgs(tp=4, dp=1, data_type="bfloat16", linear_att_hash_page_size=256, loop_cache_mode=cache_mode)
    monkeypatch.setattr("lightllm.models.qwen3next.config.get_env_start_args", lambda: args)
    monkeypatch.setattr("transformers.configuration_utils.PretrainedConfig.get_config_dict", lambda _: (config, {}))
    config["text_config"]["num_attention_heads"] = 16
    original = deepcopy(config)
    normalized = SenseNovaLoopTpPartModel.normalize_model_config(config)
    cache_config = config["text_config"] if cache_mode == "overwrite" else normalized
    raw_cache = Qwen3NextResourceConfig.from_model_config(cache_config, args)
    logical_ids = tuple(range(normalized["num_hidden_layers"]))
    (full,) = SenseNovaLoopTpPartModel.create_full_att_states(
        model_config=normalized,
        args=args,
        model_logical_layer_ids=logical_ids,
    )
    (bounded,) = SenseNovaLoopTpPartModel.create_bounded_att_states(
        model_config=normalized,
        args=args,
        model_logical_layer_ids=logical_ids,
    )
    full_layers = raw_cache.get_full_att_layer_indices()
    assert full.config.logical_layer_groups == tuple((i,) for i in full_layers)
    assert bounded.config.logical_layer_ids == tuple(i for i in range(raw_cache.all_layer_num) if i not in full_layers)
    runtime_ids = normalized["loop_physical_indices"] if cache_mode == "overwrite" else logical_ids
    (runtime_full,) = SenseNovaLoopTpPartModel.create_full_att_states(
        model_config=normalized,
        args=args,
        model_logical_layer_ids=runtime_ids,
    )
    (runtime_bounded,) = SenseNovaLoopTpPartModel.create_bounded_att_states(
        model_config=normalized,
        args=args,
        model_logical_layer_ids=runtime_ids,
    )
    assert runtime_full.config == full.config
    assert runtime_bounded.config == bounded.config
    assert config == original


def test_plain_loop_registry_does_not_enable_mhc():
    registration = ModelRegistry.get_model_config(_config())
    assert registration.model_class == "lightllm.models.sensenova_loop.model:SenseNovaLoopTpPartModel"
    assert registration.is_multimodal


def test_cache_mode_cli_preserves_independent_default():
    from lightllm.server.api_cli import make_argument_parser

    parser = make_argument_parser()
    assert parser.parse_args([]).loop_cache_mode == StartArgs().loop_cache_mode == "independent"
    for mode in ("independent", "overwrite"):
        args = StartArgs(**vars(parser.parse_args(["--loop_cache_mode", mode])))
        assert args.loop_cache_mode == mode
    with pytest.raises(SystemExit):
        parser.parse_args(["--loop_cache_mode", "unknown"])


def test_awq_config_applies_to_main_model_and_excludes_float_mtp():
    from lightllm.models.sensenova_loop.model import SenseNovaLoopTpPartModel
    from lightllm.models.qwen3_5_mtp.model import Qwen3_5MTPModel

    config = _config()
    quantization = {"quant_method": "awq", "bits": 4, "group_size": 128, "modules_to_not_convert": ["mtp"]}
    config["quantization_config"] = quantization
    original = deepcopy(config)
    assert SenseNovaLoopTpPartModel.normalize_model_config(config)["quantization_config"] == quantization
    assert "quantization_config" not in Qwen3_5MTPModel.normalize_model_config(config)
    assert config == original

    config["text_config"]["quantization_config"] = None
    assert SenseNovaLoopTpPartModel.normalize_model_config(config)["quantization_config"] == quantization


@pytest.mark.parametrize("suffix,factor", [("qweight", 8), ("qzeros", 8), ("scales", 1)])
def test_awq_fused_projections_split_on_output_axis(suffix, factor):
    from lightllm.models.qwen3_5.layer_weights.transformer_layer_weight import Qwen35TransformerLayerWeight

    method = SimpleNamespace(method_name="awq_marlin", pack_factor=8)
    layer = object.__new__(Qwen35TransformerLayerWeight)
    layer.layer_num_ = 0
    layer.linear_num_k_heads = 2
    layer.linear_num_v_heads = 4
    layer.linear_k_head_dim = layer.linear_v_head_dim = 16
    layer.linear_in_proj = SimpleNamespace(quant_method=method)
    prefix = "model.layers.0.linear_attn.in_proj_"
    qkv = torch.arange(256 // factor).reshape(2, 128 // factor)
    fused_name = f"{prefix}qkv.{suffix}"
    weights = {fused_name: qkv}
    layer._split_linear_in_proj_qkv(weights)
    q_end, k_end = 32 // factor, 64 // factor
    assert fused_name not in weights
    assert torch.equal(weights[f"{prefix}q.{suffix}"], qkv[:, :q_end])
    assert torch.equal(weights[f"{prefix}k.{suffix}"], qkv[:, q_end:k_end])
    assert torch.equal(weights[f"{prefix}v.{suffix}"], qkv[:, k_end:])

    layer.is_linear_attention_layer = False
    layer.q_head_num_, layer.head_dim = 2, 16
    layer.qkvo_gate_proj = SimpleNamespace(quant_method=method)
    layer._q_weight_name = "model.layers.0.self_attn.q_proj.weight"
    layer._o_gate_weight_name = "model.layers.0.self_attn.o_gate_proj.weight"
    q_name = f"model.layers.0.self_attn.q_proj.{suffix}"
    gate_name = f"model.layers.0.self_attn.o_gate_proj.{suffix}"
    head_width = layer.head_dim // factor
    q_gate = torch.arange(8 * head_width).reshape(2, 4 * head_width)
    q1, gate1, q2, gate2 = q_gate.split(head_width, dim=1)
    weights = {q_name: q_gate}
    layer._split_q_with_gate(weights)
    assert torch.equal(weights[q_name], torch.cat((q1, q2), dim=1))
    assert torch.equal(weights[gate_name], torch.cat((gate1, gate2), dim=1))
    layer._split_q_with_gate(weights)
    assert torch.equal(weights[q_name], torch.cat((q1, q2), dim=1))
    assert torch.equal(weights[gate_name], torch.cat((gate1, gate2), dim=1))


def test_loop_binds_split_gdn_projection_only_for_weights_with_separate_gates(monkeypatch):
    from lightllm.models.qwen3_5.model import Qwen3_5TpPartModel
    from lightllm.models.qwen3_5.layer_infer.transformer_layer_infer import Qwen35TransformerLayerInfer
    from lightllm.models.qwen3next.layer_infer.transformer_layer_infer import Qwen3NextTransformerLayerInfer
    from lightllm.models.sensenova_loop.model import SenseNovaLoopTpPartModel
    from lightllm.models.sensenova_loop.layer_infer.transformer_layer_infer import SenseNovaLoopTransformerLayerInfer

    qkvz = torch.arange(8).reshape(2, 4)
    gates = torch.arange(8, 12).reshape(2, 2)
    fused = torch.cat((qkvz, gates), dim=-1)
    fused_weight = SimpleNamespace(linear_in_proj=SimpleNamespace(mm=lambda _: fused))
    split_weight = SimpleNamespace(
        linear_in_proj=SimpleNamespace(mm=lambda _: qkvz),
        linear_gate_proj=SimpleNamespace(mm=lambda _: gates),
    )
    model = object.__new__(SenseNovaLoopTpPartModel)
    model.loop_layer_indices = (0, 1, 1)
    model.args = StartArgs()
    model.physical_layer_weights = [fused_weight, split_weight]
    model.trans_layers_weight = [model.physical_layer_weights[i] for i in model.loop_layer_indices]
    layers = [object.__new__(SenseNovaLoopTransformerLayerInfer) for _ in model.trans_layers_weight]
    for layer in layers:
        layer.embed_dim_ = 4
    monkeypatch.setattr(Qwen3_5TpPartModel, "_init_infer_layer", lambda self, _: setattr(self, "layers_infer", layers))

    model._init_infer_layer()
    assert Qwen35TransformerLayerInfer._linear_in_proj is Qwen3NextTransformerLayerInfer._linear_in_proj
    assert layers[0]._linear_in_proj.__func__ is Qwen3NextTransformerLayerInfer._linear_in_proj
    for layer in layers[1:]:
        assert layer._linear_in_proj.__func__ is SenseNovaLoopTransformerLayerInfer._linear_in_proj_with_gates
    assert model.layers_infer == layers
    for layer, weight in zip(model.layers_infer, model.trans_layers_weight):
        assert torch.equal(layer._linear_in_proj(torch.zeros(1, 2, 4), weight), fused)


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
