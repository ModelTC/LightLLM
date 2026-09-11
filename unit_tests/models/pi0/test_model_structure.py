import importlib.util
from dataclasses import replace
from types import SimpleNamespace

import torch

from lightllm.common.basemodel import PostLayerInferTpl, PreLayerInferTpl
from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.basemodel.multimodal_tokenizer import BaseMultiModalTokenizer
from lightllm.models.gemma_2b.layer_infer.transformer_layer_infer import (
    Gemma_2bTransformerLayerInfer,
)
from lightllm.models.gemma_2b.layer_weights.transformer_layer_weight import (
    Gemma_2bTransformerLayerWeight,
)
from lightllm.models.gemma_2b.layer_weights import (
    transformer_layer_weight as gemma_weight_module,
)
from lightllm.models.gemma_2b.triton_kernel.gelu_and_mul import gelu_and_mul_fwd
from lightllm.models.gemma_2b.model import Gemma_2bTpPartModel
from lightllm.models.gemma4.layer_infer.pre_layer_infer import Gemma4PreLayerInfer
from lightllm.models.llama.layer_weights.transformer_layer_weight import (
    LlamaTransformerLayerWeight,
)
from lightllm.models.pi0.infer_struct import Pi0ActionInferStateInfo
from lightllm.models.pi0.layer_infer import (
    Pi0ActionPostLayerInfer,
    Pi0ActionPreLayerInfer,
    Pi0ActionTransformerLayerInfer,
)
from lightllm.models.pi0.layer_weights import (
    Pi0ActionPreAndPostLayerWeight,
    Pi0ActionTransformerLayerWeight,
    Pi0VLMTransformerLayerWeight,
)
from lightllm.models.pi0.layer_infer import (
    transformer_layer_infer as pi0_transformer_infer,
)
from lightllm.models.pi0.model import (
    PI_VOCAB_SIZE,
    Pi0ActionExpertModel,
    Pi0VLMModel,
)
from lightllm.models.pi0.tokenizer import Pi0TokenizerAdapter
from lightllm.models.pi05.layer_infer import (
    Pi05ActionPostLayerInfer,
    Pi05ActionPreLayerInfer,
    Pi05ActionTransformerLayerInfer,
)
from lightllm.models.pi05.layer_weights import (
    Pi05ActionPreAndPostLayerWeight,
    Pi05ActionTransformerLayerWeight,
)
from lightllm.models.pi05.model import Pi05ActionExpertModel, Pi05VLMModel
from lightllm.models.registry import get_generation_model_class, get_model_class
import lightllm.models.pi0.model as pi0_model_module


def test_vlm_is_a_native_gemma_basemodel():
    assert issubclass(Pi0VLMModel, Gemma_2bTpPartModel)
    assert Pi0VLMModel.pre_layer_infer_class is Gemma4PreLayerInfer
    assert Pi0VLMModel.transformer_layer_infer_class is Gemma_2bTransformerLayerInfer
    assert Pi0VLMModel._init_req_manager is TpPartBaseModel._init_req_manager
    assert Pi0VLMModel._init_mem_manager is Gemma_2bTpPartModel._init_mem_manager
    assert Pi0VLMModel.prefill_causal is False
    assert PI_VOCAB_SIZE == 257_152


def test_pi0_weights_use_standard_meta_weight_containers():
    assert issubclass(Pi0VLMTransformerLayerWeight, LlamaTransformerLayerWeight)
    assert issubclass(Pi0VLMTransformerLayerWeight, Gemma_2bTransformerLayerWeight)
    assert issubclass(Pi0ActionTransformerLayerWeight, Gemma_2bTransformerLayerWeight)
    assert issubclass(Pi0ActionTransformerLayerWeight, Pi0VLMTransformerLayerWeight)
    assert issubclass(
        Pi05ActionPreAndPostLayerWeight,
        Pi0ActionPreAndPostLayerWeight,
    )
    assert pi0_transformer_infer.gelu_and_mul_fwd is gelu_and_mul_fwd


def test_shared_gemma_norm_initializer_matches_current_meta_weight_api(monkeypatch):
    created = []

    def fake_norm(*, dim, weight_name, data_type):
        value = SimpleNamespace(
            dim=dim,
            weight_name=weight_name,
            data_type=data_type,
        )
        created.append(value)
        return value

    monkeypatch.setattr(gemma_weight_module, "NoTpGEMMANormWeight", fake_norm)
    weight = object.__new__(Gemma_2bTransformerLayerWeight)
    weight.n_embed = 16
    weight.data_type_ = torch.float32
    weight._att_norm_weight_name = "attention_norm.weight"
    weight._ffn_norm_weight_name = "ffn_norm.weight"

    weight._init_norm()

    assert [value.weight_name for value in created] == [
        "attention_norm.weight",
        "ffn_norm.weight",
    ]


def test_action_expert_uses_standard_layer_infer_components():
    assert not issubclass(Pi0ActionInferStateInfo, InferStateInfo)
    assert issubclass(Pi0ActionPreLayerInfer, PreLayerInferTpl)
    assert issubclass(Pi0ActionPostLayerInfer, PostLayerInferTpl)
    assert issubclass(Pi0ActionTransformerLayerInfer, Gemma_2bTransformerLayerInfer)
    assert issubclass(Pi0TokenizerAdapter, BaseMultiModalTokenizer)


def test_action_infer_state_copies_dynamic_graph_inputs():
    group = object()

    class PrefixKV:
        def __init__(self, value):
            self.value = value

        def copy_from(self, source):
            self.value = source.value

    state = Pi0ActionInferStateInfo(
        batch_size=1,
        suffix_length=2,
        prefix_kv=PrefixKV(1),
        prefix_lengths=torch.tensor([3], dtype=torch.int32),
        position_cos=torch.zeros((2, 2)),
        position_sin=torch.zeros((2, 2)),
        state=torch.zeros((1, 2)),
        noise=torch.zeros((1, 1, 2)),
        action_horizon=1,
        time_embeddings=torch.zeros((1, 2)),
        dt=torch.tensor(-1.0),
        first_suffix_token_isolated=True,
        dist_group=group,
    )

    graph_state = state
    replacement_group = object()
    source = replace(
        state,
        state=torch.full_like(state.state, 3),
        noise=torch.full_like(state.noise, 4),
        prefix_kv=PrefixKV(2),
        dist_group=replacement_group,
    )
    graph_state.copy_for_cuda_graph(source)
    assert graph_state.dist_group is replacement_group
    assert graph_state.prefix_kv.value == 2
    torch.testing.assert_close(graph_state.state, source.state)
    torch.testing.assert_close(graph_state.noise, source.noise)


def test_pi0_and_pi05_have_fixed_model_paths():
    assert get_model_class({"model_type": "pi0"}) is Pi0VLMModel
    assert get_model_class({"model_type": "pi05"}) is Pi05VLMModel
    assert (
        Pi0ActionExpertModel.transformer_layer_infer_class
        is Pi0ActionTransformerLayerInfer
    )
    assert (
        Pi05ActionExpertModel.transformer_layer_infer_class
        is Pi05ActionTransformerLayerInfer
    )
    assert Pi05ActionExpertModel.pre_layer_infer_class is Pi05ActionPreLayerInfer
    assert Pi05ActionExpertModel.post_layer_infer_class is Pi05ActionPostLayerInfer
    assert (
        Pi05ActionExpertModel.transformer_weight_class
        is Pi05ActionTransformerLayerWeight
    )
    assert (
        get_generation_model_class({"model_type": "pi0"}, "action")
        is Pi0ActionExpertModel
    )
    assert (
        get_generation_model_class({"model_type": "pi05"}, "action")
        is Pi05ActionExpertModel
    )
    assert not hasattr(Pi0VLMModel, "generation_model_class")
    assert not hasattr(Pi05VLMModel, "generation_model_class")


def test_pi0_has_no_parallel_attention_or_kv_runtime_modules():
    for module_name in (
        "action_cuda_graph",
        "attention_plan",
        "generation",
        "kv_cache",
        "runtime_ops",
    ):
        assert importlib.util.find_spec(f"lightllm.models.pi0.{module_name}") is None

    assert not hasattr(pi0_model_module, "_Pi0ActionRuntime")
    assert not hasattr(Pi0ActionExpertModel, "sample_actions")


def test_action_forward_respects_runtime_cuda_graph_switch(monkeypatch):
    class FakeEvent:
        def record(self):
            pass

    class FakeGraph:
        def __init__(self):
            self.run_count = 0

        def needs_capture(self, _key):
            return True

        def run(self, _key, state, forward):
            self.run_count += 1
            return forward(state)

    monkeypatch.setattr(
        pi0_model_module.torch.cuda,
        "Event",
        lambda enable_timing: FakeEvent(),
    )
    model = Pi0ActionExpertModel.__new__(Pi0ActionExpertModel)
    model.vla_config = {
        "num_denoise_steps": 1,
        "max_action_dim": 2,
    }
    model.device = torch.device("cpu")
    model.data_type = torch.float32
    model.graph = FakeGraph()
    model._prepare_state = lambda state: state
    model._validate_action_metadata = lambda **_kwargs: None
    model._create_infer_state = lambda **kwargs: SimpleNamespace(**kwargs)
    model._context_forward = lambda infer_state: infer_state.noise
    inputs = {
        "prefix_kv": SimpleNamespace(batch_size=1, layout_key=("dense", 3)),
        "state": torch.zeros((1, 2)),
        "noise": torch.ones((1, 1, 2)),
    }

    model.enable_cuda_graph = False
    eager_output = model.forward(**inputs)
    assert model.graph.run_count == 0

    model.enable_cuda_graph = True
    graph_output = model.forward(**inputs)
    assert model.graph.run_count == 1
    torch.testing.assert_close(eager_output.actions, graph_output.actions)


def test_infer_state_keeps_text_prefill_causal_by_default():
    state = InferStateInfo()
    assert state.prefill_causal is True
    assert state.use_ieee_fp32_attention is False


def test_vlm_uses_the_shared_server_vla_config(monkeypatch):
    expected = {
        "action_dim": 1,
        "max_action_dim": 1,
        "action_horizon": 1,
        "num_denoise_steps": 1,
        "image_resolution": (224, 224),
        "tokenizer_max_length": 1,
        "model_dir": "/unused",
        "checkpoint_path": "/unused",
    }
    start_args = SimpleNamespace(generation_runtime_config=expected)
    monkeypatch.setattr(pi0_model_module, "get_env_start_args", lambda: start_args)
    monkeypatch.setattr(
        pi0_model_module, "with_pi_config_overrides", lambda config: config
    )
    monkeypatch.setattr(Gemma_2bTpPartModel, "__init__", lambda self, kvargs: None)

    model = Pi0VLMModel({"weight_dir": "/unused"})

    assert model.vla_config is expected
