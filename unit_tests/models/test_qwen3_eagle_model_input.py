from types import SimpleNamespace

import pytest
import torch

from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.qwen3_eagle.layer_infer.pre_layer_infer import Qwen3EaglePreLayerInfer
from lightllm.models.qwen3_eagle.layer_infer.transformer_layer_infer import Qwen3EagleTransformerLayerInfer
from lightllm.models.qwen3_eagle.model import Qwen3EagleModel


def test_qwen3_eagle_uses_configured_attention_head_dim(monkeypatch):
    def init_llama_layer(self, layer_num, network_config):
        self.head_dim_ = network_config["hidden_size"] // network_config["num_attention_heads"]

    monkeypatch.setattr(LlamaTransformerLayerInfer, "__init__", init_llama_layer)
    layer = Qwen3EagleTransformerLayerInfer(
        layer_num=0,
        network_config={
            "hidden_size": 2560,
            "num_attention_heads": 32,
            "head_dim": 128,
        },
    )

    assert layer.head_dim_ == 128


def test_qwen3_eagle_projects_concatenated_target_hiddens_before_inferstate(monkeypatch):
    parent_calls = []
    monkeypatch.setattr(
        LlamaTpPartModel,
        "_create_inferstate",
        lambda self, model_input, microbatch_index=0: parent_calls.append((model_input, microbatch_index))
        or SimpleNamespace(model_input=model_input),
    )
    projected_hiddens = torch.empty((3, 4))
    projection_inputs = []
    model = Qwen3EagleModel.__new__(Qwen3EagleModel)
    model.config = {"hidden_size": 4}
    model.pre_post_weight = SimpleNamespace(
        fc_weight_=SimpleNamespace(
            mm=lambda hiddens: projection_inputs.append(hiddens) or projected_hiddens,
        )
    )
    target_hiddens = torch.empty((3, 12))
    model_input = SimpleNamespace(
        input_ids=torch.arange(3),
        mtp_draft_input_hiddens=target_hiddens,
    )

    infer_state = model._create_inferstate(model_input=model_input, microbatch_index=1)

    assert projection_inputs == [target_hiddens]
    assert len(parent_calls) == 1
    normalized_input, microbatch_index = parent_calls[0]
    assert normalized_input is not model_input
    assert normalized_input.mtp_draft_input_hiddens is projected_hiddens
    assert model_input.mtp_draft_input_hiddens is target_hiddens
    assert microbatch_index == 1
    assert infer_state.model_input is normalized_input


def test_qwen3_eagle_keeps_recursive_draft_hiddens_without_projection(monkeypatch):
    monkeypatch.setattr(
        LlamaTpPartModel,
        "_create_inferstate",
        lambda self, model_input, microbatch_index=0: model_input,
    )
    model = Qwen3EagleModel.__new__(Qwen3EagleModel)
    model.config = {"hidden_size": 4}
    model.pre_post_weight = SimpleNamespace(
        fc_weight_=SimpleNamespace(
            mm=lambda _: pytest.fail("fixed-width recursive hidden must not be projected"),
        )
    )
    model_input = SimpleNamespace(
        input_ids=torch.arange(3),
        mtp_draft_input_hiddens=torch.empty((3, 4)),
    )

    normalized_input = model._create_inferstate(model_input=model_input)

    assert normalized_input is model_input


def test_qwen3_eagle_pre_layer_rejects_non_normalized_hidden_width():
    pre_layer = Qwen3EaglePreLayerInfer.__new__(Qwen3EaglePreLayerInfer)
    pre_layer.hidden_size_ = 4
    infer_state = SimpleNamespace(mtp_draft_input_hiddens=torch.empty((2, 12)))

    with pytest.raises(AssertionError):
        pre_layer.prepare_spec_draft_hiddens(infer_state)

    normalized_hiddens = torch.empty((2, 4))
    infer_state.mtp_draft_input_hiddens = normalized_hiddens
    pre_layer.prepare_spec_draft_hiddens(infer_state)

    assert infer_state.eagle_draft_hidden_states is normalized_hiddens
