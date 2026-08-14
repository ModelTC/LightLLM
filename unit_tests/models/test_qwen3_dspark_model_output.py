from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel import batch_objs
from lightllm.models.qwen3_5_dspark.model import Qwen3_5DSparkModel
from lightllm.models.qwen3_dflash.model import Qwen3DFlashModel
from lightllm.models.qwen3_dspark import model_output as dspark_model_output
from lightllm.models.qwen3_dspark.layer_weights import pre_and_post_layer_weight as dspark_pre_post_weight
from lightllm.models.qwen3_dspark.layer_infer.post_layer_infer import Qwen3DSparkPostLayerInfer
from lightllm.models.qwen3_dspark.model import Qwen3DSparkModel
from lightllm.models.qwen3_dspark.model_output import DSparkModelOutput


@pytest.mark.parametrize("model_class", [Qwen3DSparkModel, Qwen3_5DSparkModel])
def test_dspark_decode_unpad_preserves_output_type_and_slices_dspark_fields(model_class):
    model = model_class.__new__(model_class)
    output = DSparkModelOutput(
        logits=torch.arange(48).view(12, 4),
        spec_hidden=torch.arange(36).view(12, 3),
        confidence_logits=torch.arange(12).view(3, 4),
        draft_token_ids=torch.arange(12),
    )

    unpadded = model._create_unpad_decode_model_output(output, origin_batch_size=8)

    assert isinstance(unpadded, DSparkModelOutput)
    assert unpadded.logits.shape == (8, 4)
    assert unpadded.spec_hidden.shape == (8, 3)
    assert unpadded.confidence_logits.shape == (2, 4)
    assert unpadded.draft_token_ids.shape == (8,)
    assert output.logits.shape == (12, 4)
    assert output.confidence_logits.shape == (3, 4)
    assert output.draft_token_ids.shape == (12,)


def test_dspark_no_ref_conversion_dispatches_to_dspark_fields(monkeypatch):
    monkeypatch.setattr(batch_objs, "tensor_to_no_ref_tensor", torch.clone)
    monkeypatch.setattr(dspark_model_output, "tensor_to_no_ref_tensor", torch.clone)
    output = DSparkModelOutput(
        logits=torch.ones((2, 4)),
        spec_hidden=torch.ones((2, 3)),
        confidence_logits=torch.ones((1, 2)),
        draft_token_ids=torch.ones((2,), dtype=torch.int64),
    )
    original_ptrs = (
        output.logits.data_ptr(),
        output.spec_hidden.data_ptr(),
        output.confidence_logits.data_ptr(),
        output.draft_token_ids.data_ptr(),
    )

    output.to_no_ref_tensor()

    converted_ptrs = (
        output.logits.data_ptr(),
        output.spec_hidden.data_ptr(),
        output.confidence_logits.data_ptr(),
        output.draft_token_ids.data_ptr(),
    )
    assert all(converted != original for converted, original in zip(converted_ptrs, original_ptrs))


@pytest.mark.parametrize(
    "head_config",
    [
        {"enable_confidence_head": True},
        {"markov_rank": 4, "markov_head_type": "gated"},
        {"markov_rank": 4, "markov_head_type": "rnn"},
    ],
)
def test_biased_dspark_heads_do_not_inherit_model_quantization(monkeypatch, head_config):
    captured_kwargs = []

    def init_base_weight(self, data_type, network_config, quant_cfg):
        self.data_type_ = data_type
        self.quant_cfg = quant_cfg

    class RecordingROWMMWeight:
        def __init__(self, **kwargs):
            captured_kwargs.append(kwargs)

    class StubWeight:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(
        dspark_pre_post_weight.Qwen3DFlashPreAndPostLayerWeight,
        "__init__",
        init_base_weight,
    )
    monkeypatch.setattr(dspark_pre_post_weight, "EmbeddingWeight", StubWeight)
    monkeypatch.setattr(dspark_pre_post_weight, "LMHeadWeight", StubWeight)
    monkeypatch.setattr(dspark_pre_post_weight, "ROWMMWeight", RecordingROWMMWeight)

    quant_method = object()
    quant_cfg = SimpleNamespace(get_quant_method=lambda *_: quant_method)
    network_config = {
        "hidden_size": 16,
        "vocab_size": 32,
        **head_config,
    }
    dspark_pre_post_weight.Qwen3DSparkPreAndPostLayerWeight(
        data_type=torch.bfloat16,
        network_config=network_config,
        quant_cfg=quant_cfg,
    )

    assert captured_kwargs
    assert all(kwargs["quant_method"] is None for kwargs in captured_kwargs)


def test_fixed_dspark_does_not_require_confidence_head(monkeypatch):
    monkeypatch.setattr(Qwen3DFlashModel, "_verify_params", lambda self: None)
    model = Qwen3DSparkModel.__new__(Qwen3DSparkModel)
    model.config = {"enable_confidence_head": False}

    model._verify_params()


def test_qwen35_dspark_adapter_uses_current_checkpoint_rope_layout(monkeypatch):
    def init_dspark_config(self):
        self.config = {
            "dflash_config": {"mask_token_id": 1},
            "rope_parameters": {
                "rope_theta": 1_000_000,
                "partial_rotary_factor": 0.25,
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "rope_type": "default",
            },
        }

    monkeypatch.setattr(Qwen3DSparkModel, "_init_config", init_dspark_config)
    model = Qwen3_5DSparkModel.__new__(Qwen3_5DSparkModel)

    model._init_config()

    assert model.config["rope_scaling"] is None
    assert model.config["rope_theta"] == 1_000_000
    assert model.config["partial_rotary_factor"] == 1.0
    assert model.config["mask_token_id"] == 1


def test_vanilla_markov_local_sampling_matches_full_logits():
    post_infer = Qwen3DSparkPostLayerInfer.__new__(Qwen3DSparkPostLayerInfer)
    post_infer.block_size_ = 3
    post_infer.markov_rank_ = 2
    post_infer.markov_head_type_ = "vanilla"
    post_infer.tp_world_size_ = 1

    vocab_size = 5
    request_count = 2
    local_logits = torch.randn(vocab_size, request_count * post_infer.block_size_)

    class MarkovEmbedding:
        def __init__(self, weight):
            self.weight = weight

        def __call__(self, input_ids, alloc_func):
            return torch.nn.functional.embedding(input_ids, self.weight)

    class MarkovLMHead:
        def __init__(self, weight):
            self.weight = weight
            self.tp_vocab_start_id = 0

        def __call__(self, input, alloc_func):
            return self.weight @ input

    markov_w1 = torch.randn(vocab_size, post_infer.markov_rank_)
    markov_w2 = torch.randn(vocab_size, post_infer.markov_rank_)
    layer_weight = SimpleNamespace(
        markov_w1_weight_=MarkovEmbedding(markov_w1),
        markov_w2_weight_=MarkovLMHead(markov_w2),
    )
    anchor_token_ids = torch.tensor([1, 3])
    block_hidden = torch.empty(request_count, post_infer.block_size_, 0)
    post_infer.alloc_tensor = torch.empty

    sampled_tokens = post_infer._sample_markov(
        local_logits=local_logits,
        block_hidden=block_hidden,
        infer_state=SimpleNamespace(),
        anchor_token_ids=anchor_token_ids,
        layer_weight=layer_weight,
    )

    base_logits = local_logits.T.reshape(request_count, post_infer.block_size_, vocab_size)
    prev_token_ids = anchor_token_ids
    expected_tokens = []
    for step_idx in range(post_infer.block_size_):
        markov_bias = torch.nn.functional.linear(markov_w1[prev_token_ids], markov_w2)
        prev_token_ids = torch.argmax(base_logits[:, step_idx] + markov_bias, dim=-1)
        expected_tokens.append(prev_token_ids)
    expected_tokens = torch.stack(expected_tokens, dim=1)

    torch.testing.assert_close(sampled_tokens, expected_tokens)


def test_vanilla_markov_tp4_sampling_matches_full_logits(monkeypatch):
    torch.manual_seed(0)
    post_infer = Qwen3DSparkPostLayerInfer.__new__(Qwen3DSparkPostLayerInfer)
    post_infer.block_size_ = 3
    post_infer.markov_rank_ = 4
    post_infer.markov_head_type_ = "vanilla"
    post_infer.tp_world_size_ = 4
    post_infer.alloc_tensor = torch.empty

    vocab_size = 11
    request_count = 2
    split_indexes = torch.linspace(0, vocab_size, post_infer.tp_world_size_ + 1, dtype=torch.int64)
    tp_rank = 2
    local_start = int(split_indexes[tp_rank])
    local_end = int(split_indexes[tp_rank + 1])

    base_logits = torch.randn(request_count, post_infer.block_size_, vocab_size)
    markov_w1 = torch.randn(vocab_size, post_infer.markov_rank_)
    markov_w2 = torch.randn(vocab_size, post_infer.markov_rank_)
    anchor_token_ids = torch.tensor([1, 7])

    class MarkovEmbedding:
        weight = markov_w1

    class MarkovLMHead:
        weight = markov_w2[local_start:local_end]
        tp_vocab_start_id = local_start

    layer_weight = SimpleNamespace(
        markov_w1_weight_=MarkovEmbedding(),
        markov_w2_weight_=MarkovLMHead(),
    )
    local_logits = base_logits[:, :, local_start:local_end].reshape(-1, local_end - local_start).T.contiguous()

    expected_tokens = []
    prev_token_ids = anchor_token_ids
    for step_idx in range(post_infer.block_size_):
        scores = base_logits[:, step_idx] + torch.nn.functional.linear(markov_w1[prev_token_ids], markov_w2)
        prev_token_ids = torch.argmax(scores, dim=-1)
        expected_tokens.append(prev_token_ids)
    expected_tokens = torch.stack(expected_tokens, dim=1)

    step = 0

    def gather_tp_winners(output, local_winners, group, async_op):
        nonlocal step
        prev_tokens = anchor_token_ids if step == 0 else expected_tokens[:, step - 1]
        scores = base_logits[:, step] + torch.nn.functional.linear(markov_w1[prev_tokens], markov_w2)
        winners = []
        for rank in range(post_infer.tp_world_size_):
            start = int(split_indexes[rank])
            end = int(split_indexes[rank + 1])
            values, indexes = scores[:, start:end].max(dim=-1)
            winners.append(torch.stack((values, (indexes + start).float()), dim=-1))
        torch.testing.assert_close(local_winners, winners[tp_rank])
        output.copy_(torch.stack(winners).reshape_as(output))
        step += 1

    monkeypatch.setattr(
        "lightllm.models.qwen3_dspark.layer_infer.post_layer_infer.all_gather_into_tensor",
        gather_tp_winners,
    )

    sampled_tokens = post_infer._sample_markov(
        local_logits=local_logits,
        block_hidden=torch.empty(request_count, post_infer.block_size_, 0),
        infer_state=SimpleNamespace(dist_group=None),
        anchor_token_ids=anchor_token_ids,
        layer_weight=layer_weight,
    )

    torch.testing.assert_close(sampled_tokens, expected_tokens)
