from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel import batch_objs
from lightllm.models.qwen3_5_dspark.model import Qwen3_5DSparkModel
from lightllm.models.qwen3_dspark import model_output as dspark_model_output
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
