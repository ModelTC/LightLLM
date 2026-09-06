from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel import batch_objs
from lightllm.common.basemodel.batch_objs import ModelMtpOutputCollector, ModelOutput
from lightllm.models.qwen3_5_dflash.model import Qwen3_5DFlashModel
from lightllm.models.qwen3_5_dspark.model import Qwen3_5DSparkModel
from lightllm.models.qwen3_dflash.layer_infer import transformer_layer_infer as qwen3_dflash_layer_infer
from lightllm.models.qwen3_dflash.layer_infer.transformer_layer_infer import Qwen3DFlashTransformerLayerInfer
from lightllm.models.qwen3_dflash.model import Qwen3DFlashModel
from lightllm.models.qwen3_dspark.layer_weights import pre_and_post_layer_weight as dspark_pre_post_weight
from lightllm.models.qwen3_dspark.layer_infer.post_layer_infer import Qwen3DSparkPostLayerInfer
from lightllm.models.qwen3_dspark.model import Qwen3DSparkModel
from lightllm.models.llama.model import LlamaTpPartModel


@pytest.mark.parametrize("mtp_step", [1, 5, 7])
def test_parallel_block_runtime_width_follows_mtp_step(monkeypatch, mtp_step):
    monkeypatch.setattr(LlamaTpPartModel, "_verify_params", lambda self: None)
    model = Qwen3DFlashModel.__new__(Qwen3DFlashModel)
    model.args = SimpleNamespace(mtp_mode="dspark", mtp_step=mtp_step)
    model.config = {"block_size": 7}
    model.enable_tpsp_mix_mode = False

    model._verify_params()

    assert model.config["block_size"] == mtp_step


def test_parallel_block_rejects_mtp_step_above_checkpoint_capacity(monkeypatch):
    monkeypatch.setattr(LlamaTpPartModel, "_verify_params", lambda self: None)
    model = Qwen3DFlashModel.__new__(Qwen3DFlashModel)
    model.args = SimpleNamespace(mtp_mode="dflash", mtp_step=8)
    model.config = {"block_size": 7}
    model.enable_tpsp_mix_mode = False

    with pytest.raises(AssertionError):
        model._verify_params()


@pytest.mark.parametrize(
    "model_class",
    [Qwen3DFlashModel, Qwen3_5DFlashModel, Qwen3DSparkModel, Qwen3_5DSparkModel],
)
def test_parallel_block_decode_commits_target_hiddens_directly(model_class):
    model = model_class.__new__(model_class)
    model._cos_cached = torch.arange(24).view(6, 4)
    model._sin_cached = model._cos_cached + 100
    model.mem_manager = object()
    model.pre_post_weight = object()

    target_hiddens = torch.arange(6, dtype=torch.float32).view(2, 3)
    mem_indexes = torch.tensor([7, 11])
    observed_states = []

    class PreInfer:
        def context_forward(self, input_ids, infer_state, layer_weight):
            assert input_ids is None
            assert layer_weight is model.pre_post_weight
            assert infer_state.mtp_draft_input_hiddens is target_hiddens
            observed_states.append(infer_state)
            return target_hiddens + 1

    class TransformerLayer:
        def __init__(self, increment):
            self.increment = increment

        def context_forward(self, hidden, infer_state, layer_weight):
            assert infer_state is observed_states[0]
            assert layer_weight == self.increment
            return hidden + self.increment

    model.pre_infer = PreInfer()
    model.layers_infer = [TransformerLayer(2), TransformerLayer(3)]
    model.trans_layers_weight = [2, 3]
    model_input = SimpleNamespace(
        batch_size=2,
        b_seq_len=torch.tensor([3, 5]),
        mem_indexes=mem_indexes,
        mtp_draft_input_hiddens=target_hiddens,
    )

    output = model._decode(model_input)

    assert isinstance(output, ModelOutput)
    assert output.logits.shape == (2, 1)
    infer_state = observed_states[0]
    assert infer_state.mem_manager is model.mem_manager
    assert infer_state.mem_index is mem_indexes
    assert torch.equal(infer_state.position_cos, model._cos_cached[[2, 4]])
    assert torch.equal(infer_state.position_sin, model._sin_cached[[2, 4]])


@pytest.mark.parametrize("model_class", [Qwen3DSparkModel, Qwen3_5DSparkModel])
def test_dspark_decode_unpad_uses_common_output_and_slices_mtp_fields(model_class):
    model = model_class.__new__(model_class)
    output = ModelOutput(
        logits=torch.arange(48).view(12, 4),
        mtp_collector=ModelMtpOutputCollector(
            spec_hidden=torch.arange(36).view(12, 3),
            confidence_logits=torch.arange(12).view(3, 4),
            draft_token_ids=torch.arange(12),
        ),
    )

    unpadded = model._create_unpad_decode_model_output(output, origin_batch_size=8)

    assert isinstance(unpadded, ModelOutput)
    assert unpadded.logits.shape == (8, 4)
    assert unpadded.mtp_collector.spec_hidden.shape == (8, 3)
    assert unpadded.mtp_collector.confidence_logits.shape == (2, 4)
    assert unpadded.mtp_collector.draft_token_ids.shape == (8,)
    assert output.logits.shape == (12, 4)
    assert output.mtp_collector.confidence_logits.shape == (3, 4)
    assert output.mtp_collector.draft_token_ids.shape == (12,)


def test_common_output_no_ref_conversion_includes_dspark_fields(monkeypatch):
    monkeypatch.setattr(batch_objs, "tensor_to_no_ref_tensor", torch.clone)
    output = ModelOutput(
        logits=torch.ones((2, 4)),
        mtp_collector=ModelMtpOutputCollector(
            spec_hidden=torch.ones((2, 3)),
            confidence_logits=torch.ones((1, 2)),
            draft_token_ids=torch.ones((2,), dtype=torch.int64),
        ),
    )
    original_ptrs = (
        output.logits.data_ptr(),
        output.mtp_collector.spec_hidden.data_ptr(),
        output.mtp_collector.confidence_logits.data_ptr(),
        output.mtp_collector.draft_token_ids.data_ptr(),
    )

    output.to_no_ref_tensor()

    converted_ptrs = (
        output.logits.data_ptr(),
        output.mtp_collector.spec_hidden.data_ptr(),
        output.mtp_collector.confidence_logits.data_ptr(),
        output.mtp_collector.draft_token_ids.data_ptr(),
    )
    assert all(converted != original for converted, original in zip(converted_ptrs, original_ptrs))


def test_dspark_post_layer_publishes_head_results_through_collector():
    post_infer = Qwen3DSparkPostLayerInfer.__new__(Qwen3DSparkPostLayerInfer)
    post_infer.block_size_ = 2
    post_infer.markov_rank_ = 2
    post_infer._slice_get_last_input = lambda input_embeddings, infer_state: (
        input_embeddings,
        4,
    )
    post_infer._norm = lambda hidden, infer_state, layer_weight: hidden
    post_infer._sample_markov = lambda *args, **kwargs: torch.tensor([[1, 2], [3, 4]])
    confidence_logits = torch.arange(4, dtype=torch.float32).view(2, 2)
    post_infer.predict_confidence_logits = lambda *args, **kwargs: confidence_logits

    class RecordingCollector:
        def add_mtp_outputs(self, **kwargs):
            self.outputs = kwargs

    class LMHead:
        def __call__(self, input, alloc_func):
            return torch.empty((8, input.shape[1]))

    collector = RecordingCollector()
    infer_state = SimpleNamespace(
        is_prefill=False,
        input_ids=torch.tensor([10, 0, 20, 0]),
        hidden_collector=collector,
    )
    layer_weight = SimpleNamespace(lm_head_weight_=LMHead())

    logits = post_infer.token_forward(
        input_embdings=torch.randn(4, 3),
        infer_state=infer_state,
        layer_weight=layer_weight,
    )

    assert logits.shape == (4, 1)
    assert torch.equal(collector.outputs["draft_token_ids"], torch.tensor([1, 2, 3, 4]))
    assert collector.outputs["confidence_logits"] is confidence_logits


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
                "factor": 32.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "yarn",
            },
        }

    monkeypatch.setattr(Qwen3DSparkModel, "_init_config", init_dspark_config)
    model = Qwen3_5DSparkModel.__new__(Qwen3_5DSparkModel)

    model._init_config()

    assert model.config["rope_scaling"] == model.config["rope_parameters"]
    assert model.config["rope_theta"] == 1_000_000
    assert model.config["partial_rotary_factor"] == 1.0
    assert model.config["mask_token_id"] == 1


def test_qwen35_dspark_adapter_preserves_custom_partial_rotary_layout(monkeypatch):
    def init_dspark_config(self):
        self.config = {
            "dflash_config": {"mask_token_id": 1},
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10_000_000,
                "rope_type": "default",
            },
        }

    monkeypatch.setattr(Qwen3DSparkModel, "_init_config", init_dspark_config)
    model = Qwen3_5DSparkModel.__new__(Qwen3_5DSparkModel)

    model._init_config()

    assert model.config["partial_rotary_factor"] == 0.25
    assert model.config["rope_scaling"] == model.config["rope_parameters"]


def test_qwen3_parallel_block_draft_applies_partial_rotary_factor(monkeypatch):
    rotary_factors = []
    monkeypatch.setattr(qwen3_dflash_layer_infer, "qk_rmsnorm_forward", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        qwen3_dflash_layer_infer,
        "rotary_emb_fwd",
        lambda *args, **kwargs: rotary_factors.append(kwargs["partial_rotary_factor"]),
    )

    class Projection:
        def __init__(self, width):
            self.width = width

        def mm(self, input, **kwargs):
            return input.new_zeros((input.shape[0], self.width))

    class QKNorm:
        k_weight = object()

        def __call__(self, *args, **kwargs):
            return None

    layer = Qwen3DFlashTransformerLayerInfer.__new__(Qwen3DFlashTransformerLayerInfer)
    layer.tp_q_head_num_ = 1
    layer.tp_k_head_num_ = 1
    layer.tp_v_head_num_ = 1
    layer.head_dim_ = 8
    layer.eps_ = 1e-6
    layer.partial_rotary_factor = 0.25
    layer._post_cache_kv = lambda *args, **kwargs: None
    infer_state = SimpleNamespace(position_cos=object(), position_sin=object())
    layer_weight = SimpleNamespace(
        q_proj=Projection(8),
        kv_proj=Projection(16),
        qk_norm_weight_=QKNorm(),
    )
    inputs = torch.zeros((2, 4))

    layer.context_forward(inputs, infer_state, layer_weight)
    layer._get_qkv(inputs, infer_state, layer_weight)

    assert rotary_factors == [0.25, 0.25]


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
