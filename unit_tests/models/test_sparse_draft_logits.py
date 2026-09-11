from types import SimpleNamespace

import torch

from lightllm.common.basemodel.batch_objs import ModelOutput
from lightllm.models.llama.layer_infer import post_layer_infer as llama_post_layer
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from lightllm.models.qwen3_dspark.layer_infer.post_layer_infer import Qwen3DSparkPostLayerInfer
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend


def test_draft_logits_topk_comes_only_from_environment(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_DP", "0")
    monkeypatch.setenv("LIGHTLLM_DP_WORLD_SIZE", "1")
    config = {
        "rms_norm_eps": 1e-6,
        "vocab_size": 1024,
        "n_embed": 64,
        "draft_logits_topk": 7,
    }

    monkeypatch.setenv("LIGHTLLM_DRAFT_LOGITS_TOPK", "32")
    assert LlamaPostLayerInfer(config).draft_logits_topk_ == 32

    monkeypatch.delenv("LIGHTLLM_DRAFT_LOGITS_TOPK")
    assert LlamaPostLayerInfer(config).draft_logits_topk_ == LlamaPostLayerInfer.DEFAULT_DRAFT_LOGITS_TOPK


def test_sparse_draft_logits_gather_local_candidates_and_token_ids(monkeypatch):
    post = LlamaPostLayerInfer.__new__(LlamaPostLayerInfer)
    post.tp_world_size_ = 2
    post.draft_logits_topk_ = 2
    post.alloc_tensor = lambda shape, dtype, device="cpu": torch.empty(shape, dtype=dtype, device=device)

    # Rank 0 owns token ids [0, 4), rank 1 owns [4, 8). Each rank contributes
    # its local top-2 directly, so the returned candidate width is TP * 2.
    rank0_logits = torch.tensor(
        [
            [1.0, 8.0],
            [9.0, 2.0],
            [3.0, 4.0],
            [2.0, 5.0],
        ]
    )
    rank1_values = torch.tensor([[10.0, 7.0], [8.0, 6.0]])
    rank1_token_ids = torch.tensor([[4, 4], [5, 5]], dtype=torch.int32)
    rank1_payload = torch.empty((4, 2), dtype=torch.float32)
    rank1_payload[:2].copy_(rank1_values)
    rank1_payload[2:].view(torch.int32).copy_(rank1_token_ids)
    gather_call = 0

    def fake_all_gather_into_tensor(output_, input_, group=None, async_op=False):
        nonlocal gather_call
        assert input_.dtype == torch.float32
        assert input_.shape == (4, 2)
        output_[0].copy_(input_)
        output_[1].copy_(rank1_payload)
        gather_call += 1

    monkeypatch.setattr(llama_post_layer, "all_gather_into_tensor", fake_all_gather_into_tensor)
    infer_state = SimpleNamespace(dist_group=None, logits_token_ids=None)
    layer_weight = SimpleNamespace(
        lm_head_weight_=SimpleNamespace(
            vocab_size=8,
            tp_vocab_start_id=0,
        )
    )

    sparse_logits = post._gather_draft_topk_logits(
        logic_batch=rank0_logits,
        token_num=2,
        layer_weight=layer_weight,
        infer_state=infer_state,
    )

    assert sparse_logits.shape == (2, 4)
    assert infer_state.logits_token_ids.shape == (2, 4)
    assert sparse_logits.dtype == torch.float32
    assert infer_state.logits_token_ids.dtype == torch.int64
    reconstructed = torch.full((2, 8), float("-inf"))
    reconstructed.scatter_(1, infer_state.logits_token_ids.long(), sparse_logits)
    expected = torch.tensor(
        [
            [float("-inf"), 9.0, 3.0, float("-inf"), 10.0, 8.0, float("-inf"), float("-inf")],
            [8.0, float("-inf"), float("-inf"), 5.0, 7.0, 6.0, float("-inf"), float("-inf")],
        ]
    )
    torch.testing.assert_close(reconstructed, expected)
    assert gather_call == 1


def test_draft_argmax_and_truncated_probability_restore_vocabulary_ids():
    backend = ModeBackend.__new__(ModeBackend)
    output = ModelOutput(
        logits=torch.tensor([[3.0, 1.0, 2.0], [0.0, 5.0, 4.0]]),
        logits_token_ids=torch.tensor([[30, 10, 20], [100, 500, 400]]),
    )

    token_ids = backend._gen_argmax_token_ids(output)
    token_ids_with_prob, probs = backend._gen_argmax_token_ids_and_prob(output)

    torch.testing.assert_close(token_ids, torch.tensor([30, 500]))
    torch.testing.assert_close(token_ids_with_prob, token_ids)
    expected_probs = torch.softmax(output.logits, dim=-1).max(dim=-1).values
    torch.testing.assert_close(probs, expected_probs)


def test_dense_argmax_keeps_column_index_semantics():
    backend = ModeBackend.__new__(ModeBackend)
    output = ModelOutput(logits=torch.tensor([[1.0, 4.0, 2.0]]))

    torch.testing.assert_close(backend._gen_argmax_token_ids(output), torch.tensor([1]))


def test_dspark_confidence_path_receives_mapped_sparse_token_ids():
    post = Qwen3DSparkPostLayerInfer.__new__(Qwen3DSparkPostLayerInfer)
    post.block_size_ = 2
    post.markov_rank_ = 0
    post._slice_get_last_input = lambda input_embeddings, infer_state: (input_embeddings, 4)
    sparse_logits = torch.tensor([[1.0, 4.0], [5.0, 2.0], [3.0, 7.0], [9.0, 8.0]])
    sparse_token_ids = torch.tensor([[10, 40], [50, 20], [30, 70], [90, 80]], dtype=torch.int32)

    def gather_sparse(*args, **kwargs):
        infer_state = args[3]
        infer_state.logits_token_ids = sparse_token_ids
        return sparse_logits

    post._lm_head_and_gather = gather_sparse
    observed = {}

    def predict_confidence(block_hidden, anchor_token_ids, sampled_tokens, layer_weight):
        observed["sampled_tokens"] = sampled_tokens
        return None

    post.predict_confidence_logits = predict_confidence

    class Collector:
        def add_mtp_outputs(self, **kwargs):
            self.outputs = kwargs

    collector = Collector()
    infer_state = SimpleNamespace(
        is_prefill=False,
        is_mtp_draft_model=True,
        input_ids=torch.tensor([1, 0, 2, 0]),
        logits_token_ids=None,
        hidden_collector=collector,
    )

    returned_logits = post.token_forward(
        input_embdings=torch.ones((4, 3)),
        infer_state=infer_state,
        layer_weight=object(),
    )

    torch.testing.assert_close(returned_logits, sparse_logits)
    torch.testing.assert_close(observed["sampled_tokens"], torch.tensor([[40, 50], [70, 90]], dtype=torch.int32))
