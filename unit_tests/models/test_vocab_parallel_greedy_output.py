from types import SimpleNamespace

import torch

from lightllm.common.basemodel.batch_objs import ModelOutput
from lightllm.models.qwen3_dspark.layer_infer.post_layer_infer import Qwen3DSparkPostLayerInfer
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend


def test_argmax_restores_global_token_ids_and_exact_probabilities():
    backend = ModeBackend.__new__(ModeBackend)
    output = ModelOutput(
        logits=torch.tensor([[3.0, 1.0], [0.0, 5.0]]),
        logits_token_ids=torch.tensor([[30, 10], [100, 500]]),
        logits_logsumexp=torch.tensor([4.0, 5.25]),
    )

    token_ids = backend._gen_argmax_token_ids(output)
    token_ids_with_prob, probs = backend._gen_argmax_token_ids_and_prob(output)

    torch.testing.assert_close(token_ids, torch.tensor([30, 500]))
    torch.testing.assert_close(token_ids_with_prob, token_ids)
    torch.testing.assert_close(probs, torch.exp(torch.tensor([-1.0, -0.25])))


def test_dense_argmax_keeps_column_index_semantics():
    backend = ModeBackend.__new__(ModeBackend)
    output = ModelOutput(logits=torch.tensor([[1.0, 4.0, 2.0]]))

    torch.testing.assert_close(backend._gen_argmax_token_ids(output), torch.tensor([1]))


def test_dspark_confidence_path_receives_global_token_ids():
    post = Qwen3DSparkPostLayerInfer.__new__(Qwen3DSparkPostLayerInfer)
    post.block_size_ = 2
    post.markov_rank_ = 0
    post._slice_get_last_input = lambda input_embeddings, infer_state: (input_embeddings, 4)
    sparse_logits = torch.tensor([[4.0], [5.0], [7.0], [9.0]])
    sparse_token_ids = torch.tensor([[40], [50], [70], [90]])

    def gather_vocab_parallel(*args, **kwargs):
        infer_state = args[3]
        infer_state.logits_token_ids = sparse_token_ids
        return sparse_logits

    post._lm_head_and_gather = gather_vocab_parallel
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
    torch.testing.assert_close(observed["sampled_tokens"], torch.tensor([[40, 50], [70, 90]]))
