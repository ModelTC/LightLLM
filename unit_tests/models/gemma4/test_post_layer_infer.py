from types import SimpleNamespace

import torch

import lightllm.models.llama.layer_infer.post_layer_infer as llama_post_layer
from lightllm.models.gemma4.layer_infer.post_layer_infer import Gemma4PostLayerInfer


def test_vocab_parallel_greedy_softcaps_local_logits_before_reduction(monkeypatch):
    post_layer = Gemma4PostLayerInfer.__new__(Gemma4PostLayerInfer)
    post_layer.final_logit_softcapping = 2.0
    post_layer.tp_world_size_ = 2
    post_layer.alloc_tensor = torch.empty
    post_layer._norm = lambda hidden, infer_state, layer_weight: hidden

    local_logits = torch.tensor(
        [
            [4.0, -4.0],
            [2.0, -2.0],
            [1.0, -1.0],
        ],
        dtype=torch.bfloat16,
    )

    class LMHead:
        vocab_size = 6

        def __call__(self, input, alloc_func):
            return local_logits

    sparse_logits = torch.tensor([[5.0], [1.0]])
    token_ids = torch.tensor([[5], [1]])
    logsumexp = torch.tensor([5.25, 1.5])
    captured = {}

    def fake_vocab_parallel_greedy(logits, **kwargs):
        captured["logits"] = logits
        return sparse_logits, token_ids, logsumexp

    monkeypatch.setattr(llama_post_layer, "vocab_parallel_greedy", fake_vocab_parallel_greedy)

    infer_state = SimpleNamespace(
        dist_group=None,
        use_vocab_parallel_greedy=True,
        logits_token_ids=None,
        logits_logsumexp=None,
    )

    result = post_layer._lm_head_and_gather(
        hidden=torch.empty((2, 3)),
        token_num=2,
        layer_weight=SimpleNamespace(lm_head_weight_=LMHead()),
        infer_state=infer_state,
    )

    expected_logits = (
        torch.tanh(local_logits.float() / post_layer.final_logit_softcapping) * post_layer.final_logit_softcapping
    )
    assert captured["logits"].dtype == torch.float32
    torch.testing.assert_close(captured["logits"], expected_logits)
    assert result is sparse_logits
    assert infer_state.logits_token_ids is token_ids
    assert infer_state.logits_logsumexp is logsumexp


def test_full_logits_softcap_after_float32_conversion(monkeypatch):
    post_layer = Gemma4PostLayerInfer.__new__(Gemma4PostLayerInfer)
    post_layer.final_logit_softcapping = 2.0
    post_layer.tp_world_size_ = 1
    post_layer.alloc_tensor = torch.empty
    post_layer._norm = lambda hidden, infer_state, layer_weight: hidden

    local_logits = torch.tensor(
        [
            [1.234375, -3.140625],
            [2.71875, -0.333984375],
            [0.10009765625, -1.609375],
        ],
        dtype=torch.bfloat16,
    )

    class LMHead:
        vocab_size = 3

        def __call__(self, input, alloc_func):
            return local_logits

    result = post_layer._lm_head_and_gather(
        hidden=torch.empty((2, 3), dtype=torch.bfloat16),
        token_num=2,
        layer_weight=SimpleNamespace(lm_head_weight_=LMHead()),
        infer_state=SimpleNamespace(dist_group=None, use_vocab_parallel_greedy=True),
        force_full_logits=True,
    )

    expected = torch.tanh(local_logits.T.float() / post_layer.final_logit_softcapping)
    expected *= post_layer.final_logit_softcapping
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, expected)
