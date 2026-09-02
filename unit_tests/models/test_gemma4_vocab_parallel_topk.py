from types import SimpleNamespace

import torch

import lightllm.models.llama.layer_infer.post_layer_infer as llama_post_layer
from lightllm.models.gemma4.layer_infer.post_layer_infer import Gemma4PostLayerInfer


def test_vocab_parallel_topk_softcaps_local_logits_before_candidate_selection(monkeypatch):
    post = Gemma4PostLayerInfer.__new__(Gemma4PostLayerInfer)
    post.final_logit_softcapping = 2.0
    post.vocab_parallel_topk_ = 2
    post.tp_world_size_ = 1
    post.alloc_tensor = torch.empty
    post._norm = lambda hidden, infer_state, layer_weight: hidden

    local_logits = torch.tensor(
        [[4.0, -4.0], [2.0, -2.0], [1.0, -1.0]],
        dtype=torch.bfloat16,
    )

    class LMHead:
        vocab_size = 3
        tp_vocab_start_id = 0

        def __call__(self, input, alloc_func):
            return local_logits

    sparse_logits = torch.tensor([[1.5, 1.0], [-1.0, -1.5]])
    token_ids = torch.tensor([[0, 1], [2, 1]])
    captured = {}

    def fake_vocab_parallel_topk(logits, **kwargs):
        captured["logits"] = logits
        return sparse_logits, token_ids

    monkeypatch.setattr(llama_post_layer, "vocab_parallel_topk", fake_vocab_parallel_topk)
    infer_state = SimpleNamespace(
        dist_group=None,
        use_vocab_parallel_topk=True,
        logits_token_ids=None,
    )

    result = post._lm_head_and_gather(
        hidden=torch.empty((2, 3)),
        token_num=2,
        layer_weight=SimpleNamespace(lm_head_weight_=LMHead()),
        infer_state=infer_state,
    )

    expected = torch.tanh(local_logits.float() / 2.0) * 2.0
    assert captured["logits"].dtype == torch.float32
    torch.testing.assert_close(captured["logits"], expected)
    assert result is sparse_logits
    assert infer_state.logits_token_ids is token_ids


def test_full_logits_softcap_after_float32_conversion():
    post = Gemma4PostLayerInfer.__new__(Gemma4PostLayerInfer)
    post.final_logit_softcapping = 2.0
    logits = torch.tensor([[1.234375, -3.140625]], dtype=torch.bfloat16)

    actual = post._apply_logit_postprocessing(logits)
    expected = torch.tanh(logits.float() / 2.0) * 2.0

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
