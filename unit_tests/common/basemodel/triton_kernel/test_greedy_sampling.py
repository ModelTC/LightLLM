import pytest
import torch

from lightllm.common.basemodel.triton_kernel.post_process.argmax import argmax
from lightllm.common.basemodel.triton_kernel.post_process.greedy import greedy_sample

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("vocab", [13, 4101, 248320])
def test_greedy_matches_torch_with_special_logits(vocab):
    torch.manual_seed(417)
    logits = torch.randn(7, vocab, device="cuda")
    logits[0].fill_(-float("inf"))
    logits[1, 2] = float("nan")
    logits[1, 9] = float("nan")
    logits[2, 4] = float("inf")
    logits[3, 4] = logits[3, 9] = 100
    temperatures = torch.tensor([1, 1, 1, 0.5, 0.7, 1.3, 2.0], device="cuda")
    expected = logits / temperatures[:, None]
    ids = expected.argmax(-1)
    probs = expected.softmax(-1).gather(1, ids[:, None]).log().flatten()
    actual_ids, actual_probs = greedy_sample(logits, temperatures)
    assert torch.equal(ids, actual_ids)
    assert torch.equal(argmax(expected), ids)
    torch.testing.assert_close(logits, expected, atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(actual_probs, probs, atol=3e-6, rtol=1e-6, equal_nan=True)


@pytest.mark.parametrize("vocab", [13, 4101])
def test_identity_penalty_skip_preserves_eos_and_masks(vocab):
    from types import SimpleNamespace
    from lightllm.common.basemodel.triton_kernel.post_process.apply_penalty_gpu_cache import apply_penalty_gpu_cache

    torch.manual_seed(41)
    manager = SimpleNamespace(
        vocab_size=vocab,
        req_to_presence_penalty=torch.zeros(7, device="cuda"),
        req_to_frequency_penalty=torch.zeros(7, device="cuda"),
        req_to_repetition_penalty=torch.ones(7, device="cuda"),
        req_to_out_token_id_counter=torch.randint(0, 20, (7, vocab), device="cuda", dtype=torch.int32),
        req_to_exponential_decay_length_penalty=torch.tensor([1.0, 1.01, 1.2, 1.0, 1.5, 0.9, 1.0], device="cuda"),
    )
    logits = torch.randn(5, vocab, device="cuda")
    original = logits.clone()
    params = dict(
        b_req_idx=torch.tensor([5, 3, 0, 4, 1], device="cuda", dtype=torch.int32),
        b_length_penalty_param=torch.tensor([8, 0, 3, 4, 2], device="cuda", dtype=torch.int32),
        b_mask_eos_reqs=torch.tensor([False, True, False, True, False], device="cuda"),
        eos_ids=torch.tensor([2, 9], device="cuda", dtype=torch.int32),
        sampling_params_manager=manager,
    )
    apply_penalty_gpu_cache(original, **params)
    apply_penalty_gpu_cache(logits, **params, skip_token_penalties=True)
    torch.testing.assert_close(logits, original, rtol=0, atol=0)
    assert (logits[[1, 3]][:, [2, 9]] == -10000000).all()


@pytest.mark.parametrize("greedy", [True, False])
@pytest.mark.parametrize("identity", [True, False])
def test_full_sampler_with_penalties_invalid_ids_and_random_sampling(monkeypatch, greedy, identity):
    from types import SimpleNamespace
    import lightllm.server.router.model_infer.mode_backend.generic_post_process as post

    vocab, batch = 4101, 4
    manager = SimpleNamespace(
        penalty_counter_mode="gpu_counter",
        req_to_bitmask_enabled=None,
        vocab_size=vocab,
        req_to_presence_penalty=torch.full((batch,), 0.0 if identity else 0.5, device="cuda"),
        req_to_frequency_penalty=torch.full((batch,), 0.0 if identity else 0.1, device="cuda"),
        req_to_repetition_penalty=torch.full((batch,), 1.0 if identity else 1.2, device="cuda"),
        req_to_out_token_id_counter=torch.randint(0, 10, (batch, vocab), device="cuda", dtype=torch.int32),
        req_to_exponential_decay_length_penalty=torch.full((batch,), 1.01, device="cuda"),
    )
    reqs = []
    for index in range(batch):
        param = SimpleNamespace(
            presence_penalty=0.0 if identity else 0.5,
            frequency_penalty=0.0 if identity else 0.1,
            repetition_penalty=1.0 if identity else 1.2,
            temperature=0.7 + 0.2 * index,
            top_k=1 if greedy else vocab,
            top_p=1.0,
            min_new_tokens=12 if index % 2 else 0,
            exponential_decay_length_penalty=SimpleNamespace(to_tuple=lambda: (0, 1.01)),
        )
        reqs.append(
            SimpleNamespace(
                sampling_param=SimpleNamespace(shm_param=param, invalid_token_ids=[0, 5, 13]),
                shm_req=SimpleNamespace(input_len=3),
                get_cur_total_len=lambda: 10,
                vocab_size=vocab,
                generator=None,
                req_idx=index,
            )
        )
    monkeypatch.setattr(
        post, "g_infer_context", SimpleNamespace(req_manager=SimpleNamespace(req_sampling_params_manager=manager))
    )
    monkeypatch.setattr(
        post,
        "g_pin_mem_manager",
        SimpleNamespace(gen_from_list=lambda key, data, dtype: torch.tensor(data, dtype=dtype)),
    )
    torch.manual_seed(62)
    logits = torch.randn(batch, vocab, device="cuda")
    torch.manual_seed(31)
    actual = post.sample(logits.clone(), reqs, eos_id=[2, 9])

    original_penalty = post.apply_penalty_gpu_cache

    def unskipped_penalty(**kwargs):
        kwargs["skip_token_penalties"] = False
        return original_penalty(**kwargs)

    def reference_greedy(logits, temperatures):
        logits.div_(temperatures[:, None])
        probs = logits.softmax(-1)
        ids = logits.argmax(-1)
        return ids, probs.gather(1, ids[:, None]).log().flatten()

    monkeypatch.setattr(post, "apply_penalty_gpu_cache", unskipped_penalty)
    monkeypatch.setattr(post, "greedy_sample", reference_greedy)
    torch.manual_seed(31)
    expected = post.sample(logits.clone(), reqs, eos_id=[2, 9])
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-6, atol=3e-6)


def test_greedy_fully_masked_partition_keeps_finite_logprob():
    logits = torch.randn(3, 8201, device="cuda")
    logits[0, :4096] = -float("inf")
    logits[1, 4096:8192] = -float("inf")
    logits[2, 8192:] = -float("inf")
    temperatures = torch.tensor([0.7, 1.0, 1.3], device="cuda")
    scaled = logits / temperatures[:, None]
    expected_ids = scaled.argmax(-1)
    expected_logp = scaled.softmax(-1).gather(1, expected_ids[:, None]).log().flatten()
    ids, logp = greedy_sample(logits, temperatures)
    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(logp, expected_logp, rtol=1e-6, atol=3e-6)
    assert torch.isfinite(logp).all()
