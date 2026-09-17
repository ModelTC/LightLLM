from types import SimpleNamespace

import pytest
import torch

from lightllm.server.router.model_infer.mode_backend import generic_post_process as sampling


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")


def make_context(batch: int, vocab: int, counter_mode: str) -> tuple[SimpleNamespace, list]:
    ids = torch.tensor([2, 5] * batch, device="cuda", dtype=torch.int64)
    counts = torch.ones_like(ids, dtype=torch.int32)
    offsets = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * 2
    history = torch.zeros((batch, vocab), device="cuda", dtype=torch.int32)
    history[:, [2, 5]] = 1
    manager = SimpleNamespace(
        vocab_size=vocab,
        penalty_counter_mode=counter_mode,
        req_to_presence_penalty=torch.zeros(batch, device="cuda"),
        req_to_frequency_penalty=torch.zeros(batch, device="cuda"),
        req_to_repetition_penalty=torch.ones(batch, device="cuda"),
        req_to_exponential_decay_length_penalty=torch.ones(batch, device="cuda"),
        req_to_out_token_id_counter=history,
        gen_cpu_out_token_counter_sampling_params=lambda req_objs: (ids, counts, offsets),
    )
    reqs = []
    for i in range(batch):
        params = SimpleNamespace(
            temperature=0.7 + i * 0.1,
            top_k=1,
            top_p=1.0,
            min_new_tokens=0,
            exponential_decay_length_penalty=SimpleNamespace(to_tuple=lambda: (0, 1.0)),
        )
        reqs.append(
            SimpleNamespace(
                req_idx=i,
                vocab_size=vocab,
                generator=None,
                shm_req=SimpleNamespace(input_len=1),
                get_cur_total_len=lambda: 3,
                sampling_param=SimpleNamespace(shm_param=params, invalid_token_ids=[]),
            )
        )
    return manager, reqs


def torch_greedy(logits: torch.Tensor, temperatures: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits.div_(temperatures[:, None])
    probs = torch.softmax(logits, dim=-1)
    ids = logits.argmax(dim=-1)
    return ids, probs.gather(1, ids[:, None]).flatten().log()


@pytest.mark.parametrize("counter_mode", ["cpu_counter", "gpu_counter"])
@pytest.mark.parametrize("case", ["normal", "penalties", "eos", "invalid", "constraint", "length", "candidates"])
@pytest.mark.parametrize("large", [False, True])
def test_generic_greedy_preserves_preprocessing(
    monkeypatch: pytest.MonkeyPatch, counter_mode: str, case: str, large: bool
) -> None:
    torch.manual_seed(29)
    batch, vocab = (128, 131072) if large else (3, 10001)
    manager, reqs = make_context(batch, vocab, counter_mode)
    monkeypatch.setattr(
        sampling, "g_infer_context", SimpleNamespace(req_manager=SimpleNamespace(req_sampling_params_manager=manager))
    )
    logits = torch.randn((batch, vocab), device="cuda")
    logits[:, 2] = 10.0
    logits[:, 5] = 9.0
    if case == "penalties":
        manager.req_to_repetition_penalty.fill_(2.0)
        manager.req_to_presence_penalty.fill_(0.6)
        manager.req_to_frequency_penalty.fill_(0.4)
    elif case == "eos":
        for req in reqs:
            req.sampling_param.shm_param.min_new_tokens = 10
    elif case == "invalid":
        for req in reqs:
            req.sampling_param.invalid_token_ids = [2, 5]
    elif case == "constraint":
        logits[:, :4096] = -1e6
    elif case == "length":
        manager.req_to_exponential_decay_length_penalty.fill_(1.5)
    elif case == "candidates":
        logits.fill_(-1e7)
        logits[:, 5] = 9.0
    actual_logits = logits.clone()
    actual = sampling.sample(actual_logits, reqs, eos_id=[2])
    monkeypatch.setattr(sampling, "greedy_sample", torch_greedy)
    expected_logits = logits.clone()
    expected = sampling.sample(expected_logits, reqs, eos_id=[2])
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    # 旧 FP32 softmax 在概率接近 1 时有数个 e-6 的归约误差，同时以 FP64 校验新结果。
    torch.testing.assert_close(actual[1], expected[1], atol=5e-6, rtol=1e-6)
    precise = expected_logits.double().log_softmax(-1).gather(1, expected[0][:, None]).flatten()
    torch.testing.assert_close(actual[1].double(), precise, atol=1e-6, rtol=1e-7)
    # 调用方继续使用 logits 计算 token rank，原地惩罚和温度处理必须一致。
    torch.testing.assert_close(actual_logits, expected_logits, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["random", "filtered", "mixed", "fp16"])
def test_other_sampling_paths_keep_existing_behavior(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    manager, reqs = make_context(3, 128, "gpu_counter")
    monkeypatch.setattr(
        sampling, "g_infer_context", SimpleNamespace(req_manager=SimpleNamespace(req_sampling_params_manager=manager))
    )
    monkeypatch.setattr(sampling, "get_env_start_args", lambda: SimpleNamespace(sampling_backend="triton"))

    def unexpected_call(logits: torch.Tensor) -> None:
        pytest.fail("非 FP32 全 greedy 批次不应调用新内核")

    monkeypatch.setattr(sampling, "greedy_sample", unexpected_call)
    if mode != "fp16":
        for i, req in enumerate(reqs):
            req.sampling_param.shm_param.top_k = (
                1 if mode == "mixed" and i == 0 else (16 if mode == "filtered" else 128)
            )
            req.sampling_param.shm_param.top_p = 0.9 if mode == "filtered" else 1.0
    logits = torch.randn((3, 128), device="cuda", dtype=torch.float16 if mode == "fp16" else torch.float32)
    ids, logprobs = sampling.sample(logits, reqs, eos_id=[2])
    assert ids.shape == logprobs.shape == (3,)
    assert torch.isfinite(logprobs).all()
