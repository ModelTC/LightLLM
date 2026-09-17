import pytest
import torch

from lightllm.common.basemodel.triton_kernel.post_process.greedy_sample import _fused_greedy_sample as greedy_sample


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")


def torch_greedy(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits, dim=-1)
    ids = torch.argmax(logits, dim=-1)
    return ids, probs.gather(1, ids[:, None]).flatten().log()


@pytest.mark.parametrize("batch", [1, 3, 128])
@pytest.mark.parametrize("vocab", [1, 31, 4095, 4096, 4097, 151936, 262144])
def test_greedy_matches_torch(batch: int, vocab: int) -> None:
    torch.manual_seed(29)
    logits = torch.randn((batch, vocab), device="cuda")
    actual = greedy_sample(logits)
    expected = torch_greedy(logits)
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[1], expected[1], atol=2e-6, rtol=1e-6)


@pytest.mark.parametrize("case", ["tied", "negative_offset", "positive_offset", "masked", "all_masked", "nan", "inf"])
@pytest.mark.parametrize("strided", [False, True])
def test_greedy_edge_cases(case: str, strided: bool) -> None:
    logits = torch.randn((3, 20002), device="cuda")[:, ::2] if strided else torch.randn((3, 10001), device="cuda")
    if case == "tied":
        logits.fill_(-1e7)
        logits[:, [17, 4096, 8193]] = 10.0
    elif case in ("negative_offset", "positive_offset"):
        logits.add_(-1e7 if case == "negative_offset" else 1e7)
    elif case == "masked":
        logits.fill_(float("-inf"))
        logits[:, 9000] = -1e7
    elif case == "all_masked":
        logits.fill_(float("-inf"))
    elif case == "nan":
        logits[:, 0] = float("inf")
        logits[:, [17, 4096, 8193]] = float("nan")
    else:
        logits[:, [17, 4096, 8193]] = float("inf")
    before = logits.clone()
    actual = greedy_sample(logits)
    expected = torch_greedy(logits)
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[1], expected[1], atol=2e-6, rtol=1e-6, equal_nan=True)
    torch.testing.assert_close(logits, before, atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("apply_temperature", [False, True])
def test_greedy_cuda_graph_replay(apply_temperature: bool) -> None:
    logits = torch.randn((4, 151936), device="cuda")
    temperatures = torch.tensor([0.7, 1.0, 1.1, 2.0], device="cuda") if apply_temperature else None
    for _ in range(3):
        greedy_sample(logits, temperatures)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = greedy_sample(logits, temperatures)
    logits.neg_()
    before = logits.clone()
    graph.replay()
    if apply_temperature:
        before.div_(temperatures[:, None])
    torch.testing.assert_close(logits, before, atol=0, rtol=0)
    expected = torch_greedy(logits)
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[1], expected[1], atol=2e-6, rtol=1e-6)


def test_empty_batch() -> None:
    ids, logprobs = greedy_sample(torch.empty((0, 32), device="cuda"))
    assert ids.shape == logprobs.shape == (0,)


@pytest.mark.parametrize("seed", [29, 2026])
@pytest.mark.parametrize("vocab", [151936, 262144])
def test_concentrated_distribution_matches_fp64(seed: int, vocab: int) -> None:
    torch.manual_seed(seed)
    logits = torch.randn((128, vocab), device="cuda")
    logits[:, 2] = 22.5
    temperatures = torch.tensor([0.7 + i * 0.1 for i in range(128)], device="cuda")
    expected_logits = logits.clone().div_(temperatures[:, None])
    ids, logprobs = greedy_sample(logits, temperatures)
    expected_ids = expected_logits.argmax(-1)
    precise = expected_logits.double().log_softmax(-1).gather(1, expected_ids[:, None]).flatten()
    torch.testing.assert_close(ids, expected_ids, atol=0, rtol=0)
    torch.testing.assert_close(logits, expected_logits, atol=0, rtol=0)
    torch.testing.assert_close(logprobs.double(), precise, atol=1e-6, rtol=1e-7)
