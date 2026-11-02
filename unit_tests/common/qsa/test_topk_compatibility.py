from types import SimpleNamespace

import pytest
import torch


from lightllm.common.qsa import topk
from lightllm.common.qsa.indexer import QSAIndexer


@pytest.fixture(autouse=True)
def no_cuda_initialization():
    initially_initialized = torch.cuda.is_initialized()
    yield
    assert torch.cuda.is_initialized() == initially_initialized


@pytest.mark.parametrize("shape,k", [((3, 1027), 512), ((0, 1027), 512), ((2, 11), 0), ((2, 11), 11), ((2, 3, 11), 4)])
def test_cpu_fallback_keeps_actual_torch_indices(shape, k, monkeypatch):
    def unexpected_query(device):
        raise AssertionError("CPU fallback queried CUDA")

    monkeypatch.setattr(topk, "_supports_device", unexpected_query)
    scores = torch.arange(torch.tensor(shape).prod().item()).reshape(shape).float()
    assert torch.equal(topk.cuda_topk_indices(scores, k), torch.topk(scores, k, dim=-1, sorted=False).indices)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64])
def test_noncontiguous_cpu_fallback(dtype):
    scores = torch.arange(22).reshape(11, 2).T.to(dtype)
    assert not scores.is_contiguous()
    assert torch.equal(topk.cuda_topk_indices(scores, 4), torch.topk(scores, 4, sorted=False).indices)


@pytest.mark.parametrize("k", [-1, 12])
def test_invalid_budget_keeps_torch_exception(k):
    with pytest.raises(RuntimeError):
        topk.cuda_topk_indices(torch.ones((2, 11)), k)


@pytest.mark.parametrize(
    "change",
    [
        dict(runtime=False),
        dict(dtype=torch.bfloat16),
        dict(shape=(3, 1024)),
        dict(shape=(3, 8, 1027)),
        dict(k=256),
        dict(contiguous=False),
        dict(device_supported=False),
    ],
)
def test_unqualified_cuda_metadata_delegates_without_launch(change, monkeypatch):
    config = dict(runtime=True, dtype=torch.float32, shape=(3, 1027), k=512, contiguous=True, device_supported=True)
    config.update(change)
    scores = SimpleNamespace(
        is_cuda=True,
        dtype=config["dtype"],
        ndim=len(config["shape"]),
        shape=config["shape"],
        device=torch.device("cuda", 0),
        is_contiguous=lambda: config["contiguous"],
    )
    sentinel = object()
    calls = []
    monkeypatch.setattr(topk, "_CUDA_TOPK_RUNTIME", config["runtime"])
    monkeypatch.setattr(topk, "_supports_device", lambda _: config["device_supported"])

    def original(value, k, *, dim, sorted):
        calls.append((value, k, dim, sorted))
        return SimpleNamespace(indices=sentinel)

    monkeypatch.setattr(topk.torch, "topk", original)
    assert topk.cuda_topk_indices(scores, config["k"]) is sentinel
    assert calls == [(scores, config["k"], -1, False)]


def test_shared_default_and_select_api_remain_compatible():
    import inspect

    assert QSAIndexer.__dataclass_fields__["use_cuda_topk"].default is False
    signature = inspect.signature(QSAIndexer.select)
    assert list(signature.parameters) == [
        "self",
        "query",
        "cache",
        "requests",
        "visible_lengths",
        "indices",
        "query_chunk_size",
    ]
    assert signature.parameters["query_chunk_size"].default == 64


@pytest.mark.parametrize(
    "change",
    [
        dict(dtype=torch.int64),
        dict(device=torch.device("cpu")),
        dict(ndim=2),
        dict(contiguous=False),
        dict(first=-1),
        dict(first=2),
        dict(ratio=0),
    ],
)
def test_invalid_visible_metadata_fails_before_allocation(change, monkeypatch):
    device = torch.device("cuda", 0)
    config = dict(dtype=torch.int32, device=device, ndim=1, contiguous=True, first=0, ratio=4)
    config.update(change)
    scores = SimpleNamespace(
        is_cuda=True, dtype=torch.float32, ndim=2, shape=(3, 1027), device=device, is_contiguous=lambda: True
    )
    visible = SimpleNamespace(
        dtype=config["dtype"],
        device=config["device"],
        ndim=config["ndim"],
        is_contiguous=lambda: config["contiguous"],
        numel=lambda: 3,
    )
    monkeypatch.setattr(topk, "_CUDA_TOPK_RUNTIME", True)
    monkeypatch.setattr(topk, "_supports_device", lambda _: True)

    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("Invalid visible metadata reached GPU allocation")

    monkeypatch.setattr(topk.torch, "empty", unexpected_allocation)
    with pytest.raises(ValueError, match="visible lengths"):
        topk.cuda_topk_indices(
            scores, 512, visible_lengths=visible, first=config["first"], compress_ratio=config["ratio"]
        )
