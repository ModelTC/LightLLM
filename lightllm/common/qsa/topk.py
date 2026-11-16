from functools import lru_cache
from inspect import signature

import torch
import triton
import triton.language as tl
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


_CUDA_TOPK_RUNTIME = str(torch.__version__) == "2.9.1+cu128" and triton.__version__ == "3.5.1"


@lru_cache(maxsize=16)
def _supports_device(device):
    supported = torch.cuda.get_device_capability(device) == (9, 0) and "H100" in torch.cuda.get_device_name(device)
    if supported:
        logger.info("QSA CUDA TopK selection enabled")
    return supported


@triton.jit
def _key(value):
    bits = value.to(tl.uint32, bitcast=True)
    mask = tl.where((bits & 0x80000000) != 0, 0xFFFFFFFF, 0x80000000).to(tl.uint32)
    return tl.where(value == value, bits ^ mask, 0xFFFFFFFF).to(tl.uint32)


@triton.jit
def _topk(
    Scores,
    Selected,
    Lengths,
    FIRST: tl.constexpr,
    RATIO: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    B: tl.constexpr,
):
    tl.static_assert(N == 1027 and K == 512 and B == 2048)
    row = tl.program_id(0).to(tl.int64)
    if Lengths is None:
        short = tl.full((), False, tl.int1)
    else:
        short = tl.load(Lengths + row + FIRST) // RATIO < K
    if short:
        short_col = tl.arange(0, K)
        tl.store(Selected + row * K + short_col, short_col.to(tl.int64))
    else:
        prefix_col = tl.arange(0, 1024)
        prefix = tl.topk(_key(tl.load(Scores + row * N + prefix_col)), K)
        merge_col = tl.arange(0, 4)
        tail = tl.load(Scores + row * N + 1024 + merge_col, merge_col < 3, other=0.0)
        tail = tl.sort(tl.where(merge_col < 3, _key(tail), 0).to(tl.uint32), descending=True)

        prefix_cutoffs = tl.gather(prefix, K - 1 - merge_col, 0)
        tail_cutoffs = tl.gather(tail, tl.maximum(merge_col - 1, 0), 0)
        tail_cutoffs = tl.where(merge_col == 0, 0xFFFFFFFF, tail_cutoffs).to(tl.uint32)
        threshold = tl.max(tl.minimum(prefix_cutoffs, tail_cutoffs))
        col = tl.arange(0, B)
        value = tl.load(Scores + row * N + col, col < N, other=-float("inf"))
        key = _key(value)
        greater = (col < N) & (key > threshold)
        equal = (col < N) & (key == threshold)
        greater_rank = tl.cumsum(greater.to(tl.int32)) - 1
        equal_rank = tl.cumsum(equal.to(tl.int32)) - 1
        rank = tl.where(greater, greater_rank, tl.sum(greater.to(tl.int32)) + equal_rank)
        selected = greater | (equal & (rank < K))
        tl.store(Selected + row * K + rank, col.to(tl.int64), selected)


def cuda_topk_supported(device, blocks: int, k: int) -> bool:
    return _CUDA_TOPK_RUNTIME and device.type == "cuda" and blocks == 1027 and k == 512 and _supports_device(device)


@lru_cache(maxsize=1)
def _flashinfer_topk():
    try:
        from flashinfer import top_k_ragged_transform
    except ImportError:
        return None
    if "dsa_graph_safe" not in signature(top_k_ragged_transform).parameters:
        return None
    return top_k_ragged_transform


@lru_cache(maxsize=16)
def flashinfer_topk_supported(device, blocks: int, k: int) -> bool:
    return (
        device.type == "cuda"
        and k == 512
        and blocks >= k
        and torch.cuda.get_device_capability(device) == (9, 0)
        and _flashinfer_topk() is not None
    )


def flashinfer_topk_indices(scores, k, *, visible_lengths, first=0, compress_ratio=4):
    lengths = visible_lengths[first : first + scores.shape[0]] // compress_ratio
    return _flashinfer_topk()(
        scores, torch.zeros_like(lengths), lengths, k, deterministic=True, tie_break=1, dsa_graph_safe=True
    )


def cuda_topk_indices(scores: torch.Tensor, k: int, *, visible_lengths=None, first=0, compress_ratio=4) -> torch.Tensor:
    if not (
        scores.is_cuda
        and scores.dtype == torch.float32
        and scores.ndim == 2
        and scores.is_contiguous()
        and cuda_topk_supported(scores.device, scores.shape[1], k)
    ):
        return torch.topk(scores, k, dim=-1, sorted=False).indices
    if visible_lengths is not None:
        if (
            visible_lengths.dtype != torch.int32
            or visible_lengths.device != scores.device
            or visible_lengths.ndim != 1
            or not visible_lengths.is_contiguous()
            or first < 0
            or first + scores.shape[0] > visible_lengths.numel()
            or compress_ratio < 1
        ):
            raise ValueError("QSA TopK requires matching contiguous int32 visible lengths and valid bounds")
    output = torch.empty((scores.shape[0], k), dtype=torch.int64, device=scores.device)
    if scores.shape[0]:
        _topk[(scores.shape[0],)](scores, output, visible_lengths, first, compress_ratio, 1027, 512, 2048, num_warps=16)
    return output
