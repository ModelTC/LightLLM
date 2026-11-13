"""FP32 DSA selection with the baseline's exact lowest-index tie rule.

The trained BF16 dot products and FP32 reductions are shared with the packed
reference. Only their storage and selection change. Never perturb scores to
break ties: an epsilon would change the order of unequal near-boundary scores.
"""

import glob
import os

import torch

from .dsa_topk import canonical_topk


_VLLM_TOPK_LOAD_STATE = None


def _vllm_topk_library():
    """Load an optional vLLM TopK operator once per worker.

    The operator's raw tie behavior is not exact, so callers always run the
    baseline canonical repair afterward. An explicit ``DSA_VLLM_TOPK_LIBRARY``
    wins; otherwise use an installed vLLM ``_C`` extension when present.
    """
    global _VLLM_TOPK_LOAD_STATE
    if _VLLM_TOPK_LOAD_STATE is False:
        return False
    if _VLLM_TOPK_LOAD_STATE is True:
        return True
    if hasattr(torch.ops, "_C") and hasattr(torch.ops._C, "top_k_per_row_decode"):
        _VLLM_TOPK_LOAD_STATE = True
        return True
    explicit = os.environ.get("DSA_VLLM_TOPK_LIBRARY", "")
    libraries = [explicit] if explicit else sorted(
        glob.glob("/opt/conda/lib/python*/site-packages/vllm/_C*.so")
    )
    for library in libraries:
        if not library or not os.path.isfile(library):
            continue
        try:
            torch.ops.load_library(library)
            if hasattr(torch.ops, "_C") and hasattr(torch.ops._C, "top_k_per_row_decode"):
                _VLLM_TOPK_LOAD_STATE = True
                return True
        except (OSError, RuntimeError):
            continue
    _VLLM_TOPK_LOAD_STATE = False
    return False


def _vllm_exact_topk(scores, lengths, count):
    if not scores.is_cuda or not _vllm_topk_library():
        raise RuntimeError("DSA vLLM TopK backend is unavailable")
    raw = torch.empty((scores.shape[0], count), dtype=torch.int32, device=scores.device)
    torch.ops._C.top_k_per_row_decode(
        scores,
        1,
        lengths,
        raw,
        scores.shape[0],
        scores.stride(0),
        scores.stride(1),
        count,
    )
    valid = (raw >= 0) & (raw < scores.shape[1]) & (raw < lengths[:, None])
    values = scores.gather(1, raw.clamp_min(0)).masked_fill(~valid, -torch.inf)
    raw = raw.masked_fill(~valid, scores.shape[1])
    return canonical_topk(scores, raw, values)[0].to(torch.int32)


def exact_float_topk(scores, lengths, topk, *, backend="auto"):
    """Return increasing int32 token IDs and -1 padding, matching packed TopK.

    scores contains finite FP32 scores and -inf for noncausal/padded positions.
    lengths is an int32 vector. The radix path resolves the full FP32 cutoff
    before compacting tokens. The Torch fallback repairs ties in its candidates.
    """
    if scores.ndim != 2 or scores.dtype != torch.float32:
        raise ValueError("DSA selection requires a two-dimensional FP32 score tensor")
    if lengths.shape != (scores.shape[0],) or lengths.dtype != torch.int32 or lengths.device != scores.device:
        raise ValueError("DSA lengths must be one int32 value per score row on the same device")
    scores, lengths = scores.contiguous(), lengths.contiguous()
    count = min(topk, scores.shape[1])
    if count <= 0:
        raise ValueError("DSA topk and score width must be positive")
    if backend == "auto":
        # Optional vLLM per-row TopK is substantially faster for large score
        # matrices. Canonical repair below restores exact FP32/tie semantics.
        if scores.shape[1] >= 32768 and scores.numel() >= 16 * 1024**2 and _vllm_topk_library():
            backend = "vllm"
        else:
            # The shared-H800 graph benchmark favors radix for short query
            # batches and Torch's FP32 selection for large matrices when the
            # optional vLLM operator is unavailable.
            backend = "torch" if scores.numel() >= 16 * 1024**2 else "radix"
    if backend == "vllm":
        return _vllm_exact_topk(scores, lengths, count)
    if backend == "radix" and scores.is_cuda:
        from .dsa_radix import exact_radix_topk

        block_size = 2048 if scores.shape[1] <= 131072 else 4096
        return exact_radix_topk(scores, lengths, count, block_size=block_size)
    elif backend in ("torch", "radix"):
        values, indices = scores.topk(count, dim=-1, sorted=False)
    else:
        raise ValueError(f"Unsupported exact DSA selector: {backend}")
    return canonical_topk(scores, indices, values)[0].to(torch.int32)
