import functools
import hashlib
import os

import torch


@functools.lru_cache(maxsize=1)
def _load_cuda():
    from torch.utils.cpp_extension import load

    src = os.path.join(os.path.dirname(__file__), "csrc", "topk_transform.cu")
    flags = ["-O3"]
    with open(src, "rb") as source_file:
        source = source_file.read()
    capability = torch.cuda.get_device_capability()
    cache_key = b"\0".join(
        [
            source,
            " ".join(flags).encode(),
            torch.__version__.encode(),
            str(torch.version.cuda).encode(),
            f"sm{capability[0]}{capability[1]}".encode(),
            os.environ.get("TORCH_CUDA_ARCH_LIST", "").encode(),
        ]
    )
    module_name = f"lightllm_dsv4_topk_v1_{hashlib.sha256(cache_key).hexdigest()[:16]}"
    return load(
        name=module_name,
        sources=[src],
        extra_cuda_cflags=flags,
        verbose=False,
    )


@torch.no_grad()
def topk_transform_512(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: torch.Tensor = None,
) -> None:
    """Masked top-512 selection over per-token scores + page-translate, for the DeepSeek-V4 c4
    indexer. Drop-in replacement for the former vendored topk_transform_512 op.

    LightLLM-local CUDA radix port (from SGLang topk_v1.cuh, TVM-FFI stripped and PDL preserved): it
    early-exits per token at seq_len, so it matches the original perf (unlike torch.topk which
    scans the full captured c4_cap width). Output is an unordered SET of physical c4 slots (-1 pad).

    Args:
        scores:           [T, c4_cap] fp32 (deep_gemm fp8_paged_mqa_logits output, -inf beyond ctx)
        seq_lens:         [T] int32 (valid causal entries per token)
        page_tables:      [T, npages] int32 (logical->physical c4 page map)
        out_page_indices: [T, 512] int32 (output physical slots, -1 pad)
        page_size:        c4 pool page size (64)
        out_raw_indices:  optional [T, 512] int32 (raw logical indices, -1 pad)
    """
    _load_cuda().topk_transform_512(scores, seq_lens, page_tables, out_page_indices, int(page_size), out_raw_indices)
    return
