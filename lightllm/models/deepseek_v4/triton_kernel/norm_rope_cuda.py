import functools
import hashlib
import os

import torch


@functools.lru_cache(maxsize=1)
def _load_cuda():
    from torch.utils.cpp_extension import load

    src = os.path.join(os.path.dirname(__file__), "csrc", "norm_rope.cu")
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
    module_name = f"lightllm_dsv4_norm_rope_v1_{hashlib.sha256(cache_key).hexdigest()[:16]}"
    return load(
        name=module_name,
        sources=[src],
        extra_cuda_cflags=flags,
        verbose=False,
    )


def _as_interleaved_freqs(freqs_cis: torch.Tensor) -> torch.Tensor:
    assert freqs_cis.dtype == torch.complex64
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    assert freqs_real.is_contiguous()
    return freqs_real


@torch.no_grad()
def fused_q_norm_rope(
    q_input: torch.Tensor,
    q_output: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    _load_cuda().fused_q_norm_rope(q_input, q_output, _as_interleaved_freqs(freqs_cis), positions, float(eps))
    return


@torch.no_grad()
def fused_k_norm_rope_flashmla(
    kv: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    out_loc: torch.Tensor,
    kvcache: torch.Tensor,
    page_size: int,
) -> None:
    _load_cuda().fused_k_norm_rope_flashmla(
        kv,
        kv_weight,
        _as_interleaved_freqs(freqs_cis),
        positions,
        out_loc,
        kvcache,
        float(eps),
        int(page_size),
    )
    return


@torch.no_grad()
def fused_q_indexer_rope_hadamard_quant(
    q_input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
):
    q_fp8 = torch.empty(q_input.shape, dtype=torch.float8_e4m3fn, device=q_input.device)
    weights_out = torch.empty((*q_input.shape[:-1], 1), dtype=torch.float32, device=q_input.device)
    _load_cuda().fused_q_indexer_rope_hadamard_quant(
        q_input,
        q_fp8,
        weight,
        weights_out,
        float(weight_scale),
        _as_interleaved_freqs(freqs_cis),
        positions,
    )
    return q_fp8, weights_out
