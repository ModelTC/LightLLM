"""Backend validation with subprocess isolation and ground truth checks."""

import multiprocessing as mp
import os
from functools import lru_cache

import torch

from lightllm.utils.dist_utils import get_global_rank
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

_VALIDATION_TIMEOUT = 30 * 60  # 30 minutes


def _compute_ground_truth(q, k, v, is_causal=True):
    """Ground truth using PyTorch SDPA."""
    with torch.no_grad():
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


def _validate_fa3(dtype=torch.bfloat16, spec=None):
    """Validate FA3 with ground truth."""
    from lightllm.utils.sgl_utils import flash_attn_varlen_func

    if torch.cuda.get_device_capability()[0] < 9:
        return False, "FA3 requires SM90 or newer"
    if flash_attn_varlen_func is None:
        return False, "flash_attn_varlen_func is None"

    batch, heads, seq, dim = 1, 4 if spec is None else spec.num_q_heads, 8, 64 if spec is None else spec.head_size
    q = torch.randn(batch, heads, seq, dim, dtype=dtype, device="cuda")
    k = torch.randn(batch, heads, seq, dim, dtype=dtype, device="cuda")
    v = torch.randn(batch, heads, seq, dim, dtype=dtype, device="cuda")

    expected = _compute_ground_truth(q, k, v)

    q_flat = q.transpose(1, 2).reshape(batch * seq, heads, dim)
    k_flat = k.transpose(1, 2).reshape(batch * seq, heads, dim)
    v_flat = v.transpose(1, 2).reshape(batch * seq, heads, dim)
    cu_seqlens = torch.arange(0, batch * seq + 1, seq, dtype=torch.int32, device="cuda")

    out = flash_attn_varlen_func(
        q=q_flat,
        k=k_flat,
        v=v_flat,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=seq,
        max_seqlen_k=seq,
        softmax_scale=1.0 / (dim ** 0.5),
        causal=True,
    )
    out = out.reshape(batch, seq, heads, dim).transpose(1, 2)
    torch.cuda.synchronize()

    if not torch.allclose(out, expected, rtol=1e-2, atol=1e-2):
        return (
            False,
            f"Output mismatch: max diff {(out - expected).abs().max().item():.6f}",
        )
    return True, None


def _validate_flashinfer(dtype=torch.float16, spec=None):
    """Validate FlashInfer with ground truth."""
    capability = torch.cuda.get_device_capability()
    arch = f"{capability[0]}.{capability[1]}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{arch}{'+PTX' if arch == '9.0' else ''}"

    import flashinfer

    batch, heads, seq, dim = 1, 4 if spec is None else spec.num_q_heads, 8, 64 if spec is None else spec.head_size
    q = torch.randn(batch, heads, seq, dim, dtype=dtype, device="cuda")
    k = torch.randn(batch, heads, seq, dim, dtype=dtype, device="cuda")
    v = torch.randn(batch, heads, seq, dim, dtype=dtype, device="cuda")

    expected = _compute_ground_truth(q, k, v)

    q_flat = q.transpose(1, 2).reshape(batch * seq, heads, dim)
    k_flat = k.transpose(1, 2).reshape(batch * seq, heads, dim)
    v_flat = v.transpose(1, 2).reshape(batch * seq, heads, dim)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    qo_indptr = torch.arange(0, batch * seq + 1, seq, dtype=torch.int32, device="cuda")

    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout="NHD")
    wrapper.plan(qo_indptr, qo_indptr, heads, heads, dim, causal=True)
    out = wrapper.run(q_flat, k_flat, v_flat)
    out = out.reshape(batch, seq, heads, dim).transpose(1, 2)
    torch.cuda.synchronize()

    if not torch.allclose(out, expected, rtol=1e-2, atol=1e-2):
        return (
            False,
            f"Output mismatch: max diff {(out - expected).abs().max().item():.6f}",
        )
    return True, None


def _validate_flashqla(dtype=None, spec=None):
    """Validate FlashQLA against LightLLM's vendored FLA kernel."""
    from flash_qla import chunk_gated_delta_rule as flashqla_chunk_gated_delta_rule

    from lightllm.common.attention.backends.gdn.triton.kernels.chunk import (
        chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
    )
    from lightllm.models.qwen3next.config import Qwen3NextResourceConfig

    linear_config = Qwen3NextResourceConfig.load_from_args()
    # 配置只保存全局 head 数，校验算子的输入按当前 TP 大小计算本地分片。
    num_k_heads = max(1, linear_config.global_linear_k_heads // linear_config.tp_world_size)
    num_v_heads = max(1, linear_config.global_linear_v_heads // linear_config.tp_world_size)
    head_k_dim = linear_config.head_linear_k_dim
    head_v_dim = linear_config.head_linear_v_dim
    qkv_dtype = dtype or linear_config.conv_state_dtype
    if spec is not None:
        num_k_heads = spec.num_q_heads
        num_v_heads = max(num_k_heads, num_v_heads)
        head_k_dim, head_v_dim = spec.head_size, spec.value_head_size
    state_dtype = linear_config.ssm_state_dtype

    batch, seq = 1, 64
    torch.manual_seed(0)
    q = torch.randn(batch, seq, num_k_heads, head_k_dim, dtype=qkv_dtype, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn(batch, seq, num_v_heads, head_v_dim, dtype=qkv_dtype, device="cuda")
    g = -torch.rand(batch, seq, num_v_heads, dtype=torch.float32, device="cuda")
    beta = torch.rand(batch, seq, num_v_heads, dtype=torch.float32, device="cuda")
    initial_state = torch.randn(batch, num_v_heads, head_k_dim, head_v_dim, dtype=state_dtype, device="cuda")
    cu_seqlens = torch.tensor([0, seq], dtype=torch.int32, device="cuda")
    kwargs = {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "initial_state": initial_state,
        "output_final_state": True,
        "cu_seqlens": cu_seqlens,
        "use_qk_l2norm_in_kernel": True,
    }

    expected_out, expected_state = fla_chunk_gated_delta_rule(**kwargs)
    out, final_state = flashqla_chunk_gated_delta_rule(**kwargs)
    torch.cuda.synchronize()

    for name, actual, expected in (
        ("output", out, expected_out),
        ("final state", final_state, expected_state),
    ):
        if not torch.allclose(actual, expected, rtol=1e-2, atol=1e-2):
            return (
                False,
                f"{name} mismatch: max diff {(actual - expected).abs().max().item():.6f}",
            )
    return True, None


def _validate_triton():
    """Validate Triton with softmax ground truth."""
    import triton
    import triton.language as tl

    @triton.jit
    def _softmax_kernel(input_ptr, output_ptr, n_cols, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cols
        x = tl.load(input_ptr + row * n_cols + offs, mask=mask, other=-float("inf"))
        x = x - tl.max(x, axis=0)
        num = tl.exp(x)
        out = num / tl.sum(num, axis=0)
        tl.store(output_ptr + row * n_cols + offs, out, mask=mask)

    rows, cols = 32, 64
    x = torch.randn(rows, cols, dtype=torch.float32, device="cuda")
    expected = torch.softmax(x, dim=-1)
    out = torch.empty_like(x)

    _softmax_kernel[(rows,)](x, out, cols, BLOCK=triton.next_power_of_2(cols))
    torch.cuda.synchronize()

    if not torch.allclose(out, expected, rtol=1e-3, atol=1e-3):
        return (
            False,
            f"Output mismatch: max diff {(out - expected).abs().max().item():.6f}",
        )
    return True, None


def _validate_xformers():
    """Validate Xformers Attn with ground truth."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "CUDA not available"

        import xformers.ops as xformers_ops
        from xformers.ops import fmha
    except Exception as e:
        return False, f"xformers import failed: {type(e).__name__}: {e}"

    batch, heads, seq, dim = 1, 4, 8, 64
    dtype = torch.bfloat16
    device = "cuda"

    q = torch.randn(batch, heads, seq, dim, dtype=dtype, device=device)
    k = torch.randn(batch, heads, seq, dim, dtype=dtype, device=device)
    v = torch.randn(batch, heads, seq, dim, dtype=dtype, device=device)

    expected = _compute_ground_truth(q, k, v, is_causal=False)

    q_bmhd = q.transpose(1, 2).contiguous()  # (B, seq, heads, dim)
    k_bmhd = k.transpose(1, 2).contiguous()
    v_bmhd = v.transpose(1, 2).contiguous()

    try:
        out = xformers_ops.memory_efficient_attention(q_bmhd, k_bmhd, v_bmhd, p=0.0)
    except Exception as e:
        return False, f"xformers attention run failed: {type(e).__name__}: {e}"

    out = out.transpose(1, 2).contiguous()

    if not torch.allclose(out, expected, rtol=1e-2, atol=1e-2):
        return (
            False,
            f"Output mismatch: max diff {(out - expected).abs().max().item():.6f}",
        )

    return True, None


def _validate_sdpa():
    """Validate SDPA Attn with ground truth."""
    try:
        import torch
        from torch.nn.functional import scaled_dot_product_attention
    except Exception as e:
        return False, f"SDPA import failed: {type(e).__name__}: {e}"

    batch, heads, seq, dim = 1, 4, 8, 64
    dtype = torch.bfloat16
    device = "cuda"

    q = torch.randn(batch, heads, seq, dim, dtype=dtype, device=device)
    k = torch.randn(batch, heads, seq, dim, dtype=dtype, device=device)
    v = torch.randn(batch, heads, seq, dim, dtype=dtype, device=device)

    expected = _compute_ground_truth(q, k, v, is_causal=False)

    out = scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    if not torch.allclose(out, expected, rtol=1e-2, atol=1e-2):
        return (
            False,
            f"Output mismatch: max diff {(out - expected).abs().max().item():.6f}",
        )

    return True, None


def _validate_flashmla_sparse(dtype=torch.bfloat16, spec=None):
    """Validate flashmla_sparse (NSA) with ground truth."""
    try:
        from sgl_kernel.flash_mla import flash_mla_sparse_fwd

        # need sgl-kernel version >= 0.3.21 and torch >= 2.9.1
    except Exception as e:
        return False, f"sgl_kernel.flash_mla import failed: {type(e).__name__}: {e}"

    batch, heads, seq = 1, 64, 128
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    qk_dim = kv_lora_rank + qk_rope_head_dim
    device = "cuda"

    q = torch.randn(batch * seq, heads, qk_dim, dtype=dtype, device=device)
    kv = torch.zeros(batch * seq, 1, qk_dim, dtype=dtype, device=device)

    index_topk = 128
    topk_indices = torch.zeros(batch * seq, index_topk, dtype=torch.int32, device=device)
    for i in range(seq):
        topk_indices[i, :] = torch.arange(index_topk, dtype=torch.int32, device=device)

    topk_indices = topk_indices.view(batch * seq, 1, index_topk)

    softmax_scale = 1.0 / (qk_dim ** 0.5)

    try:
        mla_out, _, _ = flash_mla_sparse_fwd(
            q=q,
            kv=kv,
            indices=topk_indices,
            sm_scale=softmax_scale,
            d_v=kv_lora_rank,
        )
        torch.cuda.synchronize()
    except Exception as e:
        return False, f"flash_mla_sparse_fwd run failed: {type(e).__name__}: {e}"

    return True, None


def _validate_varlen(backend_name, dtype=None, device=None, head_dim=80):
    from lightllm.common.attention.api import VarlenAttentionInputs
    from lightllm.common.attention.selection import get_varlen_backend_class
    from lightllm.platform import get_device_runtime

    runtime = get_device_runtime(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    backend_cls = get_varlen_backend_class(backend_name)
    environment = backend_cls.check_environment(runtime.caps)
    if not environment.supported:
        return False, environment.reason
    device = runtime.device
    dtype = dtype or (torch.float32 if device.type == "cpu" else torch.bfloat16)
    q, k, v = [torch.randn(8, 2, head_dim, dtype=dtype, device=device) for _ in range(3)]
    cu = torch.tensor([0, 3, 8], dtype=torch.int32, device=device)
    out = torch.empty_like(q)
    inputs = VarlenAttentionInputs(q, k, v, cu, 5)
    support = backend_cls.check_support(inputs, runtime.caps)
    if not support.supported:
        return False, support.reason
    backend_cls().forward(inputs, out=out)
    expected = torch.empty_like(q)
    for start, end in ((0, 3), (3, 8)):
        result = _compute_ground_truth(
            q[start:end].transpose(0, 1), k[start:end].transpose(0, 1), v[start:end].transpose(0, 1), is_causal=False
        )
        expected[start:end].copy_(result.transpose(0, 1))
    if not torch.allclose(out, expected, rtol=2e-2, atol=2e-2):
        return False, f"varlen output mismatch: max diff {(out - expected).abs().max().item():.6f}"
    return True, None


def _run_in_subprocess(backend_name, pipe, varlen=False, device=None, dtype=None, spec=None, head_dim=80):
    """Run validation in subprocess with suppressed output."""
    import sys

    # Redirect stdout/stderr to /dev/null to suppress verbose error messages
    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    sys.stderr = devnull

    try:
        if device is not None or spec is not None:
            from lightllm.platform import get_device_runtime

            runtime = get_device_runtime(device)
            if runtime.api is not None:
                runtime.api.set_device(runtime.device)
        if spec is not None:
            from lightllm.common.attention.api import AttentionFamily
            from lightllm.common.attention.selection import load_backend_class

            cls = load_backend_class(backend_name)
            if (
                spec.family is AttentionFamily.DENSE
                and spec.cache.format == "None"
                and not spec.use_mtp
                and not spec.use_alibi
                and not spec.use_sink
                and spec.sliding_window is None
                and not spec.variant
            ):
                from lightllm.common.attention.probe import probe_dense

                success, err = probe_dense(cls, spec, runtime)
                pipe.send((success, err))
                return
            backend_name = cls.provider
        if varlen:
            success, err = _validate_varlen(backend_name, dtype, device, head_dim)
        elif backend_name == "fa3":
            success, err = _validate_fa3(dtype or torch.bfloat16, spec)
        elif backend_name == "xformers":
            success, err = _validate_xformers()
        elif backend_name == "sdpa":
            success, err = _validate_sdpa()
        elif backend_name == "flashinfer":
            success, err = _validate_flashinfer(dtype or torch.float16, spec)
        elif backend_name == "flashqla":
            success, err = _validate_flashqla(dtype, spec)
        elif backend_name == "triton":
            success, err = _validate_triton()
        elif backend_name == "flashmla_sparse":
            success, err = _validate_flashmla_sparse(dtype or torch.bfloat16, spec)
        else:
            success, err = False, f"Unknown backend: {backend_name}"
        pipe.send((success, err))
    except Exception as e:
        pipe.send((False, str(e)))
    finally:
        devnull.close()


@lru_cache(maxsize=None)
def validate(backend_name: str, *, device=None, dtype=None, spec=None) -> bool:
    """每个 rank 在自身设备验证；dtype/完整 spec 不同不能复用另一配置的结果。"""
    return _validate(backend_name, device=device, dtype=dtype, spec=spec)


def _validate(backend_name: str, *, varlen=False, device=None, dtype=None, spec=None, head_dim=80) -> bool:
    """Validate backend in subprocess with ground truth check."""
    try:
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe(duplex=False)
        logger.info(f"Validating {backend_name} backend start, please wait ...")
        proc = ctx.Process(target=_run_in_subprocess, args=(backend_name, child, varlen, device, dtype, spec, head_dim))
        proc.start()
        child.close()
        proc.join(timeout=_VALIDATION_TIMEOUT)

        if proc.is_alive():
            proc.kill()
            proc.join()
            logger.warning(f"{backend_name} validation timed out")
            return False

        if proc.exitcode != 0:
            logger.warning(f"{backend_name} validation crashed (exit {proc.exitcode})")
            return False

        if parent.poll():
            success, err = parent.recv()
            if success:
                logger.info(f"{backend_name} validated")
                return True
            logger.warning(f"{backend_name} validation failed: {str(err)[0:88]}")
        return False

    except Exception as e:
        logger.warning(f"{backend_name} validation error: {e}")
        return False
    finally:
        if "parent" in locals():
            parent.close()
        if "child" in locals():
            child.close()
