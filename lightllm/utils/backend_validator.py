"""Backend validation with subprocess isolation and ground truth checks."""

import multiprocessing as mp
import torch
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

_VALIDATION_TIMEOUT = 30


def _compute_ground_truth(q, k, v, is_causal=True):
    """Ground truth using PyTorch SDPA."""
    with torch.no_grad():
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


def _validate_fa3():
    """Validate FA3 with ground truth."""
    from lightllm.utils.sgl_utils import flash_attn_varlen_func

    if flash_attn_varlen_func is None:
        return False, "flash_attn_varlen_func is None"

    batch, heads, seq, dim = 1, 4, 8, 64
    q = torch.randn(batch, heads, seq, dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(batch, heads, seq, dim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(batch, heads, seq, dim, dtype=torch.bfloat16, device="cuda")

    expected = _compute_ground_truth(q, k, v)

    # Reshape for varlen: (total, heads, dim)
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
        return False, f"Output mismatch: max diff {(out - expected).abs().max().item():.6f}"
    return True, None


def _validate_flashinfer():
    """Validate FlashInfer with ground truth."""
    import flashinfer

    batch, heads, seq, dim = 1, 4, 8, 64
    q = torch.randn(batch, heads, seq, dim, dtype=torch.float16, device="cuda")
    k = torch.randn(batch, heads, seq, dim, dtype=torch.float16, device="cuda")
    v = torch.randn(batch, heads, seq, dim, dtype=torch.float16, device="cuda")

    expected = _compute_ground_truth(q, k, v)

    # Reshape: (batch * seq, heads, dim)
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
        return False, f"Output mismatch: max diff {(out - expected).abs().max().item():.6f}"
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
        return False, f"Output mismatch: max diff {(out - expected).abs().max().item():.6f}"
    return True, None


def _run_in_subprocess(backend_name, pipe):
    """Run validation in subprocess."""
    try:
        if backend_name == "fa3":
            success, err = _validate_fa3()
        elif backend_name == "flashinfer":
            success, err = _validate_flashinfer()
        elif backend_name == "triton":
            success, err = _validate_triton()
        else:
            success, err = False, f"Unknown backend: {backend_name}"
        pipe.send((success, err))
    except Exception as e:
        pipe.send((False, str(e)))


def is_available(backend_name: str) -> bool:
    """Quick check if backend dependencies exist."""
    try:
        if backend_name == "fa3":
            from lightllm.utils.device_utils import is_hopper
            from lightllm.utils.sgl_utils import flash_attn_varlen_func

            return is_hopper() and flash_attn_varlen_func is not None
        elif backend_name == "flashinfer":
            import flashinfer  # noqa: F401

            return True
        elif backend_name == "triton":
            return True
    except Exception:
        pass
    return False


def validate(backend_name: str) -> bool:
    """Validate backend in subprocess with ground truth check."""
    if not is_available(backend_name):
        return False

    try:
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_run_in_subprocess, args=(backend_name, child))
        proc.start()
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
            logger.warning(f"{backend_name} validation failed: {err}")
        return False

    except Exception as e:
        logger.warning(f"{backend_name} validation error: {e}")
        return False
