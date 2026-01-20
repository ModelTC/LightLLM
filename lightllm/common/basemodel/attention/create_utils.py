import multiprocessing as mp
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger
from .base_att import BaseAttBackend
from .triton.fp import TritonAttBackend
from .triton.int4kv import Int4kvTritonAttBackend
from .triton.int8kv import Int8kvTritonAttBackend
from .triton.mla import MlaTritonAttBackend
from .fa3.fp import Fa3AttBackend
from .fa3.fp8 import Fp8Fa3AttBackend
from .fa3.mla import MlaFa3AttBackend
from .flashinfer.fp8 import Fp8FlashInferAttBackend
from .flashinfer.fp import FlashInferAttBackend
from .flashinfer.mla import MlaFlashInferAttBackend

logger = init_logger(__name__)

# Timeout for backend validation in seconds
_VALIDATION_TIMEOUT = 30

data_type_to_backend = {
    "None": {
        "triton": TritonAttBackend,
        "fa3": Fa3AttBackend,
        "flashinfer": FlashInferAttBackend,
    },
    "int4kv": {
        "triton": Int4kvTritonAttBackend,
        "fa3": Fp8Fa3AttBackend,
        "flashinfer": Fp8FlashInferAttBackend,
    },
    "int8kv": {
        "triton": Int8kvTritonAttBackend,
        "fa3": Fp8Fa3AttBackend,
        "flashinfer": Fp8FlashInferAttBackend,
    },
}

mla_data_type_to_backend = {
    "None": {
        "triton": MlaTritonAttBackend,
        "fa3": MlaFa3AttBackend,
        "flashinfer": MlaFlashInferAttBackend,
    },
}


def _is_fa3_available() -> bool:
    """Check if FA3 backend can be used.

    FA3 requires:
    1. Hopper GPU (H100/H200/H800)
    2. sgl_kernel package with flash_attn_with_kvcache
    """
    try:
        from lightllm.utils.device_utils import is_hopper
        from lightllm.utils.sgl_utils import flash_attn_with_kvcache

        if not is_hopper():
            return False
        if flash_attn_with_kvcache is None:
            return False
        return True
    except Exception:
        return False


def _is_flashinfer_available() -> bool:
    """Check if FlashInfer backend can be used."""
    try:
        import flashinfer  # noqa: F401

        return True
    except ImportError:
        return False


def _validate_backend_in_subprocess(backend_name: str, result_pipe) -> None:
    """Run backend validation in a forked subprocess.

    This function runs in a child process to safely test if a backend works.
    If the backend crashes, only the child process dies, not the main server.
    """
    try:
        import torch

        if backend_name == "fa3":
            from lightllm.utils.sgl_utils import flash_attn_with_kvcache

            if flash_attn_with_kvcache is None:
                result_pipe.send(("error", "flash_attn_with_kvcache is None"))
                return

            # Create minimal test tensors and call the actual function
            batch_size, seq_len, num_heads, head_dim = 1, 4, 1, 64
            q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
            k_cache = torch.randn(16, 1, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
            v_cache = torch.randn(16, 1, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
            page_table = torch.arange(seq_len, dtype=torch.int32, device="cuda").unsqueeze(0)
            cache_seqlens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")

            # Actually call the function
            _ = flash_attn_with_kvcache(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
            )
            torch.cuda.synchronize()
            result_pipe.send(("ok", None))

        elif backend_name == "flashinfer":
            import flashinfer

            # Create a minimal decode attention test
            num_heads, head_dim = 1, 64
            batch_size, page_size, num_pages = 1, 16, 4

            q = torch.randn(batch_size, num_heads, head_dim, dtype=torch.float16, device="cuda")
            kv_data = torch.randn(num_pages, 2, page_size, num_heads, head_dim, dtype=torch.float16, device="cuda")
            kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device="cuda")
            kv_indices = torch.arange(num_pages, dtype=torch.int32, device="cuda")
            kv_last_page_len = torch.tensor([page_size], dtype=torch.int32, device="cuda")

            workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
            wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD")
            wrapper.begin_forward(
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                num_heads,
                num_heads,
                head_dim,
                page_size,
                "NONE",
                q.dtype,
            )
            _ = wrapper.forward(q, kv_data)
            wrapper.end_forward()
            torch.cuda.synchronize()
            result_pipe.send(("ok", None))

        elif backend_name == "triton":
            # Triton validation: compile and run a minimal kernel
            import triton
            import triton.language as tl

            @triton.jit
            def _test_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
                pid = tl.program_id(0)
                offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                tl.store(x_ptr + offsets, tl.zeros((BLOCK_SIZE,), dtype=tl.float32))

            x = torch.ones(128, dtype=torch.float32, device="cuda")
            _test_kernel[(1,)](x, BLOCK_SIZE=128)
            torch.cuda.synchronize()

            if not torch.allclose(x, torch.zeros_like(x)):
                result_pipe.send(("error", "Triton kernel output mismatch"))
                return
            result_pipe.send(("ok", None))

        else:
            result_pipe.send(("error", f"Unknown backend: {backend_name}"))

    except Exception as e:
        result_pipe.send(("error", str(e)))


def _try_backend(backend_name: str) -> bool:
    """Validate a backend by running actual operations in a forked subprocess.

    This isolates validation failures (including crashes) from the main process.
    """
    try:
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_validate_backend_in_subprocess, args=(backend_name, child_conn))
        proc.start()
        proc.join(timeout=_VALIDATION_TIMEOUT)

        if proc.is_alive():
            proc.kill()
            proc.join()
            logger.warning(f"Backend {backend_name} validation timed out")
            return False

        if proc.exitcode != 0:
            logger.warning(f"Backend {backend_name} validation subprocess crashed (exit code {proc.exitcode})")
            return False

        if parent_conn.poll():
            status, error = parent_conn.recv()
            if status == "ok":
                return True
            else:
                logger.warning(f"Backend {backend_name} validation failed: {error}")
                return False
        else:
            logger.warning(f"Backend {backend_name} validation produced no result")
            return False

    except Exception as e:
        logger.warning(f"Backend {backend_name} validation exception: {e}")
        return False


def _auto_select_backend(llm_dtype: str, is_mla: bool = False) -> type:
    """Auto-select the best available backend with validation.

    Priority: FA3 > FlashInfer > Triton
    """
    backend_map = mla_data_type_to_backend if is_mla else data_type_to_backend

    # Build candidate list based on availability checks
    candidates = []
    if _is_fa3_available():
        candidates.append("fa3")
    if _is_flashinfer_available():
        candidates.append("flashinfer")
    candidates.append("triton")  # Always available as fallback

    # Try each candidate with runtime validation
    for backend_name in candidates:
        if _try_backend(backend_name):
            logger.info(f"Auto-selected {backend_name} backend (validated)")
            return backend_map[llm_dtype][backend_name]

    # Should never reach here since triton is always available
    logger.warning("No backend validation succeeded, falling back to triton")
    return backend_map[llm_dtype]["triton"]


def get_prefill_att_backend_class(index=0) -> BaseAttBackend:
    args = get_env_start_args()
    llm_dtype = args.llm_kv_type
    backend_str = args.llm_prefill_att_backend[index]
    if backend_str != "None":
        return data_type_to_backend[llm_dtype][backend_str]
    else:
        # Auto-select best available backend
        return _auto_select_backend(llm_dtype, is_mla=False)


def get_decode_att_backend_class(index=0) -> BaseAttBackend:
    args = get_env_start_args()
    llm_dtype = args.llm_kv_type
    backend_str = args.llm_decode_att_backend[index]
    if backend_str != "None":
        return data_type_to_backend[llm_dtype][backend_str]
    else:
        # Auto-select best available backend
        return _auto_select_backend(llm_dtype, is_mla=False)


def get_mla_prefill_att_backend_class(index=0) -> BaseAttBackend:
    args = get_env_start_args()
    llm_dtype = args.llm_kv_type
    backend_str = args.llm_prefill_att_backend[index]
    if backend_str != "None":
        return mla_data_type_to_backend[llm_dtype][backend_str]
    else:
        # Auto-select best available backend for MLA
        return _auto_select_backend(llm_dtype, is_mla=True)


def get_mla_decode_att_backend_class(index=0) -> BaseAttBackend:
    args = get_env_start_args()
    llm_dtype = args.llm_kv_type
    backend_str = args.llm_decode_att_backend[index]
    if backend_str != "None":
        return mla_data_type_to_backend[llm_dtype][backend_str]
    else:
        # Auto-select best available backend for MLA
        return _auto_select_backend(llm_dtype, is_mla=True)
