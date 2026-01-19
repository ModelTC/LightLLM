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
        import flashinfer

        return True
    except ImportError:
        return False


def _try_backend(backend_name: str) -> bool:
    """Try to validate a backend works at runtime."""
    try:
        if backend_name == "fa3":
            from lightllm.utils.sgl_utils import flash_attn_with_kvcache

            # Verify function is callable (not None)
            if flash_attn_with_kvcache is None:
                return False
            return True

        elif backend_name == "flashinfer":
            import torch
            import flashinfer  # noqa: F401

            # Try creating a minimal workspace buffer to verify flashinfer works
            _ = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
            return True

        elif backend_name == "triton":
            return True  # Always available

    except Exception as e:
        logger.warning(f"Backend {backend_name} validation failed: {e}")
        return False
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
