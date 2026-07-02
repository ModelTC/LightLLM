"""Attention backend selection utilities."""
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger
from lightllm.utils.backend_validator import validate
from typing import Dict
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
from .nsa.flashmla_sparse import NsaFlashMlaSparseAttBackend
from .nsa.fp8_flashmla_sparse import NsaFlashMlaFp8SparseAttBackend

logger = init_logger(__name__)

# Backend class mappings by data type
data_type_to_backend = {
    "None": {
        "triton": TritonAttBackend,
        "fa3": Fa3AttBackend,
        "flashinfer": FlashInferAttBackend,
    },
    "int4kv": {
        "triton": Int4kvTritonAttBackend,
        # "fa3": Fp8Fa3AttBackend,
        # "flashinfer": Fp8FlashInferAttBackend,
    },
    "int8kv": {
        "triton": Int8kvTritonAttBackend,
        # "fa3": Fp8Fa3AttBackend,
        # "flashinfer": Fp8FlashInferAttBackend,
    },
    "fp8kv_sph": {
        "fa3": Fp8Fa3AttBackend,
    },
    "fp8kv_spt": {
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

nsa_data_type_to_backend = {
    "None": {
        "flashmla_sparse": NsaFlashMlaSparseAttBackend,
        # Future backends: "fa3", "tilelang", "aiter"
    },
    "fp8kv_dsa": {
        "flashmla_sparse": NsaFlashMlaFp8SparseAttBackend,
    },
}


def _auto_select_backend(
    llm_dtype: str,
    kv_type_to_backend: Dict[str, Dict[str, BaseAttBackend]],
    priority_list: list = ["fa3", "flashinfer", "triton"],
) -> type:
    """Auto-select the best available backend with validation.

    Priority follows the provided priority_list.
    Each backend is validated in a subprocess with ground truth checks.
    """
    backend_map = kv_type_to_backend

    args = get_env_start_args()
    if args.enable_ep_moe:
        logger.info("Expert parallelism with MoE enabled, excluding flashinfer attention backend")
        priority_list = [name for name in priority_list if name != "flashinfer"]

    for backend_name in priority_list:
        if backend_name in backend_map[llm_dtype] and validate(backend_name):
            logger.info(f"Auto-selected {backend_name} backend (validated)")
            return backend_map[llm_dtype][backend_name]

    # Fallback to triton without validation (should not happen)
    logger.warning("No backend validation succeeded, falling back to triton")
    return backend_map[llm_dtype]["triton"]


def get_prefill_att_backend_class(index=0, priority_list: list = ["fa3", "flashinfer", "triton"]) -> BaseAttBackend:
    args = get_env_start_args()
    llm_dtype = args.llm_kv_type
    backend_str = args.llm_prefill_att_backend[index]
    if backend_str != "auto":
        return data_type_to_backend[llm_dtype][backend_str]
    else:
        return _auto_select_backend(llm_dtype, kv_type_to_backend=data_type_to_backend, priority_list=priority_list)


def get_decode_att_backend_class(index=0, priority_list: list = ["flashinfer", "fa3", "triton"]) -> BaseAttBackend:
    args = get_env_start_args()
    llm_dtype = args.llm_kv_type
    backend_str = args.llm_decode_att_backend[index]
    if backend_str != "auto":
        return data_type_to_backend[llm_dtype][backend_str]
    else:
        return _auto_select_backend(llm_dtype, kv_type_to_backend=data_type_to_backend, priority_list=priority_list)


def get_mla_prefill_att_backend_class(index=0, priority_list: list = ["fa3", "flashinfer", "triton"]) -> BaseAttBackend:
    args = get_env_start_args()
    llm_dtype = args.llm_kv_type
    backend_str = args.llm_prefill_att_backend[index]
    if backend_str != "auto":
        return mla_data_type_to_backend[llm_dtype][backend_str]
    else:
        return _auto_select_backend(llm_dtype, kv_type_to_backend=mla_data_type_to_backend, priority_list=priority_list)


def get_mla_decode_att_backend_class(index=0, priority_list: list = ["flashinfer", "fa3", "triton"]) -> BaseAttBackend:
    args = get_env_start_args()
    llm_dtype = args.llm_kv_type
    backend_str = args.llm_decode_att_backend[index]
    if backend_str != "auto":
        return mla_data_type_to_backend[llm_dtype][backend_str]
    else:
        return _auto_select_backend(llm_dtype, kv_type_to_backend=mla_data_type_to_backend, priority_list=priority_list)


def get_nsa_prefill_att_backend_class(index=0, priority_list: list = ["flashmla_sparse"]) -> BaseAttBackend:
    args = get_env_start_args()
    llm_dtype = args.llm_kv_type
    backend_str = args.llm_prefill_att_backend[index]
    if backend_str != "auto":
        return nsa_data_type_to_backend[llm_dtype][backend_str]
    else:
        return _auto_select_backend(llm_dtype, kv_type_to_backend=nsa_data_type_to_backend, priority_list=priority_list)


def get_nsa_decode_att_backend_class(index=0, priority_list: list = ["flashmla_sparse"]) -> BaseAttBackend:
    args = get_env_start_args()
    llm_dtype = args.llm_kv_type
    backend_str = args.llm_decode_att_backend[index]
    if backend_str != "auto":
        return nsa_data_type_to_backend[llm_dtype][backend_str]
    else:
        return _auto_select_backend(llm_dtype, kv_type_to_backend=nsa_data_type_to_backend, priority_list=priority_list)
