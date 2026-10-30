"""Attention backend selection utilities."""

from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger
from lightllm.utils.backend_validator import validate
from .base_att import BaseAttBackend
from .import_utils import load_backend_class

logger = init_logger(__name__)

# Lazy backend class references by data type.
data_type_to_backend = {
    "None": {
        "triton": "triton.fp:TritonAttBackend",
        "fa3": "fa3.fp:Fa3AttBackend",
        "flashinfer": "flashinfer.fp:FlashInferAttBackend",
    },
    "int4kv": {
        "triton": "triton.int4kv:Int4kvTritonAttBackend",
    },
    "int8kv": {
        "triton": "triton.int8kv:Int8kvTritonAttBackend",
    },
    "fp8kv_sph": {
        "fa3": "fa3.fp8:Fp8Fa3AttBackend",
    },
    "fp8kv_spt": {
        "flashinfer": "flashinfer.fp8:Fp8FlashInferAttBackend",
    },
}

mla_data_type_to_backend = {
    "None": {
        "triton": "triton.mla:MlaTritonAttBackend",
        "fa3": "fa3.mla:MlaFa3AttBackend",
        "flashinfer": "flashinfer.mla:MlaFlashInferAttBackend",
    },
}

nsa_data_type_to_backend = {
    "None": {
        "flashmla_sparse": "nsa.flashmla_sparse:NsaFlashMlaSparseAttBackend",
        # Future backends: "fa3", "tilelang", "aiter"
    },
    "fp8kv_dsa": {
        "flashmla_sparse": "nsa.fp8_flashmla_sparse:NsaFlashMlaFp8SparseAttBackend",
    },
}

neo_data_type_to_backend = {
    "None": {
        "triton": "triton.neo:NeoTritonAttBackend",
        "fa3": "fa3.neo:NeoFa3AttBackend",
    }
}


def _auto_select_backend(
    llm_dtype: str,
    kv_type_to_backend: dict,
    priority_list: list = ["fa3", "flashinfer", "triton"],
) -> type:
    """Auto-select the best available backend with validation.

    Priority follows the provided priority_list.
    Each backend is validated in a subprocess with ground truth checks.
    """
    backend_map = kv_type_to_backend[llm_dtype]

    args = get_env_start_args()
    if args.enable_ep_moe:
        logger.info("Expert parallelism with MoE enabled, excluding flashinfer attention backend")
        priority_list = [name for name in priority_list if name != "flashinfer"]

    for backend_name in priority_list:
        if backend_name in backend_map and validate(backend_name):
            logger.info(f"Auto-selected {backend_name} backend (validated)")
            return load_backend_class(backend_map[backend_name])

    # Fallback to triton without validation (should not happen)
    logger.warning("No backend validation succeeded, falling back to triton")
    return load_backend_class(backend_map["triton"])


def _get_decode_backend_priority(priority_list: list, mtp_step: int) -> list:
    """Return the auto-selection priority for decode attention."""
    # With MTP, FA3 can make better use of Tensor Core compute and delivers better decode performance.
    if mtp_step <= 0 or "fa3" not in priority_list:
        return priority_list
    return ["fa3"] + [backend_name for backend_name in priority_list if backend_name != "fa3"]


def get_prefill_att_backend_class(index=0, priority_list: list = ["fa3", "flashinfer", "triton"]) -> BaseAttBackend:
    args = get_env_start_args()
    llm_dtype = args.llm_kv_type
    backend_str = args.llm_prefill_att_backend[index]
    if backend_str != "auto":
        return load_backend_class(data_type_to_backend[llm_dtype][backend_str])
    return _auto_select_backend(llm_dtype, kv_type_to_backend=data_type_to_backend, priority_list=priority_list)


def get_decode_att_backend_class(index=0, priority_list: list = ["flashinfer", "fa3", "triton"]) -> BaseAttBackend:
    args = get_env_start_args()
    llm_dtype = args.llm_kv_type
    backend_str = args.llm_decode_att_backend[index]
    if backend_str != "auto":
        return load_backend_class(data_type_to_backend[llm_dtype][backend_str])
    priority_list = _get_decode_backend_priority(priority_list, args.mtp_step)
    return _auto_select_backend(llm_dtype, kv_type_to_backend=data_type_to_backend, priority_list=priority_list)


def get_mla_prefill_att_backend_class(index=0, priority_list: list = ["fa3", "flashinfer", "triton"]) -> BaseAttBackend:
    args = get_env_start_args()
    llm_dtype = args.llm_kv_type
    backend_str = args.llm_prefill_att_backend[index]
    if backend_str != "auto":
        return load_backend_class(mla_data_type_to_backend[llm_dtype][backend_str])
    return _auto_select_backend(llm_dtype, kv_type_to_backend=mla_data_type_to_backend, priority_list=priority_list)


def get_mla_decode_att_backend_class(index=0, priority_list: list = ["flashinfer", "fa3", "triton"]) -> BaseAttBackend:
    args = get_env_start_args()
    llm_dtype = args.llm_kv_type
    backend_str = args.llm_decode_att_backend[index]
    if backend_str != "auto":
        return load_backend_class(mla_data_type_to_backend[llm_dtype][backend_str])
    priority_list = _get_decode_backend_priority(priority_list, args.mtp_step)
    return _auto_select_backend(llm_dtype, kv_type_to_backend=mla_data_type_to_backend, priority_list=priority_list)


def get_nsa_prefill_att_backend_class(
    index=0, priority_list: list = ["flashmla_sparse"], backend_map=nsa_data_type_to_backend
) -> BaseAttBackend:
    args = get_env_start_args()
    llm_dtype = args.llm_kv_type
    backend_str = args.llm_prefill_att_backend[index]
    if backend_str != "auto":
        return load_backend_class(backend_map[llm_dtype][backend_str])
    return _auto_select_backend(llm_dtype, kv_type_to_backend=backend_map, priority_list=priority_list)


def get_nsa_decode_att_backend_class(
    index=0, priority_list: list = ["flashmla_sparse"], backend_map=nsa_data_type_to_backend
) -> BaseAttBackend:
    args = get_env_start_args()
    llm_dtype = args.llm_kv_type
    backend_str = args.llm_decode_att_backend[index]
    if backend_str != "auto":
        return load_backend_class(backend_map[llm_dtype][backend_str])
    return _auto_select_backend(llm_dtype, kv_type_to_backend=backend_map, priority_list=priority_list)


def get_neo_prefill_att_backend_class(index=0, priority_list: list = ["fa3", "triton"]) -> BaseAttBackend:
    args = get_env_start_args()
    llm_dtype = args.llm_kv_type
    backend_str = args.llm_prefill_att_backend[index]
    backend_map = neo_data_type_to_backend
    if backend_str == "fa3" or (backend_str == "auto" and "fa3" in priority_list):
        # Only Neo FA3 selection needs the optional image-token attention interface.
        from .fa3.neo import HAS_FLASH_ATTN_INTERFACE

        if not HAS_FLASH_ATTN_INTERFACE:
            backend_map = {"None": {"triton": neo_data_type_to_backend["None"]["triton"]}}
    if backend_str != "auto":
        return load_backend_class(backend_map[llm_dtype][backend_str])
    return _auto_select_backend(llm_dtype, kv_type_to_backend=backend_map, priority_list=priority_list)
