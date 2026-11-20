"""Attention backend selection utilities."""

from typing import TYPE_CHECKING

from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger
from lightllm.utils.backend_validator import validate, validate_attention
from .base_att import BaseAttBackend
from .import_utils import load_backend_class

if TYPE_CHECKING:
    from lightllm.common.basemodel.basemodel import TpPartBaseModel

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

    raise RuntimeError(f"No available attention backend for KV type {llm_dtype} among {priority_list}")


def _select_attention_backend(
    model: "TpPartBaseModel", family: str, phase: str, index: int, backend_map: dict, priority_list: list
) -> type[BaseAttBackend]:
    from lightllm.utils.attention_probe import AttentionProbe

    args = get_env_start_args()
    kv_type = args.llm_kv_type
    if phase == "prefill":
        requested = args.llm_prefill_att_backend[index]
    else:
        requested = args.llm_decode_att_backend[index]
        if family != "sparse_mla":
            priority_list = _get_decode_backend_priority(priority_list, args.mtp_step)

    # Quantized GQA and model-specific sparse adapters have different operand layouts.
    if (family == "gqa" and kv_type != "None") or (
        family == "sparse_mla" and (kv_type != "fp8kv_dsa" or backend_map is not nsa_data_type_to_backend)
    ):
        if requested != "auto":
            return load_backend_class(backend_map[kv_type][requested])
        return _auto_select_backend(kv_type, kv_type_to_backend=backend_map, priority_list=priority_list)

    candidates = backend_map.get(kv_type, {})
    if requested == "auto":
        names = [
            name for name in priority_list if name in candidates and not (name == "flashinfer" and args.enable_ep_moe)
        ]
    else:
        names = [requested] if requested in candidates else []
    probe = AttentionProbe.from_model(model=model, family=family, phase=phase)
    for name in names:
        path = candidates[name]
        if validate_attention(backend_path=path, probe=probe, device=model.req_manager.req_to_token_indexs.device):
            logger.info(f"Selected {family} {phase} backend {name}")
            return load_backend_class(path)
    raise RuntimeError(
        f"No supported {family} {phase} backend for {kv_type} KV, {model.data_type}, "
        f"requested {requested}, tried {names}. See per-device validation errors."
    )


def _get_decode_backend_priority(priority_list: list, mtp_step: int) -> list:
    """Return the auto-selection priority for decode attention."""
    # With MTP, FA3 can make better use of Tensor Core compute and delivers better decode performance.
    if mtp_step <= 0 or "fa3" not in priority_list:
        return priority_list
    return ["fa3"] + [backend_name for backend_name in priority_list if backend_name != "fa3"]


def get_prefill_att_backend_class(
    index: int = 0, priority_list: list = ["fa3", "flashinfer", "triton"], *, model: "TpPartBaseModel"
) -> type[BaseAttBackend]:
    return _select_attention_backend(
        model=model,
        family="gqa",
        phase="prefill",
        index=index,
        backend_map=data_type_to_backend,
        priority_list=priority_list,
    )


def get_decode_att_backend_class(
    index: int = 0, priority_list: list = ["flashinfer", "fa3", "triton"], *, model: "TpPartBaseModel"
) -> type[BaseAttBackend]:
    return _select_attention_backend(
        model=model,
        family="gqa",
        phase="decode",
        index=index,
        backend_map=data_type_to_backend,
        priority_list=priority_list,
    )


def get_mla_prefill_att_backend_class(
    index: int = 0, priority_list: list = ["fa3", "flashinfer", "triton"], *, model: "TpPartBaseModel"
) -> type[BaseAttBackend]:
    return _select_attention_backend(
        model=model,
        family="mla",
        phase="prefill",
        index=index,
        backend_map=mla_data_type_to_backend,
        priority_list=priority_list,
    )


def get_mla_decode_att_backend_class(
    index: int = 0, priority_list: list = ["flashinfer", "fa3", "triton"], *, model: "TpPartBaseModel"
) -> type[BaseAttBackend]:
    return _select_attention_backend(
        model=model,
        family="mla",
        phase="decode",
        index=index,
        backend_map=mla_data_type_to_backend,
        priority_list=priority_list,
    )


def get_nsa_prefill_att_backend_class(
    index: int = 0,
    priority_list: list = ["flashmla_sparse"],
    backend_map=nsa_data_type_to_backend,
    *,
    model: "TpPartBaseModel",
) -> type[BaseAttBackend]:
    return _select_attention_backend(
        model=model,
        family="sparse_mla",
        phase="prefill",
        index=index,
        backend_map=backend_map,
        priority_list=priority_list,
    )


def get_nsa_decode_att_backend_class(
    index: int = 0,
    priority_list: list = ["flashmla_sparse"],
    backend_map=nsa_data_type_to_backend,
    *,
    model: "TpPartBaseModel",
) -> type[BaseAttBackend]:
    return _select_attention_backend(
        model=model,
        family="sparse_mla",
        phase="decode",
        index=index,
        backend_map=backend_map,
        priority_list=priority_list,
    )


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
