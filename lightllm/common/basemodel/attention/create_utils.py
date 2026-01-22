"""Attention backend selection utilities."""

import os
import torch
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger
from lightllm.utils.backend_validator import validate
from .base_att import BaseAttBackend
from .triton.fp import TritonAttBackend
from .triton.int4kv import Int4kvTritonAttBackend
from .triton.int8kv import Int8kvTritonAttBackend
from .triton.mla import MlaTritonAttBackend
from .fa3.fp import Fa3AttBackend, Fa3ViTAttBackend
from .fa3.fp8 import Fp8Fa3AttBackend
from .fa3.mla import MlaFa3AttBackend
from .flashinfer.fp8 import Fp8FlashInferAttBackend
from .flashinfer.fp import FlashInferAttBackend
from .flashinfer.mla import MlaFlashInferAttBackend

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

vit_data_type_to_backend = {
    "None": {
        "triton": TritonAttBackend,
        "fa3": Fa3ViTAttBackend,
    },
}


def _auto_select_backend(
    llm_dtype: str, is_mla: bool = False, priority_list: list = ["fa3", "flashinfer", "triton"]
) -> type:
    """Auto-select the best available backend with validation.

    Priority: FA3 > FlashInfer > Triton
    Each backend is validated in a subprocess with ground truth checks.
    """
    backend_map = mla_data_type_to_backend if is_mla else data_type_to_backend

    for backend_name in priority_list:
        if validate(backend_name):
            logger.info(f"Auto-selected {backend_name} backend (validated)")
            print(f"llm_dtype is {llm_dtype}, backend_name is {backend_name} ")
            return backend_map[llm_dtype][backend_name]

    # Fallback to triton without validation (should not happen)
    logger.warning("No backend validation succeeded, falling back to triton")
    return backend_map[llm_dtype]["triton"]


def _auto_select_vit_backend(llm_dtype: str, priority_list: list = ["fa3", "triton"]) -> type:
    """Auto-select the best available backend with validation for vit.

    Priority: FA3 > Triton
    Each backend is validated in a subprocess with ground truth checks.
    """
    backend_map = vit_data_type_to_backend

    for backend_name in priority_list:
        if validate(backend_name):
            logger.info(f"Auto-selected {backend_name} backend (validated) for ViT")
            print(f"llm_dtype is {llm_dtype}, backend_name is {backend_name} ")
            return backend_map[llm_dtype][backend_name]

    # Fallback to triton without validation (should not happen)
    logger.warning("No backend validation succeeded for vit, falling back to triton")
    return backend_map[llm_dtype]["triton"]


def get_prefill_att_backend_class(index=0, priority_list: list = ["fa3", "flashinfer", "triton"]) -> BaseAttBackend:
    args = get_env_start_args()
    llm_dtype = args.llm_kv_type
    backend_str = args.llm_prefill_att_backend[index]
    if backend_str != "auto":
        return data_type_to_backend[llm_dtype][backend_str]
    else:
        return _auto_select_backend(llm_dtype, is_mla=False, priority_list=priority_list)


def get_decode_att_backend_class(index=0, priority_list: list = ["fa3", "flashinfer", "triton"]) -> BaseAttBackend:
    args = get_env_start_args()
    llm_dtype = args.llm_kv_type
    backend_str = args.llm_decode_att_backend[index]
    if backend_str != "auto":
        return data_type_to_backend[llm_dtype][backend_str]
    else:
        return _auto_select_backend(llm_dtype, is_mla=False, priority_list=priority_list)


def get_mla_prefill_att_backend_class(index=0, priority_list: list = ["fa3", "flashinfer", "triton"]) -> BaseAttBackend:
    args = get_env_start_args()
    llm_dtype = args.llm_kv_type
    backend_str = args.llm_prefill_att_backend[index]
    if backend_str != "auto":
        return mla_data_type_to_backend[llm_dtype][backend_str]
    else:
        return _auto_select_backend(llm_dtype, is_mla=True, priority_list=priority_list)


def get_mla_decode_att_backend_class(index=0, priority_list: list = ["fa3", "flashinfer", "triton"]) -> BaseAttBackend:
    args = get_env_start_args()
    llm_dtype = args.llm_kv_type
    backend_str = args.llm_decode_att_backend[index]
    if backend_str != "auto":
        return mla_data_type_to_backend[llm_dtype][backend_str]
    else:
        return _auto_select_backend(llm_dtype, is_mla=True, priority_list=priority_list)


def get_vit_att_backend_class(index=0, priority_list: list = ["fa3", "triton"]) -> BaseAttBackend:
    return _auto_select_vit_backend(llm_dtype="None", priority_list=priority_list)
