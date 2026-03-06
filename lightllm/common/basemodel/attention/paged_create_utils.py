"""Paged attention backend selection utilities."""

import os
import torch
from lightllm.utils.envs_utils import get_env_start_args, get_page_size
from lightllm.utils.log_utils import init_logger
from lightllm.utils.backend_validator import validate
from .base_att import BaseAttBackend
from .triton.fp import TritonAttBackend
from .triton.int4kv import Int4kvTritonAttBackend
from .triton.int8kv import Int8kvTritonAttBackend
from .triton.mla import MlaTritonAttBackend
from .fa3.fp8 import Fp8Fa3AttBackend
from .paged_fa3.fp import PagedFa3AttBackend
from .paged_fa3.mla import PagedMlaFa3AttBackend
from .paged_flashinfer.fp import PagedFlashInferAttBackend
from .paged_flashinfer.mla import PagedMlaFlashInferAttBackend

logger = init_logger(__name__)

data_type_to_backend = {
    "None": {
        "triton": TritonAttBackend,
        "fa3": PagedFa3AttBackend,
        "flashinfer": PagedFlashInferAttBackend,
    },
    "int4kv": {
        "triton": Int4kvTritonAttBackend,
        "fa3": Fp8Fa3AttBackend,
    },
    "int8kv": {
        "triton": Int8kvTritonAttBackend,
        "fa3": Fp8Fa3AttBackend,
    },
}

mla_data_type_to_backend = {
    "None": {
        "triton": MlaTritonAttBackend,
        "fa3": PagedMlaFa3AttBackend,
        "flashinfer": PagedMlaFlashInferAttBackend,
    },
}


def _auto_select_backend(
    llm_dtype: str, is_mla: bool = False, priority_list: list = ["fa3", "flashinfer", "triton"]
) -> type:
    backend_map = mla_data_type_to_backend if is_mla else data_type_to_backend

    for backend_name in priority_list:
        if backend_name not in backend_map[llm_dtype]:
            continue
        if validate(backend_name):
            logger.info(f"Auto-selected {backend_name} backend (validated)")
            return backend_map[llm_dtype][backend_name]

    logger.warning("No backend validation succeeded, falling back to triton")
    return backend_map[llm_dtype]["triton"]


def get_prefill_att_backend_class(index=0, priority_list: list = ["fa3", "flashinfer", "triton"]) -> BaseAttBackend:
    args = get_env_start_args()
    llm_dtype = args.llm_kv_type
    backend_str = args.llm_prefill_att_backend[index]
    if backend_str != "auto":
        if backend_str not in data_type_to_backend[llm_dtype]:
            raise RuntimeError(
                f"att backend {backend_str} does not support llm_kv_type={llm_dtype} when PAGE_SIZE={get_page_size()}"
            )
        return data_type_to_backend[llm_dtype][backend_str]
    else:
        return _auto_select_backend(llm_dtype, is_mla=False, priority_list=priority_list)


def get_decode_att_backend_class(index=0, priority_list: list = ["fa3", "flashinfer", "triton"]) -> BaseAttBackend:
    args = get_env_start_args()
    llm_dtype = args.llm_kv_type
    backend_str = args.llm_decode_att_backend[index]
    if backend_str != "auto":
        if backend_str not in data_type_to_backend[llm_dtype]:
            raise RuntimeError(
                f"att backend {backend_str} does not support llm_kv_type={llm_dtype} when PAGE_SIZE={get_page_size()}"
            )
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
