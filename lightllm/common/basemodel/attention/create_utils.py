from lightllm.utils.envs_utils import get_env_start_args
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


def get_prefill_att_backend_class(index=0) -> BaseAttBackend:
    args = get_env_start_args()
    llm_dtype = args.llm_kv_type
    backend_str = args.llm_prefill_att_backend[index]
    if backend_str != "None":
        return data_type_to_backend[llm_dtype][backend_str]
    else:
        # 根据环境自动选择最好的
        raise NotImplementedError(f"error")


def get_decode_att_backend_class(index=0) -> BaseAttBackend:
    args = get_env_start_args()
    llm_dtype = args.llm_kv_type
    backend_str = args.llm_decode_att_backend[index]
    if backend_str != "None":
        return data_type_to_backend[llm_dtype][backend_str]
    else:
        # 根据环境自动选择最好的
        raise NotImplementedError(f"error")


def get_mla_prefill_att_backend_class(index=0) -> BaseAttBackend:
    args = get_env_start_args()
    llm_dtype = args.llm_kv_type
    backend_str = args.llm_prefill_att_backend[index]
    if backend_str != "None":
        return mla_data_type_to_backend[llm_dtype][backend_str]
    else:
        raise NotImplementedError(f"error")


def get_mla_decode_att_backend_class(index=0) -> BaseAttBackend:
    args = get_env_start_args()
    llm_dtype = args.llm_kv_type
    backend_str = args.llm_decode_att_backend[index]
    if backend_str != "None":
        return mla_data_type_to_backend[llm_dtype][backend_str]
    else:
        raise NotImplementedError(f"error")
