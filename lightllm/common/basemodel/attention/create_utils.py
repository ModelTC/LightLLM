from lightllm.utils.envs_utils import get_env_start_args
from .base_att import BaseAttBackend
from .triton_backend import TritonAttBackend
from .int4kv_triton_backend import Int4kvTritonAttBackend
from .int8kv_triton_backend import Int8kvTritonAttBackend
from .fa3_backend import Fa3AttBackend
from .fp8_fa3_backend import Fp8Fa3AttBackend
from .flashinfer_backend import FlashInferAttBackend
from .fp8_flashinfer_backend import Fp8FlashInferAttBackend

backend_dict = {
    None: {
        "triton": TritonAttBackend,
        "fa3": Fa3AttBackend,
        "flash_infer": FlashInferAttBackend,
    },
    "int4kv": {
        "triton": Int4kvTritonAttBackend,
        "fa3": Fp8Fa3AttBackend,
        "flash_infer": Fp8FlashInferAttBackend,
    },
    "int8kv": {
        "triton": Int8kvTritonAttBackend,
        "fa3": Fp8Fa3AttBackend,
        "flash_infer": Fp8FlashInferAttBackend,
    },
}


def get_prefill_att_backend_class() -> BaseAttBackend:
    args = get_env_start_args()
    llm_dtype = args.llm_kv_type
    if args.llm_prefill_att_backend is not None:
        return backend_dict[llm_dtype][args.llm_prefill_att_backend]
    else:
        # 根据环境自动选择最好的
        raise NotImplementedError(f"error")


def get_decode_att_backend_class() -> BaseAttBackend:
    args = get_env_start_args()
    llm_dtype = args.llm_kv_type
    if args.llm_decode_att_backend is not None:
        return backend_dict[llm_dtype][args.llm_decode_att_backend]
    else:
        # 根据环境自动选择最好的
        raise NotImplementedError(f"error")


def get_mla_prefill_att_backend_class() -> BaseAttBackend:
    # args = get_env_start_args()
    # llm_dtype = args.llm_kv_type
    # 根据环境自动选择最好的
    raise NotImplementedError(f"error")


def get_mla_decode_att_backend_class() -> BaseAttBackend:
    # args = get_env_start_args()
    # llm_dtype = args.llm_kv_type
    # 根据环境自动选择最好的
    raise NotImplementedError(f"error")
