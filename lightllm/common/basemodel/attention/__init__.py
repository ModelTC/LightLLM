from .base_att import BaseAttBackend, BasePrefillAttState, BaseDecodeAttState, AttControl
from .triton.fp import TritonAttBackend
from .triton.int4kv import Int4kvTritonAttBackend
from .triton.int8kv import Int8kvTritonAttBackend
from .triton.mla import MlaTritonAttBackend
from .fa3.fp import Fa3AttBackend
from .fa3.fp8 import Fp8Fa3AttBackend
from .fa3.mla import MlaFa3AttBackend
from .paged_fa3.fp import PagedFa3AttBackend
from .paged_fa3.mla import PagedMlaFa3AttBackend
from .flashinfer.fp8 import Fp8FlashInferAttBackend
from .flashinfer.fp import FlashInferAttBackend
from .flashinfer.mla import MlaFlashInferAttBackend
from .paged_flashinfer.fp import PagedFlashInferAttBackend
from .paged_flashinfer.mla import PagedMlaFlashInferAttBackend
from lightllm.utils.envs_utils import get_page_size

if get_page_size() > 1:
    from .paged_create_utils import (
        get_prefill_att_backend_class,
        get_decode_att_backend_class,
        get_mla_prefill_att_backend_class,
        get_mla_decode_att_backend_class,
    )
else:
    from .create_utils import (
        get_prefill_att_backend_class,
        get_decode_att_backend_class,
        get_mla_prefill_att_backend_class,
        get_mla_decode_att_backend_class,
    )
