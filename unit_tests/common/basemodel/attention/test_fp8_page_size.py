from dataclasses import replace
from unit_tests.common.attention.helpers import make_spec
import pytest

from lightllm.common.attention.backends.dense.fa3 import fp8 as fa3_fp8
from lightllm.common.attention.backends.dense.fa3.backend import Fa3DenseDecodeBackend
from lightllm.common.attention.backends.dense.fa3.fp8 import Fa3DenseFp8DecodeBackend
from lightllm.common.attention.backends.dense.flashinfer import fp8 as flashinfer_fp8
from lightllm.common.attention.backends.dense.flashinfer.backend import FlashInferDenseDecodeBackend
from lightllm.common.attention.backends.dense.flashinfer.fp8 import FlashInferDenseFp8DecodeBackend
from lightllm.common.attention.backends.mla.fa3.backend import Fa3MLADecodeBackend
from lightllm.common.attention.backends.mla.flashinfer.backend import FlashInferMLADecodeBackend


@pytest.mark.parametrize(
    "backend_class",
    [Fa3DenseDecodeBackend, Fa3MLADecodeBackend, FlashInferDenseDecodeBackend, FlashInferMLADecodeBackend],
)
def test_default_kernel_page_size_matches_model_page_size(backend_class):
    backend = object.__new__(backend_class)
    backend.spec = make_spec(cache=replace(make_spec().cache, page_size=16))

    assert backend.kernel_page_size == 16


@pytest.mark.parametrize(
    ("backend_class", "backend_module"),
    [
        (Fa3DenseFp8DecodeBackend, fa3_fp8),
        (FlashInferDenseFp8DecodeBackend, flashinfer_fp8),
    ],
)
def test_fp8_attention_backend_uses_single_token_infer_pages(backend_class, backend_module):
    backend = object.__new__(backend_class)
    backend.spec = make_spec(cache=replace(make_spec().cache, page_size=16))

    assert backend.kernel_page_size == 1
