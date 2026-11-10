from types import SimpleNamespace
from unittest.mock import patch

import pytest

from lightllm.common.attention.backends.dense.fa3 import fp8 as fa3_fp8
from lightllm.common.attention.backends.dense.fa3.backend import Fa3AttBackend
from lightllm.common.attention.backends.dense.fa3.fp8 import Fp8Fa3AttBackend
from lightllm.common.attention.backends.mla.fa3.backend import MlaFa3AttBackend
from lightllm.common.attention.backends.dense.flashinfer import fp8 as flashinfer_fp8
from lightllm.common.attention.backends.dense.flashinfer.backend import FlashInferAttBackend
from lightllm.common.attention.backends.dense.flashinfer.fp8 import Fp8FlashInferAttBackend
from lightllm.common.attention.backends.mla.flashinfer.backend import MlaFlashInferAttBackend


@pytest.mark.parametrize(
    "backend_class",
    [Fa3AttBackend, MlaFa3AttBackend, FlashInferAttBackend, MlaFlashInferAttBackend],
)
def test_default_infer_page_size_matches_model_page_size(backend_class):
    backend = object.__new__(backend_class)
    backend.page_size = 16

    backend._init_infer_page_size()

    assert backend.infer_page_size == 16


@pytest.mark.parametrize(
    ("backend_class", "backend_module"),
    [
        (Fp8Fa3AttBackend, fa3_fp8),
        (Fp8FlashInferAttBackend, flashinfer_fp8),
    ],
)
def test_fp8_attention_backend_uses_single_token_infer_pages(backend_class, backend_module):
    backend = object.__new__(backend_class)
    backend.page_size = 16

    with patch.object(backend_module.logger, "warning") as warning:
        backend._init_infer_page_size()

    assert backend.infer_page_size == 1
    warning.assert_called_once()
    assert "infer_page_size=1" in warning.call_args.args[0]
