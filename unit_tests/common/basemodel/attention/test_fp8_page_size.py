from types import SimpleNamespace
from unittest.mock import patch

import pytest

from lightllm.common.basemodel.attention.fa3 import fp8 as fa3_fp8
from lightllm.common.basemodel.attention.fa3.fp import Fa3AttBackend
from lightllm.common.basemodel.attention.fa3.fp8 import Fp8Fa3AttBackend
from lightllm.common.basemodel.attention.fa3.mla import MlaFa3AttBackend
from lightllm.common.basemodel.attention.flashinfer import fp8 as flashinfer_fp8
from lightllm.common.basemodel.attention.flashinfer.fp import FlashInferAttBackend
from lightllm.common.basemodel.attention.flashinfer.fp8 import Fp8FlashInferAttBackend
from lightllm.common.basemodel.attention.flashinfer.mla import MlaFlashInferAttBackend


@pytest.mark.parametrize(
    "backend_class",
    [Fa3AttBackend, MlaFa3AttBackend, FlashInferAttBackend, MlaFlashInferAttBackend],
)
def test_default_infer_page_size_matches_model_page_size(backend_class):
    backend = object.__new__(backend_class)
    backend.model = SimpleNamespace(args=SimpleNamespace(page_size=16))

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
    backend.model = SimpleNamespace(args=SimpleNamespace(page_size=16))

    with patch.object(backend_module.logger, "warning") as warning:
        backend._init_infer_page_size()

    assert backend.infer_page_size == 1
    warning.assert_called_once()
    assert "infer_page_size=1" in warning.call_args.args[0]
