import pytest

from lightllm.common.basemodel.attention.fa3.fp8 import Fp8Fa3AttBackend
from lightllm.common.basemodel.attention.flashinfer.fp8 import Fp8FlashInferAttBackend


@pytest.mark.parametrize(
    ("backend_class", "error_message"),
    [
        (Fp8Fa3AttBackend, "Fp8Fa3AttBackend only supports page_size == 1"),
        (Fp8FlashInferAttBackend, "Fp8FlashInferAttBackend only supports page_size == 1"),
    ],
)
def test_fp8_attention_backend_rejects_multi_token_pages(backend_class, error_message):
    class FakeModel:
        class args:
            page_size = 2

    model = FakeModel()

    with pytest.raises(AssertionError, match=error_message):
        backend_class(model=model)
