import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "lightllm.common.execution.compute.attention.backend",
        "lightllm.core.compute.attention.backend",
    ],
)
def test_attention_backend_import_has_no_contract_cycle(module_name):
    module = importlib.import_module(module_name)

    assert module.AttentionSpec is not None
    assert module.AttentionParams is not None


@pytest.mark.parametrize(
    "package_name",
    ["lightllm.common.execution.contracts", "lightllm.core.contracts"],
)
def test_contract_exports_remain_lazy_and_public(package_name):
    package = importlib.import_module(package_name)

    assert package.ModelAdapter is not None
    assert package.StateSpec is not None
    assert package.DeviceCaps is not None
