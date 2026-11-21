import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("tomllib")

spec = importlib.util.spec_from_file_location("update_nightly", Path(__file__).parents[1] / "tools/update_nightly.py")
nightly = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nightly)


def test_lower_priority_package_cannot_downgrade_frozen_core():
    calls = []

    def compile_requirements(requirements):
        calls.append(requirements)
        if requirements == ["torch"]:
            return ["torch==2.16.0.dev1", "triton==3.9.0"]
        assert requirements == ["torch==2.16.0.dev1", "kernel"]
        return ["kernel==0.4.6", "torch==2.16.0.dev1", "triton==3.9.0"]

    pins, resolved = nightly.resolve_priority(["torch", "kernel"], compile_requirements)
    assert pins == ["torch==2.16.0.dev1", "kernel==0.4.6"]
    assert "triton==3.9.0" in resolved
    assert len(calls) == 2


def test_wheel_reference_selects_target_python_and_preserves_hash():
    package = {
        "name": "torch",
        "wheels": [
            {
                "url": "https://example.com/torch-2.16.0-cp313-cp313-manylinux_2_28_x86_64.whl",
                "hashes": {"sha256": "wrong"},
            },
            {
                "url": "https://example.com/torch-2.16.0-cp312-cp312-manylinux_2_28_x86_64.whl",
                "hashes": {"sha256": "abc"},
            },
        ],
    }
    assert nightly.wheel_reference(package).endswith("cp312-cp312-manylinux_2_28_x86_64.whl#sha256=abc")
    assert nightly.wheel_reference({"name": "torch", "archive": package["wheels"][1]}) == nightly.wheel_reference(
        package
    )
    package["wheels"].pop()
    with pytest.raises(ValueError, match="No Python 3.12"):
        nightly.wheel_reference(package)
