import os
from email.parser import Parser
from pathlib import Path
from zipfile import ZipFile

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet


@pytest.mark.skipif(not os.environ.get("LIGHTLLM_TEST_WHEEL"), reason="requires a built wheel")
def test_wheel_contains_runtime_modules_and_kernel_configs():
    root = Path(__file__).resolve().parents[1]
    with ZipFile(os.environ["LIGHTLLM_TEST_WHEEL"]) as wheel:
        metadata_path = next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA"))
        metadata = Parser().parsestr(wheel.read(metadata_path).decode())
        assert metadata["Version"] == (root / "VERSION").read_text().strip()
        nightly = ".dev" in metadata["Version"]
        requirements_file = root / "requirements" / ("nightly.txt" if nightly else "runtime.txt")
        expected = {
            str(Requirement(line))
            for line in requirements_file.read_text().splitlines()
            if line and not line.startswith("#")
        }
        assert {str(Requirement(line)) for line in metadata.get_all("Requires-Dist")} == expected
        assert SpecifierSet(metadata["Requires-Python"]) == SpecifierSet(">=3.12,<3.13" if nightly else ">=3.10,<3.14")
        for path in (root / "lightllm").rglob("*"):
            if path.is_file() and path.suffix in (".py", ".json"):
                name = path.relative_to(root).as_posix()
                assert name in wheel.namelist(), f"Missing runtime file: {name}"
                assert wheel.read(name) == path.read_bytes(), f"Stale runtime file: {name}"
