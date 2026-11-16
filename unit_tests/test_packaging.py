"""Check deployment contracts that source-tree imports cannot exercise."""

from pathlib import Path
import runpy

import setuptools


def test_nvfp4_attention_kernels_are_packaged(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    setup_options = {}
    monkeypatch.chdir(root)
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: setup_options.update(kwargs))
    runpy.run_path(str(root / "setup.py"), run_name="__main__")

    distribution = setuptools.Distribution(setup_options)
    distribution.script_name = str(root / "setup.py")
    builder = distribution.get_command_obj("build_py")
    builder.ensure_finalized()
    modules = {f"{package}.{module}" for package, module, _ in builder.find_all_modules()}

    required = {
        "lightllm.common.attention.kernels.gqa.nvfp4.prefill",
        "lightllm.common.attention.kernels.gqa.nvfp4.decode",
        "lightllm.common.attention.kernels.mla.nvfp4",
        "lightllm.common.attention.kernels.mla.packed_sparse",
    }
    assert required <= modules, f"NVFP4 kernels missing from distribution: {sorted(required - modules)}"
