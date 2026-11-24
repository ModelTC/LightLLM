import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("lightllm_installer", Path(__file__).parents[1] / "tools/install.py")
installer = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = installer
_SPEC.loader.exec_module(installer)


def test_select_profile_uses_highest_compatible_minimum_driver():
    manifest, source = installer._read_manifest(str(Path(__file__).parents[1] / "installer/manifest.json"))
    profiles = installer.load_profiles(manifest, source)
    assert installer.select_profile(profiles, installer.Version.parse("570.124.06")).name == "cuda128"
    assert installer.select_profile(profiles, installer.Version.parse("580.65.01")).name == "cuda13"
    assert installer.select_profile(profiles, installer.Version.parse("580.65.01"), "cuda128").name == "cuda128"
    with pytest.raises(installer.InstallerError, match="no compatible wheel profile"):
        installer.select_profile(profiles, installer.Version.parse("570.86.10"))


def test_installer_resolves_profile_and_wheel_together(monkeypatch, tmp_path):
    monkeypatch.setattr(installer.shutil, "which", lambda _: "/usr/bin/uv")
    profile = installer.Profile("cuda128", installer.Version.parse("570.124.06"), None, "a.whl", "cuda128.txt", "3.12")
    venv = tmp_path / "venv"
    assert installer.install(profile, venv, None, dry_run=True) == [
        ["/usr/bin/uv", "venv", "--python", "3.12", str(venv)],
        ["/usr/bin/uv", "pip", "install", "--python", str(venv / "bin/python"), "-r", "cuda128.txt", "a.whl"],
    ]


def test_manifest_resolves_relative_artifacts(tmp_path):
    artifact_dir = tmp_path / "用户下载 space"
    artifact_dir.mkdir()
    wheel_path = artifact_dir / "dist" / "a.whl"
    wheel_path.parent.mkdir()
    wheel_path.touch()
    requirements_path = artifact_dir / "req.txt"
    requirements_path.touch()
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"profiles": [{"name": "test", "min_driver": "570.0", "wheel": "dist/a.whl", "requirements": "req.txt"}]}
        )
    )
    manifest, source = installer._read_manifest(str(manifest_path))
    profile = installer.load_profiles(manifest, source)[0]
    assert Path(profile.wheel).samefile(wheel_path)
    assert Path(profile.requirements).samefile(requirements_path)
