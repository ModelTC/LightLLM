import importlib.util
import json
import sys
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location("lightllm_installer", Path(__file__).parents[1] / "tools/install.py")
installer = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = installer
_SPEC.loader.exec_module(installer)


def test_select_profile_uses_highest_compatible_minimum_driver():
    profiles = [
        installer.Profile("cuda128", installer.Version.parse("570.0"), None, "a.whl", "a.txt"),
        installer.Profile("cuda13", installer.Version.parse("580.0"), None, "b.whl", "b.txt"),
    ]
    assert installer.select_profile(profiles, installer.Version.parse("570.124.06")).name == "cuda128"
    assert installer.select_profile(profiles, installer.Version.parse("580.65.01")).name == "cuda13"


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
