#!/usr/bin/env python3
"""Install the LightLLM wheel matching the local NVIDIA driver.

The release manifest is intentionally separate from this installer. A release
can therefore publish CUDA profiles without requiring a new installer release.
The manifest entries use local paths or URLs for the LightLLM wheel and its
CUDA profile requirements.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST_URL = "https://raw.githubusercontent.com/sufubao/LightLLM/main/installer/manifest.json"
DRIVER_RE = re.compile(r"(?:Driver Version|driver_version)\s*[:=,]?\s*([0-9]+(?:\.[0-9]+)+)", re.I)


class InstallerError(RuntimeError):
    pass


@dataclass(frozen=True, order=True)
class Version:
    parts: tuple[int, ...]

    @classmethod
    def parse(cls, value: str) -> "Version":
        try:
            return cls(tuple(int(part) for part in value.strip().split(".")))
        except ValueError as exc:
            raise InstallerError(f"invalid version in installer manifest: {value!r}") from exc


@dataclass(frozen=True)
class Hardware:
    system: str
    machine: str
    driver: Version


@dataclass(frozen=True)
class Profile:
    name: str
    minimum_driver: Version
    maximum_driver: Version | None
    wheel: str
    requirements: str
    python: str | None = None


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as exc:
        raise InstallerError(f"cannot execute {' '.join(command)}: {exc}") from exc


def detect_hardware() -> Hardware:
    machine = platform.machine().lower()
    if machine not in {"x86_64", "amd64"}:
        raise InstallerError(f"unsupported CPU architecture {machine!r}; published wheels target Linux x86_64")
    if platform.system() != "Linux":
        raise InstallerError("native LightLLM wheels currently require Linux")

    result = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"])
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "nvidia-smi returned an error"
        raise InstallerError(
            "NVIDIA driver was not detected. Install a supported NVIDIA driver, "
            f"or use the Docker delivery instead. Details: {detail}"
        )
    versions = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    if not versions:
        match = DRIVER_RE.search(result.stdout + "\n" + result.stderr)
        if match:
            versions.add(match.group(1))
    if not versions:
        raise InstallerError("nvidia-smi did not report a driver version")
    if len(versions) != 1:
        raise InstallerError(f"GPUs report different driver versions: {', '.join(sorted(versions))}")
    return Hardware(platform.system(), machine, Version.parse(next(iter(versions))))


def _read_manifest(source: str) -> tuple[dict[str, Any], str]:
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme in {"http", "https"}:
        try:
            with urllib.request.urlopen(source, timeout=30) as response:
                payload = response.read().decode()
            return json.loads(payload), source
        except (OSError, ValueError) as exc:
            raise InstallerError(f"failed to download installer manifest {source}: {exc}") from exc

    path = Path(source).expanduser().resolve()
    try:
        return json.loads(path.read_text()), path.as_uri()
    except (OSError, ValueError) as exc:
        raise InstallerError(f"failed to read installer manifest {path}: {exc}") from exc


def _resolve_reference(reference: str, base: str) -> str:
    parsed = urllib.parse.urlparse(reference)
    if parsed.scheme in {"http", "https", "file"} or Path(reference).is_absolute():
        return reference
    if base.startswith(("http://", "https://")):
        return urllib.parse.urljoin(base.rsplit("/", 1)[0] + "/", reference)
    base_path = Path(urllib.request.url2pathname(urllib.parse.urlparse(base).path)).parent
    return str((base_path / reference).resolve())


def load_profiles(manifest: dict[str, Any], source: str) -> list[Profile]:
    entries = manifest.get("profiles")
    if not isinstance(entries, list) or not entries:
        raise InstallerError("installer manifest has no profiles")
    profiles = []
    for entry in entries:
        try:
            profiles.append(
                Profile(
                    name=str(entry["name"]),
                    minimum_driver=Version.parse(str(entry["min_driver"])),
                    maximum_driver=Version.parse(str(entry["max_driver"])) if entry.get("max_driver") else None,
                    wheel=_resolve_reference(str(entry["wheel"]), source),
                    requirements=_resolve_reference(str(entry["requirements"]), source),
                    python=str(entry["python"]) if entry.get("python") else None,
                )
            )
        except (KeyError, TypeError) as exc:
            raise InstallerError(f"invalid profile in installer manifest: {entry!r}") from exc
    return profiles


def select_profile(profiles: list[Profile], driver: Version, forced: str | None = None) -> Profile:
    if forced:
        for profile in profiles:
            if profile.name == forced:
                if driver < profile.minimum_driver or (
                    profile.maximum_driver is not None and driver > profile.maximum_driver
                ):
                    maximum = (
                        f" and <= {'.'.join(map(str, profile.maximum_driver.parts))}" if profile.maximum_driver else ""
                    )
                    raise InstallerError(
                        f"profile {forced!r} does not support driver "
                        f"{'.'.join(map(str, driver.parts))}; requires >= "
                        f"{'.'.join(map(str, profile.minimum_driver.parts))}{maximum}"
                    )
                return profile
        raise InstallerError(f"requested profile {forced!r} is not present in the installer manifest")

    compatible = [
        profile
        for profile in profiles
        if driver >= profile.minimum_driver and (profile.maximum_driver is None or driver <= profile.maximum_driver)
    ]
    if not compatible:
        supported = ", ".join(f"{p.name} (driver >= {'.'.join(map(str, p.minimum_driver.parts))})" for p in profiles)
        raise InstallerError(
            f"driver {'.'.join(map(str, driver.parts))} has no compatible wheel profile. Available: {supported}"
        )
    return max(compatible, key=lambda profile: profile.minimum_driver)


def _target_python(venv: Path, requested: str | None) -> str:
    if requested:
        return requested
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return str(python)


def install(profile: Profile, venv: Path, python: str | None, dry_run: bool = False) -> list[list[str]]:
    uv = shutil.which("uv")
    if not uv:
        raise InstallerError("uv is required. Install it from https://docs.astral.sh/uv/")
    # ``--python`` is a Python version used to create the venv. The package is
    # always installed into that venv, so the pip target is stable and does not
    # accidentally point at the installer process' interpreter.
    target_python = _target_python(venv, None)
    commands: list[list[str]] = []
    if not venv.exists() and python is None:
        commands.append([uv, "venv", "--python", profile.python or sys.executable, str(venv)])
    elif not venv.exists():
        commands.append([uv, "venv", "--python", python, str(venv)])
    commands.append([uv, "pip", "install", "--python", target_python, "-r", profile.requirements, profile.wheel])
    if dry_run:
        return commands
    for command in commands:
        subprocess.run(command, check=True)
    return commands


def _default_manifest() -> str:
    artifact = Path(__file__).resolve().with_name("installer-manifest.json")
    if artifact.is_file():
        return str(artifact)
    local = Path(__file__).resolve().parents[1] / "installer" / "manifest.json"
    return str(local) if local.is_file() else os.environ.get("LIGHTLLM_INSTALL_MANIFEST", DEFAULT_MANIFEST_URL)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install LightLLM using the wheel matching the NVIDIA driver.")
    parser.add_argument("--manifest", default=_default_manifest(), help="local path or URL of the release manifest")
    parser.add_argument("--profile", help="force a profile instead of automatic driver selection")
    parser.add_argument("--venv", type=Path, default=Path(".venv"), help="target uv virtual environment")
    parser.add_argument("--python", dest="python_executable", help="Python version used to create the target venv")
    parser.add_argument("--dry-run", action="store_true", help="print commands without changing the environment")
    args = parser.parse_args(argv)

    try:
        hardware = detect_hardware()
        manifest, source = _read_manifest(args.manifest)
        profiles = load_profiles(manifest, source)
        profile = select_profile(profiles, hardware.driver, args.profile)
        if profile.python and not sys.version.startswith(profile.python) and args.python_executable is None:
            print(f"uv will create the target environment with Python {profile.python} for profile {profile.name}")
        python_version = args.python_executable or profile.python
        print(
            f"Detected {hardware.machine}, NVIDIA driver {'.'.join(map(str, hardware.driver.parts))}; "
            f"selected profile {profile.name}"
        )
        commands = install(profile, args.venv, python_version, args.dry_run)
        for command in commands:
            print("$ " + " ".join(command))
        if not args.dry_run:
            print(f"Installed LightLLM into {args.venv}; run {args.venv}/bin/lightllm serve /models/my-model")
        return 0
    except (InstallerError, subprocess.CalledProcessError) as exc:
        print(f"lightllm installer: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
