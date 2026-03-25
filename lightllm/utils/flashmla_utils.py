import sys
from importlib import import_module
from pathlib import Path


def _candidate_roots() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    return [
        repo_root / "FlashMLA",
        repo_root.parent / "FlashMLA",
    ]


def import_flash_mla():
    try:
        return import_module("flash_mla")
    except ModuleNotFoundError:
        pass

    for root in _candidate_roots():
        if root.exists():
            root_str = str(root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
            try:
                return import_module("flash_mla")
            except ModuleNotFoundError:
                continue

    raise ModuleNotFoundError(
        "flash_mla is not installed and no local FlashMLA checkout was found. "
        "Install FlashMLA or place the repository at ./FlashMLA."
    )
