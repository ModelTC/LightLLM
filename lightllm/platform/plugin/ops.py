from dataclasses import fields, replace
from typing import Callable, TypeVar

from lightllm.platform.ops import PlatformOps

OpsGroupT = TypeVar("OpsGroupT")


OPS_PLUGINS: dict[tuple[str, str], dict[str, Callable]] = {}


def register_ops(name: str, platforms: tuple[str, ...] = ("cuda",), **ops: Callable) -> None:
    for platform in platforms:
        current = OPS_PLUGINS.setdefault((platform, name), {})
        duplicates = current.keys() & ops.keys()
        if duplicates:
            raise ValueError(
                f"Ops {sorted(duplicates)} are already registered for platform {platform!r}, group {name!r}."
            )
        current.update(ops)


def build_ops_group(name: str, platform: str, default: OpsGroupT) -> OpsGroupT:
    updates = OPS_PLUGINS.get((platform, name), {})
    if not updates:
        return default
    return replace(default, **updates)


def build_platform_ops(platform: str, default: PlatformOps) -> PlatformOps:
    ops = {
        field.name: build_ops_group(
            field.name,
            platform,
            getattr(default, field.name),
        )
        for field in fields(default)
    }
    return replace(default, **ops)
