import importlib.util
from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Type

from lightllm.platform.base.graph import BackendGraph
from lightllm.platform.base.runtime import BackendRuntime

if TYPE_CHECKING:
    from lightllm.platform.base.ops.base import OpsProtocol

# For internal module registration, we use the prefix "lightllm.platform.ops."
OP_FAMILY_MODULES_PREFIX = "lightllm.platform.ops."

PLATFORMS: dict[str, "PlatformSpec"] = {}

_platforms_loaded = False


@dataclass(frozen=True)
class PlatformSpec:
    name: str
    backend_cls: Type["Backend"]
    op_fallback: tuple[str, ...]


class Backend(ABC):
    platform_name: str
    _runtime: BackendRuntime
    _graph: BackendGraph
    _ops: "OpsProtocol"

    @property
    def name(self) -> str:
       return self.platform_name

    @property
    def runtime(self) -> BackendRuntime:
        return self._runtime 

    @property
    def graph(self) -> BackendGraph:
        return self._graph

    @property
    def ops(self) -> "OpsProtocol":
        return self._ops


def register_platform(name: str, *, op_fallback: tuple[str, ...]):
    def decorator(backend_cls: Type[Backend]) -> Type[Backend]:
        if name in PLATFORMS:
            raise ValueError(f"Platform already registered: {name}")
        # set platform name to the backend class
        backend_cls.platform_name = name
        PLATFORMS[name] = PlatformSpec(
            name=name,
            backend_cls=backend_cls,
            op_fallback=op_fallback,
        )
        return backend_cls

    return decorator


def _ensure_platforms_registered() -> None:
    global _platforms_loaded
    if _platforms_loaded:
        return
    _platforms_loaded = True

    import importlib
    import pkgutil

    import lightllm.platform.backends as backends_pkg

    for module_info in pkgutil.iter_modules(backends_pkg.__path__):
        if module_info.name.startswith("_"):
            continue
        # To import the backend module to avoid adding it to __init__.py manually
        importlib.import_module(f"{backends_pkg.__name__}.{module_info.name}")


def get_platform_spec(platform: str) -> PlatformSpec:
    _ensure_platforms_registered()
    try:
        return PLATFORMS[platform]
    except KeyError as exc:
        raise RuntimeError(f"Platform is not configured: {platform}") from exc


def has_builtin_ops_module(family: str) -> bool:
    module_name = f"{OP_FAMILY_MODULES_PREFIX}{family}"
    return importlib.util.find_spec(module_name) is not None


def get_op_modules_for_fallback(fallback_chain: tuple[str, ...]) -> tuple[str, ...]:
    modules: list[str] = []
    seen: set[str] = set()
    for family in fallback_chain:
        module_name = f"{OP_FAMILY_MODULES_PREFIX}{family}"
        if module_name in seen:
            continue
        # External modules are not required to be imported by lightllm.platform.ops
        if not has_builtin_ops_module(family):
            continue

        modules.append(module_name)
        seen.add(module_name)

    return tuple(modules)
