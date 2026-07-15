from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Type

from lightllm.platform.base.graph import BackendGraph
from lightllm.platform.base.ops import build_ops
from lightllm.platform.base.runtime import BackendRuntime
from lightllm.platform.base.sampling import build_sampling

if TYPE_CHECKING:
    from lightllm.platform.base.ops.base import OpsProtocol
    from lightllm.platform.base.sampling.base import SamplingProtocol

PLATFORMS: dict[str, "PlatformSpec"] = {}

_platforms_loaded = False


@dataclass(frozen=True)
class PlatformSpec:
    name: str
    backend_cls: Type["Backend"]
    ops_fallback: tuple[str, ...]
    sampling_fallback: tuple[str, ...]


class Backend(ABC):
    platform_name: str
    _runtime: BackendRuntime
    _graph: BackendGraph
    _ops: "OpsProtocol"
    _sampling: "SamplingProtocol"


    def __init__(self, runtime: BackendRuntime, graph: BackendGraph) -> None:
        self._runtime = runtime
        self._graph = graph
        self._ops = build_ops(self.platform_name)
        self._sampling = build_sampling(self.platform_name)

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

    @property
    def sampling(self) -> "SamplingProtocol":
        return self._sampling


def register_platform(
    name: str,
    *,
    ops_fallback: tuple[str, ...],
    sampling_fallback: tuple[str, ...] | None = None,
):
    def decorator(backend_cls: Type[Backend]) -> Type[Backend]:
        if name in PLATFORMS:
            raise ValueError(f"Platform already registered: {name}")
        # Set platform name to the backend class
        backend_cls.platform_name = name
        # If sampling_fallback is not provided, use ops_fallback
        resolved_sampling_fallback = sampling_fallback or ops_fallback
        PLATFORMS[name] = PlatformSpec(
            name=name,
            backend_cls=backend_cls,
            ops_fallback=ops_fallback,
            sampling_fallback=resolved_sampling_fallback,
        )
        return backend_cls

    return decorator


def _ensure_platforms_registered() -> None:
    global _platforms_loaded
    if _platforms_loaded:
        return
    _platforms_loaded = True

    import lightllm.platform.backends  # noqa: F401 — registers all platforms

def get_platform_spec(platform: str) -> PlatformSpec:
    _ensure_platforms_registered()
    try:
        return PLATFORMS[platform]
    except KeyError as exc:
        raise RuntimeError(f"Platform is not configured: {platform}") from exc
