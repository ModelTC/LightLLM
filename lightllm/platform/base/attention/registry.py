from dataclasses import dataclass
from typing import Literal, TypeVar

AttCategory = Literal["standard", "mla", "nsa"]

BackendT = TypeVar("BackendT", bound=type)


@dataclass(frozen=True)
class AttBackendSpec:
    # ["triton", "fa3", "flashinfer", "flashmla_spare", ...]
    name: str
    # ["standard", "mla", "nsa"]
    category: AttCategory
    # ["None", "int8kv", "int4kv", "fp8kv_sph", "fp8kv_spt", "fp8kv_dsa"]
    kv_type: str
    # xxxAttBackend
    backend_cls: type
    # ["cuda", "ascend", "musa", "maca", ...]
    platforms: tuple[str, ...] | None = None
    # ["triton", "fa3", "flashinfer", "flashmla_spare"]
    validate_name: str | None = None

    @property
    def key(self) -> tuple[AttCategory, str, str]:
        return (self.category, self.name, self.kv_type)

    def effective_validate_name(self) -> str:
        return self.validate_name if self.validate_name is not None else self.name

    def supports_platform(self, platform: str | None) -> bool:
        if self.platforms is None or platform is None:
            return True
        return platform in self.platforms


class AttBackendRegistry:

    def __init__(self) -> None:
        self._specs: dict[tuple[AttCategory, str, str], AttBackendSpec] = {}

    def register_spec(self, spec: AttBackendSpec, *, allow_override: bool = False) -> None:
        if spec.key in self._specs:
            if not allow_override:
                existing = self._specs[spec.key]
                raise ValueError(
                    f"Attention backend {spec.name!r} already registered for "
                    f"category={spec.category!r}, kv_type={spec.kv_type!r} "
                    f"as {existing.backend_cls.__module__}.{existing.backend_cls.__qualname__}"
                )
        self._specs[spec.key] = spec

    def register(
        self,
        name: str,
        backend_cls: type,
        *,
        category: AttCategory,
        kv_types: tuple[str, ...],
        platforms: tuple[str, ...] | None = None,
        validate_name: str | None = None,
        allow_override: bool = False,
    ) -> None:
        for kv_type in kv_types:
            self.register_spec(
                AttBackendSpec(
                    name=name,
                    category=category,
                    kv_type=kv_type,
                    backend_cls=backend_cls,
                    platforms=platforms,
                    validate_name=validate_name,
                ),
                allow_override=allow_override,
            )

    def get_spec(
        self,
        *,
        category: AttCategory,
        name: str,
        kv_type: str,
    ) -> AttBackendSpec | None:
        return self._specs.get((category, name, kv_type))

    def get_backend_cls(
        self,
        *,
        category: AttCategory,
        name: str,
        kv_type: str,
        platform: str | None = None,
    ) -> type | None:
        """ Get the backend class for the given category, name, kv_type and platform. """
        spec = self.get_spec(category=category, name=name, kv_type=kv_type)
        if spec is None:
            return None
        if not spec.supports_platform(platform):
            return None
        return spec.backend_cls

    def resolve_backend_cls(
        self,
        *,
        category: AttCategory,
        name: str,
        kv_type: str,
        platform: str | None = None,
    ) -> type:
        """ Resolve the backend class for the given category, name, kv_type and platform. """
        backend_cls = self.get_backend_cls(
            category=category,
            name=name,
            kv_type=kv_type,
            platform=platform,
        )
        if backend_cls is not None:
            return backend_cls

        available = self.list_names(category=category, kv_type=kv_type, platform=platform)
        message = (
            f"Attention backend {name!r} is not registered for "
            f"category={category!r}, kv_type={kv_type!r}"
        )
        if platform is not None:
            message += f", platform={platform!r}"
        if available:
            message += f". Registered names: {', '.join(available)}"
        else:
            message += ". No backends are registered for this category and kv_type"
        raise ValueError(message)

    def list_specs(
        self,
        *,
        category: AttCategory | None = None,
        kv_type: str | None = None,
        platform: str | None = None,
    ) -> tuple[AttBackendSpec, ...]:
        """ List the backend specs for the given category, kv_type and platform. """
        specs: list[AttBackendSpec] = []
        for spec in self._specs.values():
            if category is not None and spec.category != category:
                continue
            if kv_type is not None and spec.kv_type != kv_type:
                continue
            if not spec.supports_platform(platform):
                continue
            specs.append(spec)
        return tuple(sorted(specs, key=lambda item: (item.category, item.name, item.kv_type)))

    def list_names(
        self,
        *,
        category: AttCategory,
        kv_type: str,
        platform: str | None = None,
    ) -> tuple[str, ...]:
        """ List the backend names for the given category, kv_type and platform. """
        names: list[str] = []
        seen: set[str] = set()
        for spec in self.list_specs(category=category, kv_type=kv_type, platform=platform):
            if spec.name in seen:
                continue
            seen.add(spec.name)
            names.append(spec.name)
        return tuple(names)

    def is_registered(
        self,
        *,
        category: AttCategory,
        name: str,
        kv_type: str,
        platform: str | None = None,
    ) -> bool:
        """ Check if the backend is registered for the given category, name, kv_type and platform. """
        return self.get_backend_cls(
            category=category,
            name=name,
            kv_type=kv_type,
            platform=platform,
        ) is not None


att_backend_registry = AttBackendRegistry()


def register_att_backend(
    name: str,
    *,
    category: AttCategory,
    kv_types: tuple[str, ...] = ("None",),
    platforms: tuple[str, ...] | None = None,
    validate_name: str | None = None,
    allow_override: bool = False,
):

    def decorator(backend_cls: BackendT) -> BackendT:
        att_backend_registry.register(
            name,
            backend_cls,
            category=category,
            kv_types=kv_types,
            platforms=platforms,
            validate_name=validate_name,
            allow_override=allow_override,
        )
        return backend_cls

    return decorator
