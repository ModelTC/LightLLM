import importlib
import importlib.util
from dataclasses import dataclass
from typing import Callable, Protocol

from lightllm.platform.plugins.common import FallbackPluginConfig, PluginKind
from lightllm.platform.plugins.registration_validation import validate_fallback_families_registered

ResolveImpl = Callable[[str, tuple[str, ...], str], Callable]
BuildImpl = Callable[[str], object]


class ImplFamilyRegistry(Protocol):
    def has_impl_family(self, impl_family: str) -> bool: ...


@dataclass(frozen=True)
class FallbackLoaderSpec:
    module_prefix: str
    op_names: tuple[str, ...]
    register_decorator: str
    platform_fallback_field: str
    view_label: str
    silent_fallback_entity: str


def make_fallback_loader(
    *,
    plugin: PluginKind[FallbackPluginConfig],
    spec: FallbackLoaderSpec,
    registry: ImplFamilyRegistry,
    resolve_impl: ResolveImpl,
    view_protocol: type | None = None,
) -> BuildImpl:
    module_prefix = spec.module_prefix
    view_bases = (view_protocol,) if view_protocol is not None else (object,)

    def has_builtin_module(family: str) -> bool:
        return importlib.util.find_spec(f"{module_prefix}{family}") is not None

    def family_modules_for_fallback(fallback_chain: tuple[str, ...]) -> tuple[str, ...]:
        modules: list[str] = []
        seen: set[str] = set()
        for family in fallback_chain:
            module_name = f"{module_prefix}{family}"
            if module_name in seen:
                continue
            if not has_builtin_module(family):
                continue
            modules.append(module_name)
            seen.add(module_name)
        return tuple(modules)

    def load(fallback_chain: tuple[str, ...], extra_modules: tuple[str, ...] = ()) -> None:
        for module_name in extra_modules:
            importlib.import_module(module_name)
        for module_name in family_modules_for_fallback(fallback_chain):
            importlib.import_module(module_name)

    class FallbackView(*view_bases):

        def __init__(self, platform: str, *, fallback_chain: tuple[str, ...]) -> None:
            self._platform = platform
            for op_name in spec.op_names:
                setattr(self, op_name, resolve_impl(op_name, fallback_chain, platform))

        def __getattr__(self, name: str) -> object:
            raise AttributeError(
                f"{spec.view_label} '{name}' is not registered for platform '{self._platform}'"
            )

    def build(platform: str) -> FallbackView:
        plugin_config = plugin.get()
        fallback_chain = plugin.resolve_fallback_for(platform, plugin_config)
        load(fallback_chain, plugin_config.extra_modules)
        validate_fallback_families_registered(
            plugin=plugin,
            plugin_config=plugin_config,
            register_decorator=spec.register_decorator,
            platform_fallback_field=spec.platform_fallback_field,
            silent_fallback_entity=spec.silent_fallback_entity,
            registry_has_impl_family=registry.has_impl_family,
            platform=platform,
        )
        return FallbackView(platform, fallback_chain=fallback_chain)

    return build
