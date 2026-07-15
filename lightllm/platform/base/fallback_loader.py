import importlib
import importlib.util
from dataclasses import dataclass
from typing import Any, Callable
from lightllm.platform.plugins.common import Plugin


def _builtin_family_modules(fallback_chain: tuple[str, ...], module_prefix: str) -> tuple[str, ...]:
    modules: list[str] = []
    seen: set[str] = set()
    for family in fallback_chain:
        module_name = f"{module_prefix}{family}"
        if module_name in seen:
            continue
        if importlib.util.find_spec(module_name) is None:
            continue
        modules.append(module_name)
        seen.add(module_name)
    return tuple(modules)


def _import_implementations(
    modules: tuple[str, ...],
    fallback_chain: tuple[str, ...],
    module_prefix: str,
) -> None:
    for module_name in modules:
        importlib.import_module(module_name)
    for module_name in _builtin_family_modules(fallback_chain, module_prefix):
        importlib.import_module(module_name)


@dataclass(frozen=True)
class FallbackLoaderSpec:
    module_prefix: str
    op_names: tuple[str, ...]


class OpEndPoint:

    def __init__(
        self,
        platform: str,
        *,
        op_names: tuple[str, ...],
        fallback_chain: tuple[str, ...],
        resolve_impl: Callable,
        label: str,
    ) -> None:
        self.platform = platform
        self.label = label
        # 给当前对象挂上每个 op 对应的实现函数，一般来说是从左往右依次尝试，sampling 还会尝试 sampling_backend
        for op_name in op_names:
            setattr(self, op_name, resolve_impl(op_name, fallback_chain))

    def __getattr__(self, name: str) -> object:
        # 如果当前对象没有对应的 op，则抛出 AttributeError
        raise AttributeError(
            f"{self.label} '{name}' is not registered for platform '{self.platform}'"
        )


def resolve_fallback_chain(
    platform: str,
    fallback: tuple[str, ...],
    platform_fallback_field: str,
) -> tuple[str, ...]:
    from lightllm.platform.base.registry import get_platform_spec

    platform_fallback = getattr(get_platform_spec(platform), platform_fallback_field)
    merged: list[str] = []
    seen: set[str] = set()
    # 按照自定义 fallback + 内置默认 fallback 的顺序，去重后返回
    for family in fallback + platform_fallback:
        if family in seen:
            continue
        seen.add(family)
        merged.append(family)
    return tuple(merged)


def make_fallback_loader(
    *,
    plugin: Plugin,
    spec: FallbackLoaderSpec,
    registry: Any,
    resolve_impl: Callable,
) -> Callable:
    
    def build(platform: str) -> OpEndPoint:
        field = plugin.platform_fallback_field
        if field is None:
            raise TypeError(f"Plugin {plugin.name} doesn't support fallback loading")

        plugin_config = plugin.get()

        # 1. 获取 fallback 链，包括自定义和内置默认
        fallback_chain = resolve_fallback_chain(platform, plugin_config.fallback, field)

        # 2. import 自定义和内置模块，触发 @register_xx 操作
        _import_implementations(plugin_config.modules, fallback_chain, spec.module_prefix)

        # 3. 验证自定义 fallback 是否都有实现
        missing = [
            family for family in plugin_config.fallback if not registry.has_impl_family(family)
        ]
        if missing:
            raise RuntimeError(
                f"{plugin.name}: {plugin.cli_flag('fallback')} families not registered "
                f"by {plugin.register_decorator}: {missing}. "
                f"Check {plugin.cli_flag('modules')} / {plugin.cli_flag('plugins')} "
                f"and that {plugin.register_decorator}(impl_family=...) matches."
            )

        # 4. 返回
        return OpEndPoint(
            platform=platform,
            op_names=spec.op_names,
            fallback_chain=fallback_chain,
            resolve_impl=resolve_impl,
            label=plugin.name,
        )

    return build
