from dataclasses import dataclass
from typing import Any, Iterable
from lightllm.platform.plugins.common import (
    ModulePluginConfig,
    load_entry_point_plugins,
    merge_config_field,
    parse_plugin_config,
    plugin_config_from_cli,
    plugin_names_from_cli,
    list_installed_plugin_names,
)

PLUGIN_KIND = "op"
ENTRY_POINT_GROUP = "lightllm.op_plugins"


@dataclass(frozen=True)
class OpsPluginConfig(ModulePluginConfig):
    extra_fallback: tuple[str, ...] = ()


_ops_plugin_config: OpsPluginConfig | None = None


def merge_ops_plugin_configs(configs: Iterable[OpsPluginConfig]) -> OpsPluginConfig:
    return OpsPluginConfig(
        extra_fallback=merge_config_field(configs, "extra_fallback"),
        extra_modules=merge_config_field(configs, "extra_modules"),
    ) 


def validate_ops_plugin_config(config: OpsPluginConfig) -> None:
    from lightllm.platform.base.ops.loader import has_builtin_ops_module

    if config.extra_modules and not config.extra_fallback:
        raise RuntimeError(
            "--extra_op_modules requires --extra_op_fallback: external modules must "
            "@register_op under impl family names listed in extra_op_fallback."
        )

    external_fallbacks = [
        family for family in config.extra_fallback if not has_builtin_ops_module(family)
    ]
    if not external_fallbacks:
        return
    
    if config.extra_modules:
        return
    
    hints: list[str] = [
        "External impl families need modules that call @register_op. "
        "Use --extra_op_modules <module> (scheme 1) or --extra_op_plugins <name> (scheme 2)."
    ]
    installed = list_installed_plugin_names(ENTRY_POINT_GROUP)
    if installed:
        hints.append(f"Installed op plugins: {installed}.")
    for family in external_fallbacks:
        if family.endswith("_plugin") or family == "example_plugin":
            hints.append(
                f"For family {family!r}, did you mean --extra_op_plugins example_op_plugin "
                f"instead of --extra_op_fallback {family}?"
            )
    raise RuntimeError(
        f"--extra_op_fallback includes external impl families {external_fallbacks} "
        f"without --extra_op_modules; no ops will be loaded for them. "
        + " ".join(hints)
    )


def configure_ops_plugins() -> OpsPluginConfig:
    global _ops_plugin_config

    plugin_names = plugin_names_from_cli(PLUGIN_KIND)

    direct_config = plugin_config_from_cli(
        OpsPluginConfig,
        plugin_kind=PLUGIN_KIND,
        fields=("extra_fallback", "extra_modules"),
    )

    has_plugins = bool(plugin_names)
    has_direct = bool(direct_config.extra_fallback or direct_config.extra_modules)
    if has_plugins and has_direct:
        raise RuntimeError(
            "Op plugin configuration is ambiguous: use either --extra_op_plugins or "
            "(--extra_op_fallback / --extra_op_modules), not both."
        )

    if has_plugins:
        _ops_plugin_config = merge_ops_plugin_configs(
            load_entry_point_plugins(
                plugin_names,
                entry_point_group=ENTRY_POINT_GROUP,
                parser=parse_ops_plugin_config,
                plugin_kind=PLUGIN_KIND,
            )
        )
    elif has_direct:
        _ops_plugin_config = direct_config
    else:
        _ops_plugin_config = OpsPluginConfig()

    validate_ops_plugin_config(_ops_plugin_config)

    return _ops_plugin_config


def parse_ops_plugin_config(value: Any) -> OpsPluginConfig:
    return parse_plugin_config(
        value,
        OpsPluginConfig,
        fields=("extra_fallback", "extra_modules"),
        plugin_kind=PLUGIN_KIND,
    )


def get_ops_plugin_config() -> OpsPluginConfig:
    if _ops_plugin_config is None:
        return OpsPluginConfig()
    return _ops_plugin_config



def resolve_ops_fallback(platform: str, plugin_config: OpsPluginConfig | None = None) -> tuple[str, ...]:
    from lightllm.platform.base.registry import get_platform_spec

    config = plugin_config or get_ops_plugin_config()
    merged: list[str] = []
    seen: set[str] = set()
    for family in config.extra_fallback + get_platform_spec(platform).op_fallback:
        if family in seen:
            continue
        seen.add(family)
        merged.append(family)

    return tuple(merged)
