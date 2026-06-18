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

PLUGIN_KIND = "sampling"
ENTRY_POINT_GROUP = "lightllm.sampling_plugins"


@dataclass(frozen=True)
class SamplingPluginConfig(ModulePluginConfig):
    extra_fallback: tuple[str, ...] = ()


_sampling_plugin_config: SamplingPluginConfig | None = None


def merge_sampling_plugin_configs(
    configs: Iterable[SamplingPluginConfig],
) -> SamplingPluginConfig:
    return SamplingPluginConfig(
        extra_fallback=merge_config_field(configs, "extra_fallback"),
        extra_modules=merge_config_field(configs, "extra_modules"),
    )


def validate_sampling_plugin_config(config: SamplingPluginConfig) -> None:
    from lightllm.platform.base.sampling.loader import has_builtin_sampling_module

    if config.extra_modules and not config.extra_fallback:
        raise RuntimeError(
            "--extra_sampling_modules requires --extra_sampling_fallback: external modules must "
            "@register_sampling_op under impl family names listed in extra_sampling_fallback."
        )

    external_fallbacks = [
        family for family in config.extra_fallback if not has_builtin_sampling_module(family)
    ]
    if not external_fallbacks:
        return

    if config.extra_modules:
        return

    hints: list[str] = [
        "External impl families need modules that call @register_sampling_op. "
        "Use --extra_sampling_modules <module> (scheme 1) or "
        "--extra_sampling_plugins <name> (scheme 2)."
    ]
    installed = list_installed_plugin_names(ENTRY_POINT_GROUP)
    if installed:
        hints.append(f"Installed sampling plugins: {installed}.")
    for family in external_fallbacks:
        if family.endswith("_plugin") or family == "example_plugin":
            hints.append(
                f"For family {family!r}, did you mean --extra_sampling_plugins example_sampling_plugin "
                f"instead of --extra_sampling_fallback {family}?"
            )
    raise RuntimeError(
        f"--extra_sampling_fallback includes external impl families {external_fallbacks} "
        f"without --extra_sampling_modules; no sampling ops will be loaded for them. "
        + " ".join(hints)
    )


def configure_sampling_plugins() -> SamplingPluginConfig:
    global _sampling_plugin_config

    plugin_names = plugin_names_from_cli(PLUGIN_KIND)

    direct_config = plugin_config_from_cli(
        SamplingPluginConfig,
        plugin_kind=PLUGIN_KIND,
        fields=("extra_fallback", "extra_modules"),
    )

    has_plugins = bool(plugin_names)
    has_direct = bool(direct_config.extra_fallback or direct_config.extra_modules)
    if has_plugins and has_direct:
        raise RuntimeError(
            "Sampling plugin configuration is ambiguous: use either --extra_sampling_plugins or "
            "(--extra_sampling_fallback / --extra_sampling_modules), not both."
        )

    if has_plugins:
        _sampling_plugin_config = merge_sampling_plugin_configs(
            load_entry_point_plugins(
                plugin_names,
                entry_point_group=ENTRY_POINT_GROUP,
                parser=parse_sampling_plugin_config,
                plugin_kind=PLUGIN_KIND,
            )
        )
    elif has_direct:
        _sampling_plugin_config = direct_config
    else:
        _sampling_plugin_config = SamplingPluginConfig()

    validate_sampling_plugin_config(_sampling_plugin_config)

    return _sampling_plugin_config


def parse_sampling_plugin_config(value: Any) -> SamplingPluginConfig:
    return parse_plugin_config(
        value,
        SamplingPluginConfig,
        fields=("extra_fallback", "extra_modules"),
        plugin_kind=PLUGIN_KIND,
    )


def get_sampling_plugin_config() -> SamplingPluginConfig:
    if _sampling_plugin_config is None:
        return SamplingPluginConfig()
    return _sampling_plugin_config


def resolve_sampling_fallback(
    platform: str,
    plugin_config: SamplingPluginConfig | None = None,
) -> tuple[str, ...]:
    from lightllm.platform.base.registry import get_platform_spec

    config = plugin_config or get_sampling_plugin_config()
    merged: list[str] = []
    seen: set[str] = set()
    for family in config.extra_fallback + get_platform_spec(platform).sampling_fallback:
        if family in seen:
            continue
        seen.add(family)
        merged.append(family)

    return tuple(merged)
