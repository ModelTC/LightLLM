from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any, Callable, Iterable
from lightllm.utils.envs_utils import get_env_start_args

OP_PLUGIN_ENTRY_GROUP = "lightllm.op_plugins"


@dataclass(frozen=True)
class OpsPluginConfig:
    extra_fallback: tuple[str, ...] = ()
    extra_modules: tuple[str, ...] = ()


_ops_plugin_config: OpsPluginConfig | None = None


def _parse_csv(value: str | None) -> tuple[str, ...]:
    """ Parse a comma-separated string into a tuple of strings. """
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _normalize_tuple(values: Iterable[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(value.strip() for value in values if value and value.strip())


def merge_ops_plugin_configs(configs: Iterable[OpsPluginConfig]) -> OpsPluginConfig:
    extra_fallback: list[str] = []
    extra_modules: list[str] = []
    seen_fallback: set[str] = set()
    seen_modules: set[str] = set()
    for config in configs:
        for family in config.extra_fallback:
            if family not in seen_fallback:
                seen_fallback.add(family)
                extra_fallback.append(family)
        for module_name in config.extra_modules:
            if module_name not in seen_modules:
                seen_modules.add(module_name)
                extra_modules.append(module_name)

    return OpsPluginConfig(
        extra_fallback=tuple(extra_fallback),
        extra_modules=tuple(extra_modules),
    )


def _coerce_plugin_config(value: Any) -> OpsPluginConfig:
    if value is None:
        return OpsPluginConfig()
    if isinstance(value, OpsPluginConfig):
        return value
    if isinstance(value, dict):
        return OpsPluginConfig(
            extra_fallback=_normalize_tuple(value.get("extra_fallback")),
            extra_modules=_normalize_tuple(value.get("extra_modules")),
        )
    raise TypeError(f"Unsupported op plugin config type: {type(value)!r}")


def _iter_op_plugin_entry_points():
    eps = entry_points()
    # For Python 3.10+
    if hasattr(eps, "select"):
        yield from eps.select(group=OP_PLUGIN_ENTRY_GROUP)
        return
    # For Python 3.9 and below
    yield from eps.get(OP_PLUGIN_ENTRY_GROUP, [])


def _load_entry_point_plugins(plugin_names: tuple[str, ...]) -> list[OpsPluginConfig]:
    if not plugin_names:
        return []

    selected = set(plugin_names)
    configs: list[OpsPluginConfig] = []
    loaded_names: set[str] = set()

    for entry_point in _iter_op_plugin_entry_points():
        if entry_point.name not in selected:
            continue
        # Load the plugin
        register_fn: Callable[[], Any] = entry_point.load()
        configs.append(_coerce_plugin_config(register_fn()))
        loaded_names.add(entry_point.name)

    # Check if any plugins are missings
    missing = selected - loaded_names
    if missing:
        available = sorted(entry_point.name for entry_point in _iter_op_plugin_entry_points())
        message = (
            f"Op plugin(s) not found in entry point group {OP_PLUGIN_ENTRY_GROUP!r}: "
            f"{sorted(missing)}"
        )
        if available:
            message += f". Installed plugins: {available}"
        else:
            message += (
                ". No op plugins installed; register entry points in group "
                f"{OP_PLUGIN_ENTRY_GROUP!r} and pip install -e your plugin package."
            )
        raise RuntimeError(message)

    return configs


def _list_installed_op_plugin_names() -> tuple[str, ...]:
    return tuple(sorted(entry_point.name for entry_point in _iter_op_plugin_entry_points()))


def _validate_direct_ops_config(config: OpsPluginConfig) -> None:
    from lightllm.platform.base.registry import has_builtin_ops_module

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
    installed = _list_installed_op_plugin_names()
    if installed:
        hints.append(f"Installed op plugins: {list(installed)}.")
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


def _plugin_config_from_cli() -> OpsPluginConfig:
    args = get_env_start_args()
    return OpsPluginConfig(
        extra_fallback=_parse_csv(getattr(args, "extra_op_fallback", None)),
        extra_modules=_parse_csv(getattr(args, "extra_op_modules", None)),
    )


def _collect_op_plugin_names() -> tuple[str, ...]:
    args = get_env_start_args()
    return _parse_csv(getattr(args, "extra_op_plugins", None))


def configure_op_plugins() -> OpsPluginConfig:
    global _ops_plugin_config

    # Collect plugin names from CLI
    plugin_names = _collect_op_plugin_names()
    # Collect plugin config from CLI
    direct_config = _plugin_config_from_cli()
    # Check if there are plugins or direct config
    has_plugins = bool(plugin_names)
    has_direct = bool(direct_config.extra_fallback or direct_config.extra_modules)
    # Check if both plugins and direct config are present
    if has_plugins and has_direct:
        raise RuntimeError(
            "Op plugin configuration is ambiguous: use either "
            "--extra_op_plugins or (--extra_op_fallback / --extra_op_modules), not both."
        )
    # Load plugins if present
    if has_plugins:
        _ops_plugin_config = merge_ops_plugin_configs(_load_entry_point_plugins(plugin_names))
    # Use direct config if present
    elif has_direct:
        _validate_direct_ops_config(direct_config)
        _ops_plugin_config = direct_config
    # Use default config if no plugins or direct config are present
    else:
        _ops_plugin_config = OpsPluginConfig()

    return _ops_plugin_config


def get_ops_plugin_config() -> OpsPluginConfig:
    if _ops_plugin_config is None:
        return OpsPluginConfig()
    return _ops_plugin_config


def resolve_op_fallback(platform: str, plugin_config: OpsPluginConfig | None = None) -> tuple[str, ...]:
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


ATT_PLUGIN_ENTRY_GROUP = "lightllm.att_plugins"


@dataclass(frozen=True)
class AttPluginConfig:
    extra_modules: tuple[str, ...] = ()


_att_plugin_config: AttPluginConfig | None = None


def _coerce_att_plugin_config(value: Any) -> AttPluginConfig:
    if value is None:
        return AttPluginConfig()
    if isinstance(value, AttPluginConfig):
        return value
    if isinstance(value, dict):
        return AttPluginConfig(
            extra_modules=_normalize_tuple(value.get("extra_modules")),
        )
    raise TypeError(f"Unsupported att plugin config type: {type(value)!r}")


def _iter_att_plugin_entry_points():
    eps = entry_points()
    if hasattr(eps, "select"):
        yield from eps.select(group=ATT_PLUGIN_ENTRY_GROUP)
        return
    yield from eps.get(ATT_PLUGIN_ENTRY_GROUP, [])


def _load_att_entry_point_plugins(plugin_names: tuple[str, ...]) -> list[AttPluginConfig]:
    if not plugin_names:
        return []

    selected = set(plugin_names)
    configs: list[AttPluginConfig] = []
    loaded_names: set[str] = set()

    for entry_point in _iter_att_plugin_entry_points():
        if entry_point.name not in selected:
            continue
        register_fn: Callable[[], Any] = entry_point.load()
        configs.append(_coerce_att_plugin_config(register_fn()))
        loaded_names.add(entry_point.name)

    missing = selected - loaded_names
    if missing:
        available = sorted(entry_point.name for entry_point in _iter_att_plugin_entry_points())
        message = (
            f"Attention plugin(s) not found in entry point group {ATT_PLUGIN_ENTRY_GROUP!r}: "
            f"{sorted(missing)}"
        )
        if available:
            message += f". Installed plugins: {available}"
        else:
            message += (
                ". No att plugins installed; register entry points in group "
                f"{ATT_PLUGIN_ENTRY_GROUP!r} and pip install -e your plugin package."
            )
        raise RuntimeError(message)

    return configs


def merge_att_plugin_configs(configs: Iterable[AttPluginConfig]) -> AttPluginConfig:
    extra_modules: list[str] = []
    seen_modules: set[str] = set()
    for config in configs:
        for module_name in config.extra_modules:
            if module_name not in seen_modules:
                seen_modules.add(module_name)
                extra_modules.append(module_name)

    return AttPluginConfig(extra_modules=tuple(extra_modules))


def _att_plugin_config_from_cli() -> AttPluginConfig:
    args = get_env_start_args()
    return AttPluginConfig(
        extra_modules=_parse_csv(getattr(args, "extra_att_modules", None)),
    )


def _collect_att_plugin_names() -> tuple[str, ...]:
    args = get_env_start_args()
    return _parse_csv(getattr(args, "extra_att_plugins", None))


def configure_att_plugins() -> AttPluginConfig:
    global _att_plugin_config

    plugin_names = _collect_att_plugin_names()
    direct_config = _att_plugin_config_from_cli()
    has_plugins = bool(plugin_names)
    has_direct = bool(direct_config.extra_modules)

    if has_plugins and has_direct:
        raise RuntimeError(
            "Attention plugin configuration is ambiguous: use either "
            "--extra_att_plugins or --extra_att_modules, not both."
        )

    if has_plugins:
        _att_plugin_config = merge_att_plugin_configs(_load_att_entry_point_plugins(plugin_names))
    elif has_direct:
        _att_plugin_config = direct_config
    else:
        _att_plugin_config = AttPluginConfig()

    return _att_plugin_config


def get_att_plugin_config() -> AttPluginConfig:
    if _att_plugin_config is None:
        return AttPluginConfig()
    return _att_plugin_config
