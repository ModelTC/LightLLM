from dataclasses import dataclass
from importlib.metadata import entry_points, EntryPoint
from typing import Any, Callable, Iterable, List, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class ModulePluginConfig:
    extra_modules: tuple[str, ...] = ()


def parse_csv(value: str | None) -> tuple[str, ...]:
    """ Parse a comma-separated string into a tuple of strings. """
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def normalize_tuple(values: Iterable[str] | None) -> tuple[str, ...]:
    """ Normalize a list of strings into a tuple of strings. """
    if not values:
        return ()
    return tuple(value.strip() for value in values if value and value.strip())


def merge_config_field(configs: Iterable[Any], field_name: str) -> tuple[str, ...]:
    """ Merge a list of config objects by a field name. """
    added: set[str] = set()
    for config in configs:
        for item in getattr(config, field_name):
            if item not in added:
                added.add(item)
    return tuple(added)


def _iter_entry_points(entry_point_group: str) -> Iterable[EntryPoint]:
    """ Iterate over entry points in a group. """
    entry_point = entry_points()
    if hasattr(entry_point, "select"):
        yield from entry_point.select(group=entry_point_group)
    else:
        yield from entry_point.get(entry_point_group, [])


def plugin_names_from_cli(plugin_kind: str) -> tuple[str, ...]:
    from lightllm.utils.envs_utils import get_env_start_args

    args = get_env_start_args()
    return parse_csv(getattr(args, f"extra_{plugin_kind}_plugins", None))


def plugin_config_from_cli(
    config_cls: type[T],
    *,
    plugin_kind: str,
    fields: tuple[str, ...],
) -> T:
    """ Collect plugin config from CLI. """
    from lightllm.utils.envs_utils import get_env_start_args

    start_args = get_env_start_args()
    # Convert to 'extra_op_fallback' / 'extra_op_modules', 'extra_att_modules', etc.
    return config_cls(**{
        field: parse_csv(getattr(start_args, f"extra_{plugin_kind}_{field.removeprefix('extra_')}", None))
        for field in fields
    })


def load_entry_point_plugins(
    plugin_names: tuple[str, ...],
    *,
    entry_point_group: str,
    parser: Callable[[Any], T],
    plugin_kind: str,
) -> List[T]:
    """ Load entry point plugins from a group. """
    if not plugin_names:
        return []

    selected = set(plugin_names)

    configs: List[T] = []
    loaded_names: set[str] = set()
    for entry_point in _iter_entry_points(entry_point_group):
        if entry_point.name not in selected:
            continue
        register_fn: Callable[[], Any] = entry_point.load()
        configs.append(parser(register_fn()))
        loaded_names.add(entry_point.name)

    missing = selected - loaded_names
    if missing:
        available = list_installed_plugin_names(entry_point_group)
        message = (
            f"{plugin_kind} plugin(s) not found in entry point group {entry_point_group!r}: "
            f"{sorted(missing)}"
        ) 
        if available:
            message += f". Installed plugins: {available}"
        else:
            message += (
                f". No {plugin_kind} plugins installed; register entry points in group "
                f"{entry_point_group!r} and pip install -e your plugin package."
            )
        raise RuntimeError(message)

    return configs


def parse_plugin_config(
    value: Any,
    config_cls: type[T],
    *,
    fields: tuple[str, ...],
    plugin_kind: str,
) -> T:
    """ Parse a plugin config from a value. """
    if value is None:
        return config_cls()
    
    if isinstance(value, config_cls):
        return value

    if isinstance(value, dict):
        return config_cls(**{
            field: normalize_tuple(value.get(field.removeprefix(f"extra_{plugin_kind}_")))
            for field in fields
        })

    raise TypeError(f"Unsupported {plugin_kind} plugin config type: {type(value)!r}")


def list_installed_plugin_names(entry_point_group: str) -> tuple[str, ...]:
    return tuple(sorted(entry_point.name for entry_point in _iter_entry_points(entry_point_group)))
