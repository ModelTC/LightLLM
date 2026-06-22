from dataclasses import dataclass
from importlib.metadata import entry_points, EntryPoint
from typing import Any, Callable, Generic, Iterable, List, TypeVar

C = TypeVar("C", bound="ModulePluginConfig")


@dataclass(frozen=True)
class ModulePluginConfig:
    """ Plugin config for module-based plugins. e.g., ATT """
    extra_modules: tuple[str, ...] = ()


@dataclass(frozen=True)
class FallbackPluginConfig(ModulePluginConfig):
    """ Plugin config for fallback-based plugins. e.g., OPS, SAMPLING """
    extra_fallback: tuple[str, ...] = ()


@dataclass(frozen=True)
class PluginKindSpec(Generic[C]):
    kind: str
    entry_point_group: str
    config_cls: type[C]
    fields: tuple[str, ...]
    ambiguous_error: str

    def merge_configs(self, configs: Iterable[C]) -> C:
        return self.config_cls(**{
            field: merge_config_field(configs, field)
            for field in self.fields
        })

    def has_direct_config(self, config: C) -> bool:
        return any(getattr(config, field) for field in self.fields)

    def cli_flag(self, field: str) -> str:
        return f"--extra_{self.kind}_{field}"

    @property
    def example_plugin_name(self) -> str:
        return f"example_{self.kind}_plugin"


class PluginKind(Generic[C]):

    def __init__(
        self,
        spec: PluginKindSpec[C],
        *,
        validator: Callable[[C], None] | None = None,
        resolve_fallback: Callable[[str, C | None], tuple[str, ...]] | None = None,
    ) -> None:
        self._spec = spec
        self._validator = validator
        self._resolve_fallback = resolve_fallback
        self._config: C | None = None

    @property
    def spec(self) -> PluginKindSpec[C]:
        return self._spec

    @property
    def kind(self) -> str:
        return self.spec.kind

    @property
    def config_cls(self) -> type[C]:
        return self.spec.config_cls

    def configure(self) -> C:
        # ops -> extra_ops_plugins, att -> extra_att_plugins, sampling -> extra_sampling_plugins
        plugin_names = plugin_names_from_cli(self._spec.kind)
        # Build *Config based on fallback/modules CLI flags
        direct_config = plugin_config_from_cli(
            self._spec.config_cls,
            plugin_kind=self._spec.kind,
            fields=self._spec.fields,
        )

        has_plugins = bool(plugin_names)
        has_direct = self._spec.has_direct_config(direct_config)
        if has_plugins and has_direct:
            raise RuntimeError(self._spec.ambiguous_error)

        if has_plugins:
            config = self._spec.merge_configs(
                load_entry_point_plugins(
                    plugin_names,
                    entry_point_group=self._spec.entry_point_group,
                    parser=self.parse,
                    plugin_kind=self._spec.kind,
                )
            )
        elif has_direct:
            config = direct_config
        else:
            config = self._spec.config_cls()

        if self._validator is not None:
            self._validator(config)

        self._config = config
        return config

    def get(self) -> C:
        if self._config is None:
            return self._spec.config_cls()
        return self._config

    def parse(self, value: Any) -> C:
        return parse_plugin_config(
            value,
            self._spec.config_cls,
            fields=self._spec.fields,
            plugin_kind=self._spec.kind,
        )

    def installed(self) -> tuple[str, ...]:
        return list_installed_plugin_names(self.spec.entry_point_group)

    def resolve_fallback_for(
        self,
        platform: str,
        config: C | None = None,
    ) -> tuple[str, ...]:
        if self._resolve_fallback is None:
            raise TypeError(
                f"Plugin kind {self.kind!r} does not support fallback resolution"
            )
        return self._resolve_fallback(platform, config)


def make_fallback_plugin_validator(
    spec: PluginKindSpec[FallbackPluginConfig],
    *,
    register_decorator: str,
) -> Callable[[FallbackPluginConfig], None]:
    def validate(config: FallbackPluginConfig) -> None:
        if config.extra_modules and not config.extra_fallback:
            modules_flag = spec.cli_flag("modules")
            fallback_flag = spec.cli_flag("fallback")
            raise RuntimeError(
                f"{modules_flag} requires {fallback_flag}: external modules must "
                f"{register_decorator} under impl family names listed in extra_fallback."
            )

    return validate


def make_module_plugin_kind(
    *,
    kind: str,
    entry_point_group: str,
    ambiguous_error: str,
    config_cls: type[ModulePluginConfig] = ModulePluginConfig,
) -> PluginKind[ModulePluginConfig]:
    spec = PluginKindSpec(
        kind=kind,
        entry_point_group=entry_point_group,
        config_cls=config_cls,
        fields=("extra_modules",),
        ambiguous_error=ambiguous_error,
    )
    return PluginKind(spec=spec)


def make_fallback_plugin_kind(
    *,
    kind: str,
    entry_point_group: str,
    ambiguous_error: str,
    platform_fallback_field: str,
    register_decorator: str,
    config_cls: type[FallbackPluginConfig] = FallbackPluginConfig,
) -> PluginKind[FallbackPluginConfig]:
    spec = PluginKindSpec(
        kind=kind,
        entry_point_group=entry_point_group,
        config_cls=config_cls,
        fields=("extra_fallback", "extra_modules"),
        ambiguous_error=ambiguous_error,
    )
    validator = make_fallback_plugin_validator(
        spec,
        register_decorator=register_decorator,
    )

    plugin = PluginKind(spec=spec, validator=validator)
    plugin._resolve_fallback = make_fallback_resolver(plugin.get, platform_fallback_field)

    return plugin


def make_fallback_resolver(
    get_config: Callable[[], FallbackPluginConfig],
    platform_fallback_field: str,
    *,
    config_fallback_field: str = "extra_fallback",
) -> Callable[[str, FallbackPluginConfig | None], tuple[str, ...]]:

    def resolve(
        platform: str,
        plugin_config: FallbackPluginConfig | None = None,
    ) -> tuple[str, ...]:
        from lightllm.platform.base.registry import get_platform_spec

        config = plugin_config or get_config()
        platform_fallback = getattr(get_platform_spec(platform), platform_fallback_field)
        merged: list[str] = []
        seen: set[str] = set()
        for family in getattr(config, config_fallback_field) + platform_fallback:
            if family in seen:
                continue
            seen.add(family)
            merged.append(family)
        return tuple(merged)

    return resolve


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
    config_cls: type[C],
    *,
    plugin_kind: str,
    fields: tuple[str, ...],
) -> C:
    from lightllm.utils.envs_utils import get_env_start_args

    start_args = get_env_start_args()
    return config_cls(**{
        field: parse_csv(getattr(start_args, f"extra_{plugin_kind}_{field.removeprefix('extra_')}", None))
        for field in fields
    })


def load_entry_point_plugins(
    plugin_names: tuple[str, ...],
    *,
    entry_point_group: str,
    parser: Callable[[Any], C],
    plugin_kind: str,
) -> List[C]:
    """ Load entry point plugins from a group. """
    if not plugin_names:
        return []

    selected = set(plugin_names)

    configs: List[C] = []
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
    config_cls: type[C],
    *,
    fields: tuple[str, ...],
    plugin_kind: str,
) -> C:
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
