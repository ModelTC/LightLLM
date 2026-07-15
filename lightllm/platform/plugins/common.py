from dataclasses import dataclass
from importlib.metadata import entry_points, EntryPoint
from typing import Iterable, Any, Callable, List

from lightllm.utils.envs_utils import get_env_start_args


@dataclass(frozen=True)
class PluginConfig:
    modules: tuple[str, ...] = ()
    fallback: tuple[str, ...] = ()


class Plugin:
    def __init__(
        self,
        name: str,
        entry_point_group: str,
        *,
        register_decorator: str,
        platform_fallback_field: str | None = None,
    ) -> None:
        self.name = name
        self.entry_point_group = entry_point_group
        self.register_decorator = register_decorator
        self.platform_fallback_field = platform_fallback_field
        self.config: PluginConfig | None = None

    @property
    def fields(self) -> tuple[str, ...]:
        if self.platform_fallback_field is not None:
            return ("modules", "fallback")
        return ("modules",)

    def cli_flag(self, field: str) -> str:
        return f"--extra_{self.name}_{field}"

    def cli_help(self, field: str) -> str:
        if field == "plugins":
            return (
                f"Comma-separated pip {self.name} plugin names "
                f"(entry point group: {self.entry_point_group})."
            )
        if field == "modules":
            return (
                f"Comma-separated Python modules to import for external "
                f"{self.register_decorator} implementations."
            )
        if field == "fallback":
            return (
                f"Comma-separated impl families prepended to the platform "
                f"{self.name} fallback chain. Use with {self.cli_flag('modules')} "
                f"for local overrides without a pip plugin package."
            )
        raise ValueError(f"Unknown plugin CLI field: {field!r}")

    def configure(self) -> None:
        plugins_config = self._config_from_plugins_cli()
        direct_config = self._config_from_direct_cli()

        # 校验，二者不可同时指定
        if plugins_config is not None and direct_config is not None:
            plugins_flag = self.cli_flag("plugins")
            if self.platform_fallback_field is not None:
                direct_flags = f"({self.cli_flag('modules')}, {self.cli_flag('fallback')})"
            else:
                direct_flags = self.cli_flag("modules")
            raise RuntimeError(
                f"{self.name} plugin configuration is ambiguous: "
                f"use either {plugins_flag} or {direct_flags}, not both."
            )

        self.config = plugins_config or direct_config or PluginConfig()
        self._validate_config(self.config)

    def get(self) -> PluginConfig:
        return self.config or PluginConfig()

    def _config_from_plugins_cli(self) -> PluginConfig | None:
        # 根据启动参数读 pip 安装的插件列表，读取自 --extra_xxx_plugins 参数
        start_args = get_env_start_args()
        names = to_str_tuple(getattr(start_args, f"extra_{self.name}_plugins", None))
        if not names:
            return None

        # 加载这些插件
        plugins_config = load_entry_point_plugins(
            names,
            entry_point_group=self.entry_point_group,
            fields=self.fields,
            plugin_kind=self.name,
        )
        return PluginConfig(
            **{field: merge_config_field(plugins_config, field) for field in self.fields}
        )

    def _config_from_direct_cli(self) -> PluginConfig | None:
        # 根据启动参数读内部插件，读取自 --extra_xxx_modules / --extra_xxx_fallback 参数
        start_args = get_env_start_args()
        config = PluginConfig(
            **{
                field: to_str_tuple(getattr(start_args, f"extra_{self.name}_{field}", None))
                for field in self.fields
            }
        )

        if not any(getattr(config, field) for field in self.fields):
            return None
        return config

    def _validate_config(self, config: PluginConfig) -> None:
        # ATT 插件（只有 modules 字段）不用校验 fallback 字段
        if self.platform_fallback_field is None:
            return

        # 校验，如果指定了 modules，则必须指定 fallback
        if config.modules and not config.fallback:
            raise RuntimeError(
                f"{self.cli_flag('modules')} requires {self.cli_flag('fallback')}: "
                f"external modules must register under impl family names listed in fallback."
            )


def merge_config_field(configs: Iterable[Any], field_name: str) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for config in configs:
        for item in getattr(config, field_name):
            if item in seen:
                continue
            seen.add(item)
            merged.append(item)
    return tuple(merged)


def _iter_entry_points(entry_point_group: str) -> Iterable[EntryPoint]:
    eps = entry_points()
    if hasattr(eps, "select"):
        yield from eps.select(group=entry_point_group)
    else:
        yield from eps.get(entry_point_group, [])


def load_entry_point_plugins(
    plugin_names: tuple[str, ...],
    *,
    entry_point_group: str,
    fields: tuple[str, ...],
    plugin_kind: str,
) -> List[PluginConfig]:
    # 按名字加载 pip 安装的 entry point 插件，每个插件解析成一份 PluginConfig
    selected = set(plugin_names)
    configs: List[PluginConfig] = []
    loaded_names: set[str] = set()
    for entry_point in _iter_entry_points(entry_point_group):
        if entry_point.name not in selected:
            continue
        # 执行插件注册函数，并用 parse_plugin_config 转成 PluginConfig
        register_fn: Callable[[], Any] = entry_point.load()
        configs.append(parse_plugin_config(register_fn(), fields, plugin_kind))
        loaded_names.add(entry_point.name)

    def list_installed_plugin_names(entry_point_group: str) -> tuple[str, ...]:
        return tuple(sorted(ep.name for ep in _iter_entry_points(entry_point_group)))

    # 校验：请求的插件名都必须能在 entry point 组里找到
    missing = selected - loaded_names
    if missing:
        available = list_installed_plugin_names(entry_point_group)
        message = (
            f"{plugin_kind} plugin(s) not found in entry point group "
            f"{entry_point_group!r}: {sorted(missing)}"
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
    fields: tuple[str, ...],
    plugin_kind: str,
) -> PluginConfig:
    if value is None:
        return PluginConfig()
    if isinstance(value, PluginConfig):
        return value
    if isinstance(value, dict):
        return PluginConfig(
            **{
                field: to_str_tuple(value.get(field))
                for field in fields
            }
        )
    raise TypeError(f"Unsupported {plugin_kind} plugin config type: {type(value)!r}")


def to_str_tuple(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()

    if isinstance(value, str):
        parts: Iterable[str] = value.split(",")
    else:
        parts = value
    return tuple(item.strip() for item in parts if item and item.strip())
