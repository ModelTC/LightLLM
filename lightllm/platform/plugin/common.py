import importlib
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any, Iterable, List, Mapping

from lightllm.utils.envs_utils import get_env_start_args

_registering_plugin = False


@contextmanager
def plugin_registration():
    global _registering_plugin
    previous = _registering_plugin
    _registering_plugin = True
    try:
        yield
    finally:
        _registering_plugin = previous


def is_plugin_registering() -> bool:
    return _registering_plugin


@dataclass(frozen=True)
class PluginConfig:
    modules: tuple[str, ...] = ()


class Plugin:
    def __init__(self, name: str, entry_point_group: str) -> None:
        """加载 pip 安装的插件。

        Args:
            name: 插件种类。
            entry_point_group: entry point 组名，用 ``importlib.metadata`` 查找。

            pyproject.toml:
                [project.entry-points."lightllm.ops_plugin"]
                example_ops = "lightllm_example_plugin.register:register_ops_plugin"

                [project.entry-points."lightllm.att_plugin"]
                example_att = "lightllm_example_plugin.register:register_att_plugin"

            setup.py:
                setup(
                    entry_points={
                        "lightllm.ops_plugin": [
                            "example_ops = lightllm_example_plugin.register:register_ops_plugin",
                        ],
                        "lightllm.att_plugin": [
                            "example_att = lightllm_example_plugin.register:register_att_plugin",
                        ],
                    },
                )

            ``pip install`` 之后，传入 ``--extra_ops example_ops`` 或 ``--extra_att example_att`` 加载插件。
            被加载的模块调用 ``register_ops`` 或 ``register_att_backend``。同一个 key 已有内置实现，则替换。
        """
        self.name = name
        self.entry_point_group = entry_point_group

    def load(self) -> None:
        names = self._name_from_cli()
        if not names:
            return

        configs = self._load_entry_point_plugins(names)
        config = PluginConfig(modules=merge_config_field(configs, "modules"))

        for module in config.modules:
            importlib.import_module(module)

    def _name_from_cli(self) -> tuple[str, ...]:
        start_args = get_env_start_args()
        return _to_str_tuple(getattr(start_args, f"extra_{self.name}", None))

    def _load_entry_point_plugins(self, plugin_names: tuple[str, ...]) -> List[PluginConfig]:
        given_names = set(plugin_names)
        configs: List[PluginConfig] = []
        loaded_names: set[str] = set()
        available_entry_points = list(entry_points().select(group=self.entry_point_group))
        for entry_point in available_entry_points:
            # 只从 CLI 加载 entry point
            if entry_point.name not in given_names:
                continue

            # example_ops = "lightllm_example_plugin.register:register_ops_plugin"
            # -> register_fn = lightllm_example_plugin.register.register_ops_plugin
            register_fn = entry_point.load()
            config = parse_plugin_config(register_fn(), plugin_kind=self.name)

            configs.append(config)
            loaded_names.add(entry_point.name)

        # 检查是否所有需要的插件都已加载
        missing = given_names - loaded_names
        if missing:
            available = tuple(sorted(ep.name for ep in available_entry_points))
            message = (
                f"{self.name} plugin(s) not found in entry point group "
                f"{self.entry_point_group!r}: {sorted(missing)}"
            )
            if available:
                message += f". Installed plugins: {available}"
            else:
                message += (
                    f". No {self.name} plugins installed; register entry points in group "
                    f"{self.entry_point_group!r} and pip install -e your plugin package."
                )
            raise RuntimeError(message)

        return configs


def parse_plugin_config(value: Any, plugin_kind: str) -> PluginConfig:
    if not isinstance(value, Mapping):
        raise TypeError(f"{plugin_kind} plugin config must be a mapping, got {type(value)}")

    return PluginConfig(modules=_to_str_tuple(value.get("modules")))


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


def _to_str_tuple(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()

    if isinstance(value, str):
        parts: Iterable[str] = value.split(",")
    else:
        parts = value
    return tuple(item.strip() for item in parts if item and item.strip())
