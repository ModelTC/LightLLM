import importlib
from dataclasses import dataclass
from importlib.metadata import entry_points, EntryPoint
from typing import Any, Iterable, List, Mapping

from lightllm.utils.envs_utils import get_env_start_args


@dataclass(frozen=True)
class PluginConfig:
    modules: tuple[str, ...] = ()


class Plugin:
    def __init__(self, name: str, entry_point_group: str) -> None:
        """Loader for pip-installed plugins.

        Args:
            name: Plugin kind.
            entry_point_group: Entry point group name. Looked up with ``importlib.metadata``.

            pyproject.toml:
                [project.entry-points."lightllm.ops_plugin"]
                example_ops = "lightllm_example_plugin.register:register_ops"

            setup.py:
                setup(
                    entry_points={
                        "lightllm.ops_plugin": [
                            "example_ops = lightllm_example_plugin.register:register_ops",
                        ],
                    },
                )

            After ``pip install``, pass ``--extra-ops example_ops`` to load and register.
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
        available_entry_points = list(_iter_entry_points(self.entry_point_group))
        for entry_point in available_entry_points:
            # Only load entry points from CLI.
            if entry_point.name not in given_names:
                continue

            # example_ops = "lightllm_example_plugin.register:register_ops"
            # -> register_fn = lightllm_example_plugin.register.register_ops
            register_fn = entry_point.load()
            config = parse_plugin_config(register_fn(), plugin_kind=self.name)

            configs.append(config)
            loaded_names.add(entry_point.name)

        # Check if all required plugins are loaded.
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


def _iter_entry_points(entry_point_group: str) -> Iterable[EntryPoint]:
    eps = entry_points()
    if hasattr(eps, "select"):
        yield from eps.select(group=entry_point_group)
    else:
        yield from eps.get(entry_point_group, [])


def _to_str_tuple(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()

    if isinstance(value, str):
        parts: Iterable[str] = value.split(",")
    else:
        parts = value
    return tuple(item.strip() for item in parts if item and item.strip())
