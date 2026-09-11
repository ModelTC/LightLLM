from lightllm.platform.plugin.common import Plugin


OPS = Plugin(name="ops", entry_point_group="lightllm.ops_plugin")
ATT = Plugin(name="att", entry_point_group="lightllm.att_plugin")

_PLUGINS = (OPS, ATT)


def configure_plugins() -> None:
    for plugin in _PLUGINS:
        plugin.load()
