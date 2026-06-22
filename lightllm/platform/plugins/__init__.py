from lightllm.platform.plugins.att import ATT
from lightllm.platform.plugins.ops import OPS
from lightllm.platform.plugins.sampling import SAMPLING

ALL_PLUGINS = (OPS, SAMPLING, ATT)


def configure_plugins() -> None:
    for plugin in ALL_PLUGINS:
        plugin.configure()


__all__ = [
    "OPS",
    "SAMPLING",
    "ATT",
    "configure_plugins",
]
