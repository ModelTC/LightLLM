__all__ = ["Qwen3NextTpPartModel"]


def __getattr__(name):
    if name == "Qwen3NextTpPartModel":
        from .model import Qwen3NextTpPartModel

        return Qwen3NextTpPartModel
    raise AttributeError(name)
