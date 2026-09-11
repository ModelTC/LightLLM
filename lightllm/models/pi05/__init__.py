__all__ = [
    "Pi05ActionExpertModel",
    "Pi05VLMModel",
]


def __getattr__(name):
    if name in __all__:
        from .model import Pi05ActionExpertModel, Pi05VLMModel

        return {
            "Pi05ActionExpertModel": Pi05ActionExpertModel,
            "Pi05VLMModel": Pi05VLMModel,
        }[name]
    raise AttributeError(name)
