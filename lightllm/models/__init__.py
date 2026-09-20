"""Model selection APIs and lazy compatibility exports for model classes."""

from .registry import get_model, get_model_class, load_model_class
from .draft_registry import get_draft_model_class
from .builtin import MODEL_CLASS_PATHS


__all__ = ["get_model", "get_model_class", "get_draft_model_class", *MODEL_CLASS_PATHS]


def __getattr__(name: str):
    if name in MODEL_CLASS_PATHS:
        return load_model_class(MODEL_CLASS_PATHS[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
