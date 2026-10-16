"""Attention selection through the configured operator backend."""

from lightllm.backends import get_operator_backend
from .base_att import BaseVitAttBackend


def get_vit_att_backend_class(backend_name: str) -> BaseVitAttBackend:
    selector = get_operator_backend().selection_module("vision_attention")
    return selector.get_vit_att_backend_class(backend_name=backend_name)


def init_vit_att_backend(index=0, priority_list: list = ["fa3", "xformers", "sdpa", "triton"]) -> str:
    selector = get_operator_backend().selection_module("vision_attention")
    return selector.init_vit_att_backend(index=index, priority_list=priority_list)


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # Retain existing imports of backend maps and private selection helpers.
    return getattr(get_operator_backend().selection_module("vision_attention"), name)
