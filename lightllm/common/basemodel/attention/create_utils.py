"""Attention selection through the configured operator backend."""

from lightllm.backends import get_operator_backend
from .base_att import BaseAttBackend


def get_prefill_att_backend_class(index=0, priority_list: list = ["fa3", "flashinfer", "triton"]) -> BaseAttBackend:
    selector = get_operator_backend().selection_module("attention")
    return selector.get_prefill_att_backend_class(index=index, priority_list=priority_list)


def get_decode_att_backend_class(index=0, priority_list: list = ["flashinfer", "fa3", "triton"]) -> BaseAttBackend:
    selector = get_operator_backend().selection_module("attention")
    return selector.get_decode_att_backend_class(index=index, priority_list=priority_list)


def get_mla_prefill_att_backend_class(index=0, priority_list: list = ["fa3", "flashinfer", "triton"]) -> BaseAttBackend:
    selector = get_operator_backend().selection_module("attention")
    return selector.get_mla_prefill_att_backend_class(index=index, priority_list=priority_list)


def get_mla_decode_att_backend_class(index=0, priority_list: list = ["flashinfer", "fa3", "triton"]) -> BaseAttBackend:
    selector = get_operator_backend().selection_module("attention")
    return selector.get_mla_decode_att_backend_class(index=index, priority_list=priority_list)


def get_nsa_prefill_att_backend_class(index=0, priority_list: list = ["flashmla_sparse"]) -> BaseAttBackend:
    selector = get_operator_backend().selection_module("attention")
    return selector.get_nsa_prefill_att_backend_class(index=index, priority_list=priority_list)


def get_nsa_decode_att_backend_class(index=0, priority_list: list = ["flashmla_sparse"]) -> BaseAttBackend:
    selector = get_operator_backend().selection_module("attention")
    return selector.get_nsa_decode_att_backend_class(index=index, priority_list=priority_list)


def get_neo_prefill_att_backend_class(index=0, priority_list: list = ["fa3", "triton"]) -> BaseAttBackend:
    selector = get_operator_backend().selection_module("attention")
    return selector.get_neo_prefill_att_backend_class(index=index, priority_list=priority_list)


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # Retain existing imports of backend maps and private selection helpers.
    return getattr(get_operator_backend().selection_module("attention"), name)
