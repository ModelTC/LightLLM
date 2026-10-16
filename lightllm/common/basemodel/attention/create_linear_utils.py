"""Attention selection through the configured operator backend."""

from lightllm.backends import get_operator_backend


def get_qwen35_linear_prefill_att_backend_class(index=1, priority_list=("flashqla", "triton")):
    selector = get_operator_backend().selection_module("linear_attention")
    return selector.get_qwen35_linear_prefill_att_backend_class(index=index, priority_list=priority_list)


def get_qwen35_linear_decode_att_backend_class(index=1, priority_list=("triton",)):
    selector = get_operator_backend().selection_module("linear_attention")
    return selector.get_qwen35_linear_decode_att_backend_class(index=index, priority_list=priority_list)


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # Retain existing imports of backend maps and private selection helpers.
    return getattr(get_operator_backend().selection_module("linear_attention"), name)
