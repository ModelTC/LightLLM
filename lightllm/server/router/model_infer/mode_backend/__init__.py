from importlib import import_module


_BACKEND_EXPORTS = {
    "ChunkedPrefillBackend": (".chunked_prefill.impl", "ChunkedPrefillBackend"),
    "FirstTokenConstraintBackend": (
        ".chunked_prefill.impl_for_first_token_constraint_mode",
        "FirstTokenConstraintBackend",
    ),
    "OutlinesConstraintBackend": (
        ".chunked_prefill.impl_for_outlines_constraint_mode",
        "OutlinesConstraintBackend",
    ),
    "ReturnPromptLogProbBackend": (
        ".chunked_prefill.impl_for_return_all_prompt_logprobs",
        "ReturnPromptLogProbBackend",
    ),
    "RewardModelBackend": (".chunked_prefill.impl_for_reward_model", "RewardModelBackend"),
    "TokenHealingBackend": (".chunked_prefill.impl_for_token_healing", "TokenHealingBackend"),
    "XgrammarBackend": (".chunked_prefill.impl_for_xgrammar_mode", "XgrammarBackend"),
    "DPChunkedPrefillBackend": (".dp_backend.impl", "DPChunkedPrefillBackend"),
    "DiversehBackend": (".diverse_backend.impl", "DiversehBackend"),
    "PDChunkedPrefillForPrefillNode": (
        ".pd.prefill_node_impl.prefill_impl",
        "PDChunkedPrefillForPrefillNode",
    ),
    "PDDPChunkedForPrefillNode": (
        ".pd.prefill_node_impl.prefill_impl_for_dp",
        "PDDPChunkedForPrefillNode",
    ),
    "PDDecodeNode": (".pd.decode_node_impl.decode_impl", "PDDecodeNode"),
    "PDDPForDecodeNode": (".pd.decode_node_impl.decode_impl_for_dp", "PDDPForDecodeNode"),
}


def __getattr__(name):
    if name not in _BACKEND_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _BACKEND_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


__all__ = list(_BACKEND_EXPORTS)
