from importlib import import_module

_EXPORTS = {
    "KvCacheAllocator": (".allocator", "KvCacheAllocator"),
    "MemoryManager": (".mem_manager", "MemoryManager"),
    "ReadOnlyStaticsMemoryManager": (".mem_manager", "ReadOnlyStaticsMemoryManager"),
    "PPLINT8KVMemoryManager": (".ppl_int8kv_mem_manager", "PPLINT8KVMemoryManager"),
    "PPLINT4KVMemoryManager": (".ppl_int4kv_mem_manager", "PPLINT4KVMemoryManager"),
    "Deepseek2MemoryManager": (".deepseek2_mem_manager", "Deepseek2MemoryManager"),
    "Deepseek3_2MemoryManager": (".deepseek3_2mem_manager", "Deepseek3_2MemoryManager"),
    "FP8PerTokenGroupQuantDeepseek3_2MemoryManager": (
        ".fp8_per_token_group_quant_deepseek3_2mem_manager",
        "FP8PerTokenGroupQuantDeepseek3_2MemoryManager",
    ),
    "FP8StaticPerHeadQuantMemManager": (".fp8_static_per_head_quant_mem_manager", "FP8StaticPerHeadQuantMemManager"),
    "FP8StaticPerTensorQuantMemManager": (
        ".fp8_static_per_tensor_quant_mem_manager",
        "FP8StaticPerTensorQuantMemManager",
    ),
    "Qwen3NextMemManager": (".qwen3next_mem_manager", "Qwen3NextMemManager"),
    "Glm5NextMemManager": (".glm5_next_mem_manager", "Glm5NextMemManager"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attr = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attr)
    globals()[name] = value
    return value
