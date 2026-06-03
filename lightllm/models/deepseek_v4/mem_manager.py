from lightllm.common.kv_cache_mem_manager.deepseek2_mem_manager import Deepseek2MemoryManager


class DeepseekV4MemoryManager(Deepseek2MemoryManager):
    """Stores the per-token MLA KV (head_num=1, head_dim=512), reusing the deepseek2 layout/operator.

    The prefill path computes attention in-layer from the request's hidden states, so it does not read
    this buffer. The decode/incremental path (M6) will add the sliding-window ring + compressed-KV +
    per-request compressor-state buffers here.
    """

    pass
