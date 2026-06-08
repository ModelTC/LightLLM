import torch
from lightllm.common.basemodel import InferStateInfo
from lightllm.common.req_manager import DeepseekV4ReqManager
from lightllm.common.kv_cache_mem_manager import DeepseekV4MemoryManager


class DeepseekV4InferStateInfo(InferStateInfo):
    req_manager: DeepseekV4ReqManager
    mem_manager: DeepseekV4MemoryManager

    """Per-token interleaved-rope cos/sin for the two rope variants (sliding / compressed), following
    the gemma4 two-variant convention (_cos_cached_* -> position_cos_*). Also exposes the full compressed
    cos/sin tables, which the KV compressor indexes at window positions (not per-token)."""

    def __init__(self):
        super().__init__()
        self.position_cos_sliding = None
        self.position_sin_sliding = None
        self.position_cos_compress = None
        self.position_sin_compress = None
        self.cos_compress_table = None
        self.sin_compress_table = None

    def init_some_extra_state(self, model):
        super().init_some_extra_state(model)  # sets position_ids, b_q_seq_len, b_q_start_loc (prefill)
        pos = self.position_ids
        self.position_cos_sliding = torch.index_select(model._cos_cached_sliding, 0, pos)  # [T, rope_dim//2]
        self.position_sin_sliding = torch.index_select(model._sin_cached_sliding, 0, pos)
        self.position_cos_compress = torch.index_select(model._cos_cached_compress, 0, pos)
        self.position_sin_compress = torch.index_select(model._sin_cached_compress, 0, pos)
        self.cos_compress_table = model._cos_cached_compress
        self.sin_compress_table = model._sin_cached_compress
