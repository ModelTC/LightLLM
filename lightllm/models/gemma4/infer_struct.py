import torch
from lightllm.common.basemodel import InferStateInfo


class Gemma4InferStateInfo(InferStateInfo):
    def __init__(self):
        super().__init__()
        # Gemma-4 uses two RoPE frequency tables (one per layer type):
        # * sliding_attention layers: theta=10000, full rotation over head_dim=256
        # * full_attention layers:    theta=1_000_000, partial rotation (first 25% of head_dim=512)
        self.position_cos_sliding = None
        self.position_sin_sliding = None
        self.position_cos_full = None
        self.position_sin_full = None

    def init_some_extra_state(self, model):
        super().init_some_extra_state(model)
        position_ids = self.position_ids
        self.position_cos_sliding = torch.index_select(model._cos_cached_sliding, 0, position_ids).view(
            position_ids.shape[0], -1
        )
        self.position_sin_sliding = torch.index_select(model._sin_cached_sliding, 0, position_ids).view(
            position_ids.shape[0], -1
        )
        self.position_cos_full = torch.index_select(model._cos_cached_full, 0, position_ids).view(
            position_ids.shape[0], -1
        )
        self.position_sin_full = torch.index_select(model._sin_cached_full, 0, position_ids).view(
            position_ids.shape[0], -1
        )
        if self.is_prefill:
            self.max_seq_len = self.max_kv_seq_len
        return
