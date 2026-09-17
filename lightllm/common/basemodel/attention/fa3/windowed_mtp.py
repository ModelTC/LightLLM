import torch

from ..base_att import AttControl, BaseAttBackend, BaseDecodeAttState
from lightllm.utils.sgl_utils import flash_attn_with_kvcache


class WindowedMTPAttBackend(BaseAttBackend):
    """Non-causal block attention over retained draft KV and temporary noise KV."""

    def create_att_decode_state(self, infer_state):
        return WindowedMTPDecodeAttState(backend=self, infer_state=infer_state)


class WindowedMTPDecodeAttState(BaseDecodeAttState):
    def init_state(self):
        model = self.backend.model
        self.block_size = model.block_size
        self.b_req_idx = self.infer_state.b_req_idx[:: self.block_size].long()
        # The base attention-state copy refreshes these tensors on graph replay.
        store = self.infer_state.mem_manager.windowed_draft_kv
        self.cache_seqlens = store.counts.index_select(0, self.b_req_idx) + self.block_size

    def decode_att(self, q, k, v, att_control: AttControl = AttControl(), alloc_func=torch.empty):
        return flash_attn_with_kvcache(
            q.view(-1, self.block_size, q.shape[-2], q.shape[-1]),
            k,
            v,
            cache_seqlens=self.cache_seqlens,
            causal=False,
            softmax_scale=q.shape[-1] ** -0.5,
        ).view(q.shape)
