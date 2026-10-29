import weakref

import torch

from ..base_att import AttControl, BaseAttBackend, BaseDecodeAttState
from lightllm.utils.sgl_utils import flash_attn_with_kvcache


class WindowedMTPAttBackend(BaseAttBackend):
    """Non-causal block attention over retained draft KV and temporary noise KV."""

    def create_att_decode_state(self, infer_state):
        return WindowedMTPDecodeAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))


class WindowedMTPDecodeAttState(BaseDecodeAttState):
    def init_state(self):
        model = self.backend.model
        self.block_size = model.block_size
        self.b_req_idx = self.infer_state.b_req_idx[:: self.block_size].contiguous()
        # The base attention-state copy refreshes these tensors on graph replay.
        store = self.infer_state.mem_manager.windowed_draft_kv
        self.cache_seqlens = (
            store.req_to_seq_len.index_select(0, self.b_req_idx).clamp(max=store.capacity).to(torch.int32)
            + self.block_size
        )

    def prepare_kv(self, layer_index, noise_kv, alloc_func=torch.empty):
        """Pack retained history and the current block in the backend's KV layout."""
        scratch = self.infer_state.mem_manager.windowed_draft_kv.pack(
            layer_index - self.backend.model.draft_layer_start,
            self.b_req_idx,
            noise_kv,
            self.block_size,
            alloc_func=alloc_func,
        )
        head_num = noise_kv.shape[1] // 2
        return scratch[:, :, :head_num], scratch[:, :, head_num:]

    def decode_att(self, q, k, v, att_control: AttControl = AttControl(), alloc_func=torch.empty):
        return flash_attn_with_kvcache(
            q.view(-1, self.block_size, q.shape[-2], q.shape[-1]),
            k,
            v,
            cache_seqlens=self.cache_seqlens,
            causal=False,
            window_size=att_control.sliding_window if att_control.use_sliding_window else (-1, -1),
            softmax_scale=q.shape[-1] ** -0.5,
        ).view(q.shape)
