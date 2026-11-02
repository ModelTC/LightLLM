import weakref
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .base_att import BaseAttBackend, BaseAttState


class TorchCPUAttentionBackend(BaseAttBackend):
    supported_attention_features = ("window",)

    def bind_cache_writer(self, cache):
        k, v = cache
        if k.device.type != "cpu" or v.device.type != "cpu":
            raise ValueError("CPU attention requires CPU-native cache views")

        def write(*, token_indices, kv):
            heads = k.shape[1]
            k.index_copy_(0, token_indices.long(), kv[:, :heads])
            v.index_copy_(0, token_indices.long(), kv[:, heads:])

        return write

    def create_att_prefill_state(self, infer_state):
        return CPUAttentionMetadata(self, weakref.ref(infer_state))

    create_att_decode_state = create_att_prefill_state


@dataclass
class CPUAttentionMetadata(BaseAttState):
    def init_state(self):
        infer = self.infer_state
        if infer.b_req_idx.device.type != "cpu":
            raise ValueError("CPU metadata cannot synchronize GPU request indices")
        self.rows = []
        start = 0
        ready_lengths = infer.b_ready_cache_len
        if ready_lengths is None:
            ready_lengths = infer.b_seq_len - 1
        for request, length, ready in zip(infer.b_req_idx.tolist(), infer.b_seq_len.tolist(), ready_lengths.tolist()):
            queries = length - ready
            indices = infer.req_manager.req_to_token_indexs[request, :length].long()
            self.rows.append((start, queries, ready, indices))
            start += queries

    def prefill_att(self, q, k, v, att_control, alloc_func):
        if q.device.type != "cpu":
            raise ValueError("CPU attention requires CPU queries")
        outputs = []
        for start, count, ready, indices in self.rows:
            keys, values = k[indices].transpose(0, 1), v[indices].transpose(0, 1)
            query = q[start : start + count].transpose(0, 1)
            positions = torch.arange(indices.numel())[None, :]
            ends = ready + torch.arange(count)[:, None]
            mask = positions <= ends
            if att_control.use_sliding_window:
                mask &= positions >= ends - att_control.sliding_window[0]
            outputs.append(
                F.scaled_dot_product_attention(
                    query,
                    keys,
                    values,
                    attn_mask=mask,
                    enable_gqa=True,
                ).transpose(0, 1)
            )
        return torch.cat(outputs)

    decode_att = prefill_att
