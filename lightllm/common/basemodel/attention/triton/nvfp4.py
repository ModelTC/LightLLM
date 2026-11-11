"""Fused NVFP4 attention with tile-local KV decoding."""

import dataclasses
import weakref

import torch

from ..base_att import BaseAttBackend, BasePrefillAttState, BaseDecodeAttState, AttControl
from ...triton_kernel.mtp_utils import build_mtp_shared_group_markers


class NVFP4TritonAttBackend(BaseAttBackend):
    supports_parallel_block_decode = True

    def __init__(self, model):
        super().__init__(model)
        architecture = " ".join(model.config.get("architectures", [])).lower()
        module = type(model).__module__
        if any(name in architecture for name in ("gemma4", "gptoss", "bloom")) or any(
            f".models.{name}" in module for name in ("gemma4", "gpt_oss", "bloom")
        ):
            raise ValueError("NVFP4 currently supports Llama/Qwen GQA attention without ALiBi or attention sinks")
        dim = model.config.get("head_dim")
        if dim is not None and dim not in (16, 32, 64, 128, 256):
            raise ValueError("NVFP4 attention supports head dimensions 16, 32, 64, 128 and 256")

    def create_att_prefill_state(self, infer_state):
        return NVFP4PrefillAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))

    def create_att_decode_state(self, infer_state):
        return NVFP4DecodeAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))


@dataclasses.dataclass
class NVFP4PrefillAttState(BasePrefillAttState):
    def init_state(self):
        pass

    def prefill_att(self, q, k, v, att_control=AttControl(), alloc_func=torch.empty):
        assert not att_control.use_alibi and not att_control.use_att_sink, "NVFP4 ALiBi/attention sinks are unsupported"
        from ...triton_kernel.att.prefill_att.nvfp4 import context_attention_nvfp4

        out = alloc_func(q.shape, dtype=q.dtype, device=q.device)
        context_attention_nvfp4(
            q,
            k,
            v,
            out,
            self.infer_state.b_req_idx,
            self.infer_state.b_q_start_loc,
            self.infer_state.b_seq_len,
            self.infer_state.b_ready_cache_len,
            self.infer_state.req_manager.req_to_token_indexs,
            self.infer_state.max_q_seq_len,
            causal=self.backend.uses_causal_attention(),
            sliding_window=att_control.sliding_window if att_control.use_sliding_window else (-1, -1),
        )
        return out


@dataclasses.dataclass
class NVFP4DecodeAttState(BaseDecodeAttState):
    markers: torch.Tensor = None
    visible_lengths: torch.Tensor = None
    max_kv_len: int = None
    group_size: int = 1

    def init_state(self):
        model = self.backend.model
        self.group_size = model.mtp_manager.get_decode_tokens_per_request(model.is_mtp_draft_model)
        self.max_kv_len = self.infer_state.max_kv_seq_len
        if self.group_size > 1:
            self.markers = build_mtp_shared_group_markers(
                self.infer_state.b_req_idx, hold_req_id=model.req_manager.HOLD_REQUEST_ID
            )
        if self.backend.uses_causal_attention():
            self.visible_lengths = self.infer_state.b_seq_len
        else:
            assert self.infer_state.batch_size % self.group_size == 0, "noncausal decode requires complete query blocks"
            self.visible_lengths = (
                self.infer_state.b_seq_len.view(-1, self.group_size).amax(1).repeat_interleave(self.group_size)
            )

    def decode_att(self, q, k, v, att_control=AttControl(), alloc_func=torch.empty):
        assert not att_control.use_alibi and not att_control.use_att_sink, "NVFP4 ALiBi/attention sinks are unsupported"
        from ...triton_kernel.att.decode_att.nvfp4 import decode_attention_nvfp4

        return decode_attention_nvfp4(
            q,
            k,
            v,
            self.infer_state.req_manager.req_to_token_indexs,
            self.infer_state.b_req_idx,
            self.visible_lengths,
            self.infer_state.b_seq_len,
            self.max_kv_len,
            markers=self.markers,
            max_group_size=self.group_size,
            sliding_window=att_control.sliding_window if att_control.use_sliding_window else (-1, -1),
            alloc_func=alloc_func,
        )
