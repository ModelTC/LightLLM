from __future__ import annotations

import dataclasses

import torch

from lightllm.common.basemodel.attention.base_att import (
    AttControl,
    BaseAttBackend,
    BasePrefillAttState,
)
from lightllm.common.basemodel.attention.paged_prefix import (
    merge_attention_states,
    paged_prefix_attention,
    prepare_paged_prefix_attention,
)
from lightllm.utils.sgl_utils import (
    flash_attn_with_kvcache,
    get_scheduler_metadata,
    merge_state_v2,
)

from .fixed_suffix import fixed_suffix_attention
from .fp32 import fp32_paged_prefix_attention, merge_attention_states_fp32


class DenseSplitPrefillAttBackend(BaseAttBackend):
    """Consumer-owned dense prefix plus a transient fixed suffix."""

    def __new__(cls, *args, **kwargs):
        # The action expert is model-owned. Sharing BaseAttBackend's process
        # singleton would let a second model overwrite the first model here.
        return object.__new__(cls)

    def __init__(self, model):
        super().__init__(model=model)
        if model.data_type not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(
                f"split-prefill attention does not support {model.data_type}"
            )
        if model.data_type in (torch.float16, torch.bfloat16):
            missing = [
                name
                for name, operation in (
                    ("flash_attn_with_kvcache", flash_attn_with_kvcache),
                    ("get_scheduler_metadata", get_scheduler_metadata),
                    ("merge_state_v2", merge_state_v2),
                )
                if operation is None
            ]
            if missing:
                raise RuntimeError(
                    "FA3 split-prefill attention requires sgl_kernel operations: "
                    + ", ".join(missing)
                )

    def create_att_prefill_state(self, infer_state) -> "SplitPrefillAttState":
        return DenseSplitPrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state):
        raise NotImplementedError("split-prefill attention has no decode state")


@dataclasses.dataclass
class DenseSplitPrefillAttState(BasePrefillAttState):
    page_table: torch.Tensor | None = None
    cu_seqlens_q: torch.Tensor | None = None
    scheduler_metadata: object = None

    def init_state(self):
        infer_state = self.infer_state
        model = self.backend.model
        prefix = infer_state.prefix_kv
        if prefix.batch_size != infer_state.batch_size:
            raise ValueError("dense Prefix KV and action batch sizes differ")
        self.page_table = prefix.page_table
        if self.cu_seqlens_q is None:
            self.cu_seqlens_q = torch.arange(
                0,
                (infer_state.batch_size + 1) * infer_state.suffix_length,
                infer_state.suffix_length,
                dtype=torch.int32,
                device=self.page_table.device,
            )

        first_key = prefix.layer(0).key
        if model.data_type != torch.float32:
            self.scheduler_metadata = prepare_paged_prefix_attention(
                first_key,
                infer_state.prefix_lengths,
                self.cu_seqlens_q,
                query_heads=model.trans_layers_weight[0].local_q_heads,
                max_q_len=infer_state.suffix_length,
                max_prefix_len=self.page_table.shape[1],
            )

    def prefill_att(
        self,
        q: torch.Tensor,
        prefix_k: torch.Tensor,
        prefix_v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
        *,
        suffix_k: torch.Tensor,
        suffix_v: torch.Tensor,
    ) -> torch.Tensor:
        if att_control != AttControl():
            raise ValueError("split-prefill attention does not support AttControl")
        infer_state = self.infer_state
        if q.dtype == torch.float32:
            prefix_output, prefix_lse = fp32_paged_prefix_attention(
                q,
                prefix_k,
                prefix_v,
                self.page_table,
                infer_state.prefix_lengths,
                self.cu_seqlens_q,
                max_q_len=infer_state.suffix_length,
            )
        else:
            prefix_output, prefix_lse = paged_prefix_attention(
                q,
                prefix_k,
                prefix_v,
                self.page_table,
                infer_state.prefix_lengths,
                self.cu_seqlens_q,
                max_q_len=infer_state.suffix_length,
                scheduler_metadata=self.scheduler_metadata,
            )
        suffix_output, suffix_lse = fixed_suffix_attention(
            q,
            suffix_k,
            suffix_v,
            batch_size=infer_state.batch_size,
            suffix_length=infer_state.suffix_length,
            first_suffix_token_isolated=infer_state.first_suffix_token_isolated,
            alloc_func=alloc_func,
        )
        merge = (
            merge_attention_states_fp32
            if q.dtype == torch.float32
            else merge_attention_states
        )
        return merge(prefix_output, prefix_lse, suffix_output, suffix_lse)


# Transitional aliases keep old imports stable while new code names the dense
# ownership boundary explicitly.
SplitPrefillAttBackend = DenseSplitPrefillAttBackend
SplitPrefillAttState = DenseSplitPrefillAttState

__all__ = [
    "DenseSplitPrefillAttBackend",
    "DenseSplitPrefillAttState",
    "SplitPrefillAttBackend",
    "SplitPrefillAttState",
]
