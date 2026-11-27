"""Triton MLA: prefill over expanded keys and values, decode over the latent cache with a split schedule."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from lightllm.common.attention.backend import AttentionBackend, TensorAllocator
from . import (
    MLADecodeLayerConfig,
    MLADecodeLayerInputs,
    MLAPrefillLayerConfig,
    MLAPrefillLayerInputs,
    expand_mla_keys,
)

if TYPE_CHECKING:
    from lightllm.common.basemodel.infer_struct import InferStateInfo
    from lightllm.common.attention.kernels.mla.decode.gqa_flash_decoding import MLADecodeSchedule


@dataclass(frozen=True)
class TritonMLAPrefillMetadata:
    """Packed prefill: request i's queries start at b_q_start_loc[i], its keys and values at
    cu_kv_seq_len[i], and b_ready_cache_len[i] of its b_seq_len[i] keys were cached before the step."""

    b_q_start_loc: torch.Tensor
    cu_kv_seq_len: torch.Tensor
    b_seq_len: torch.Tensor
    b_ready_cache_len: torch.Tensor
    max_q_seq_len: int


@dataclass(frozen=True)
class TritonMLADecodeMetadata:
    """At decode, request i reads physical slots req_to_token_indexs[b_req_idx[i], :b_seq_len[i]].

    The split schedule was built for the layer config's head count, latent/rotary dims and dtype.
    """

    req_to_token_indexs: torch.Tensor
    b_req_idx: torch.Tensor
    b_seq_len: torch.Tensor
    schedule: "MLADecodeSchedule"


class TritonMLAPrefillBackend(AttentionBackend[MLAPrefillLayerConfig, TritonMLAPrefillMetadata, MLAPrefillLayerInputs]):
    name = "triton_mla_prefill"
    phase = "prefill"

    @classmethod
    def _check_supported(cls, layer_config: MLAPrefillLayerConfig) -> None:
        if not layer_config.causal:
            raise NotImplementedError("Triton MLA prefill supports only causal attention")

    def build_att_metadata(self, infer_state: InferStateInfo) -> TritonMLAPrefillMetadata:
        return TritonMLAPrefillMetadata(
            b_q_start_loc=infer_state.b_q_start_loc,
            cu_kv_seq_len=infer_state.b1_cu_kv_seq_len,
            b_seq_len=infer_state.b_seq_len,
            b_ready_cache_len=infer_state.b_ready_cache_len,
            max_q_seq_len=infer_state.max_q_seq_len,
        )

    def forward(
        self,
        att_metadata: TritonMLAPrefillMetadata,
        layer_inputs: MLAPrefillLayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Write causal attention to out [query_tokens, num_heads, v_head_dim] and return it.

        Operands share dtype/device and have unit last stride. out must be disjoint from layer_inputs.
        """
        config, q = self.layer_config, layer_inputs.q
        nope_dim = config.qk_nope_head_dim
        assert q.shape[1:] == (
            config.num_heads,
            nope_dim + config.qk_rope_head_dim,
        ), "q does not match the layer_config"
        assert q.dtype == config.dtype, "q does not match the layer_config"

        if nope_dim + config.qk_rope_head_dim == 256:
            # 256-dim queries (GLM-4 MoE Lite: 192 nope + 64 rope) run the contiguous-KV GQA prefill
            # kernel over keys expanded per head.
            from lightllm.common.attention.kernels.gqa.bf16.prefill.context_flashattention_nopad import (
                context_attention_fwd_contiguous_kv,
            )

            context_attention_fwd_contiguous_kv(
                q,
                expand_mla_keys(layer_inputs, alloc_func),
                layer_inputs.v,
                out,
                att_metadata.b_q_start_loc,
                att_metadata.cu_kv_seq_len,
                att_metadata.b_seq_len,
                att_metadata.max_q_seq_len,
                att_metadata.b_ready_cache_len,
                softmax_scale=config.softmax_scale,
            )
        else:
            from lightllm.common.attention.kernels.mla.prefill.context_flashattention_nopad_with_v import (
                context_attention_fwd_with_v,
            )

            context_attention_fwd_with_v(
                q_nope=q[:, :, :nope_dim],
                q_rope=q[:, :, nope_dim:],
                k_nope=layer_inputs.k_nope,
                k_rope=layer_inputs.k_rope,
                v=layer_inputs.v,
                o=out,
                b_start_loc=att_metadata.b_q_start_loc,
                b_kv_start_loc=att_metadata.cu_kv_seq_len,
                b_seq_len=att_metadata.b_seq_len,
                b_prompt_cache_len=att_metadata.b_ready_cache_len,
                max_input_len=att_metadata.max_q_seq_len,
                softmax_scale=config.softmax_scale,
            )
        return out


class TritonMLADecodeBackend(AttentionBackend[MLADecodeLayerConfig, TritonMLADecodeMetadata, MLADecodeLayerInputs]):
    name = "triton_mla_decode"
    phase = "decode"

    def __init__(self, layer_config: MLADecodeLayerConfig) -> None:
        super().__init__(layer_config)
        self._graph_schedules: dict[tuple[int, int], "MLADecodeSchedule"] = {}

    @classmethod
    def _check_supported(cls, layer_config: MLADecodeLayerConfig) -> None:
        # Each query row reads its own request prefix, which is causal across a request's rows.
        if not layer_config.causal:
            raise NotImplementedError("Triton MLA decode supports only causal attention")

    @classmethod
    def metadata_config_key(cls, layer_config: MLADecodeLayerConfig):
        # The split schedule's kernel config depends on these.
        c = layer_config
        return c.num_heads, c.kv_lora_rank, c.qk_rope_head_dim, c.dtype

    def build_att_metadata(self, infer_state: InferStateInfo) -> TritonMLADecodeMetadata:
        """Compute the split schedule from the current b_seq_len, shared by every layer of the step.

        A CUDA graph step reuses one schedule per (microbatch, batch size) and rewrites it in place, so
        captured kernels read the replayed lengths' splits. Its kernel config is chosen from the graph
        capacity, so capture and replay agree; eager steps use the average length.
        """
        from lightllm.common.attention.kernels.mla.decode.gqa_flash_decoding import (
            plan_mla_decode_schedule,
            update_mla_decode_schedule,
        )

        key = (infer_state.microbatch_index, infer_state.batch_size)
        schedule = self._graph_schedules.get(key) if infer_state.uses_cuda_graph else None
        if schedule is not None:
            update_mla_decode_schedule(schedule, infer_state.b_seq_len)
        else:
            config = self.layer_config
            if infer_state.uses_cuda_graph:
                tuning_seq_len = infer_state.cuda_graph_max_kv_seq_len
            else:
                tuning_seq_len = infer_state.total_token_num // infer_state.batch_size
            schedule = plan_mla_decode_schedule(
                b_seq_len=infer_state.b_seq_len,
                q_head_num=config.num_heads,
                kv_lora_rank=config.kv_lora_rank,
                q_rope_dim=config.qk_rope_head_dim,
                dtype=config.dtype,
                tuning_seq_len=tuning_seq_len,
            )
            if infer_state.uses_cuda_graph:
                self._graph_schedules[key] = schedule

        return TritonMLADecodeMetadata(
            req_to_token_indexs=infer_state.req_to_token_indexs,
            b_req_idx=infer_state.b_req_idx,
            b_seq_len=infer_state.b_seq_len,
            schedule=schedule,
        )

    def forward(
        self,
        att_metadata: TritonMLADecodeMetadata,
        layer_inputs: MLADecodeLayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out with q_nope's [requests, num_heads, kv_lora_rank] shape, dtype and device.

        Inputs and cache are read-only. out has unit last stride and must not overlap them.
        """
        from lightllm.common.attention.kernels.mla.decode.gqa_flash_decoding import (
            gqa_token_decode_attention_flash_decoding,
        )

        config = self.layer_config
        q_nope, q_rope, kv, rank = layer_inputs.q_nope, layer_inputs.q_rope, layer_inputs.kv_cache, config.kv_lora_rank
        assert q_nope.shape[1:] == (config.num_heads, rank) and q_nope.dtype == config.dtype
        assert q_rope.shape[-1] == config.qk_rope_head_dim
        assert kv.shape[-1] == rank + config.qk_rope_head_dim
        gqa_token_decode_attention_flash_decoding(
            q_nope=q_nope,
            q_rope=q_rope,
            kv_nope=kv[:, :, :rank],
            kv_rope=kv[:, :, rank:],
            req_to_token_indexs=att_metadata.req_to_token_indexs,
            b_req_idx=att_metadata.b_req_idx,
            b_seq_len=att_metadata.b_seq_len,
            softmax_scale=config.softmax_scale,
            schedule=att_metadata.schedule,
            out=out,
            alloc_tensor_func=alloc_func,
        )
        return out
