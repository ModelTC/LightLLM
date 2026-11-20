from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from lightllm.common.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionStepBuilder,
    AttentionBatch,
    TensorAllocator,
)
from . import MLADecodeLayerInputs, MLADecodeLayerConfig, MLAPrefillLayerInputs, MLAPrefillLayerConfig

if TYPE_CHECKING:
    from lightllm.common.attention.kernels.mla.decode.gqa_flash_decoding import MLADecodeSchedule


@dataclass(frozen=True)
class TritonMLAPrefillStepArgs:
    """Packed query and KV row offsets, total lengths and cached prefix lengths per request."""

    b_q_start_loc: torch.Tensor
    b_kv_start_loc: torch.Tensor
    b_seq_len: torch.Tensor
    b_ready_cache_len: torch.Tensor
    max_q_seq_len: int


class TritonMLAPrefillStepBuilder(AttentionStepBuilder[TritonMLAPrefillStepArgs]):
    def build(self, batch: AttentionBatch) -> TritonMLAPrefillStepArgs:
        """The kernel reads the request tensors directly, so the step args only record them."""
        assert batch.b_q_start_loc is not None and batch.cu_kv_seq_len is not None, "prefill batch fields required"
        assert batch.b_ready_cache_len is not None, "prefill batch fields required"
        return TritonMLAPrefillStepArgs(
            b_q_start_loc=batch.b_q_start_loc,
            b_kv_start_loc=batch.cu_kv_seq_len,
            b_seq_len=batch.b_seq_len,
            b_ready_cache_len=batch.b_ready_cache_len,
            max_q_seq_len=batch.max_q_seq_len,
        )


class TritonMLAPrefillImpl(AttentionImpl[MLAPrefillLayerConfig, TritonMLAPrefillStepArgs, MLAPrefillLayerInputs]):
    def forward(
        self,
        step_args: TritonMLAPrefillStepArgs,
        layer_inputs: MLAPrefillLayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Write causal attention to out [query_tokens, num_heads, v_head_dim] and return it.

        Operands share dtype/device and have unit last stride. out must be disjoint from layer_inputs.
        """
        from lightllm.common.attention.kernels.mla.prefill.context_flashattention_nopad_with_v import (
            context_attention_fwd_with_v,
        )

        layer_config = self.layer_config
        nope_dim = layer_config.qk_nope_head_dim
        assert layer_inputs.q.shape[1:] == (
            layer_config.num_heads,
            nope_dim + layer_config.qk_rope_head_dim,
        ), "q does not match the layer_config"
        assert layer_inputs.q.dtype == layer_config.dtype, "q does not match the layer_config"
        context_attention_fwd_with_v(
            q_nope=layer_inputs.q[:, :, :nope_dim],
            q_rope=layer_inputs.q[:, :, nope_dim:],
            k_nope=layer_inputs.k_nope,
            k_rope=layer_inputs.k_rope,
            v=layer_inputs.v,
            o=out,
            b_start_loc=step_args.b_q_start_loc,
            b_kv_start_loc=step_args.b_kv_start_loc,
            b_seq_len=step_args.b_seq_len,
            b_prompt_cache_len=step_args.b_ready_cache_len,
            max_input_len=step_args.max_q_seq_len,
            softmax_scale=layer_config.softmax_scale,
        )
        return out


@dataclass(frozen=True)
class TritonMLADecodeStepArgs:
    """At decode, request i reads physical slots req_to_token_indexs[b_req_idx[i], :b_seq_len[i]].

    The split schedule was built for the recorded head count, latent/rotary dims and dtype.
    """

    req_to_token_indexs: torch.Tensor
    b_req_idx: torch.Tensor
    b_seq_len: torch.Tensor
    schedule: "MLADecodeSchedule"
    num_heads: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    dtype: torch.dtype


class TritonMLADecodeStepBuilder(AttentionStepBuilder[TritonMLADecodeStepArgs]):
    def __init__(self, *, num_heads: int, kv_lora_rank: int, qk_rope_head_dim: int, dtype: torch.dtype) -> None:
        """Fix the layer group's per-rank configuration, which selects the split schedule's kernel config."""
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.dtype = dtype
        self._cuda_graph_schedules: dict[tuple[int, int], "MLADecodeSchedule"] = {}

    def build(self, batch: AttentionBatch) -> TritonMLADecodeStepArgs:
        """Compute the split schedule from the current b_seq_len, shared by every layer of the step.

        A CUDA graph step reuses one schedule per (microbatch, batch size) and rewrites it in
        place, so captured kernels read the replayed lengths' splits. Its kernel config is chosen
        from the graph capacity, so capture and replay agree; eager steps use the average length.
        """
        from lightllm.common.attention.kernels.mla.decode.gqa_flash_decoding import update_mla_decode_schedule

        if batch.uses_cuda_graph:
            key = (batch.microbatch_index, batch.batch_size)
            schedule = self._cuda_graph_schedules.get(key)
            if schedule is None:
                schedule = self._new_schedule(batch.b_seq_len, batch.cuda_graph_max_kv_seq_len)
                self._cuda_graph_schedules[key] = schedule
            else:
                update_mla_decode_schedule(schedule, batch.b_seq_len)
        else:
            tuning_seq_len = batch.total_token_num // batch.batch_size
            schedule = self._new_schedule(batch.b_seq_len, tuning_seq_len)
        return TritonMLADecodeStepArgs(
            req_to_token_indexs=batch.req_to_token_indexs,
            b_req_idx=batch.b_req_idx,
            b_seq_len=batch.b_seq_len,
            schedule=schedule,
            num_heads=self.num_heads,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            dtype=self.dtype,
        )

    def _new_schedule(self, b_seq_len: torch.Tensor, tuning_seq_len: int) -> "MLADecodeSchedule":
        from lightllm.common.attention.kernels.mla.decode.gqa_flash_decoding import plan_mla_decode_schedule

        return plan_mla_decode_schedule(
            b_seq_len=b_seq_len,
            q_head_num=self.num_heads,
            kv_lora_rank=self.kv_lora_rank,
            q_rope_dim=self.qk_rope_head_dim,
            dtype=self.dtype,
            tuning_seq_len=tuning_seq_len,
        )


class TritonMLADecodeImpl(AttentionImpl[MLADecodeLayerConfig, TritonMLADecodeStepArgs, MLADecodeLayerInputs]):
    def forward(
        self,
        step_args: TritonMLADecodeStepArgs,
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

        layer_config = self.layer_config
        assert (step_args.num_heads, step_args.kv_lora_rank, step_args.qk_rope_head_dim, step_args.dtype) == (
            layer_config.num_heads,
            layer_config.kv_lora_rank,
            layer_config.qk_rope_head_dim,
            layer_config.dtype,
        ), "step args were built for a different MLA layer configuration"
        q_nope, q_rope = layer_inputs.q_nope, layer_inputs.q_rope
        assert (
            q_nope.shape[1:] == (layer_config.num_heads, layer_config.kv_lora_rank)
            and q_nope.dtype == layer_config.dtype
        )
        assert q_rope.shape[-1] == layer_config.qk_rope_head_dim
        assert layer_inputs.kv_cache.shape[-1] == layer_config.kv_lora_rank + layer_config.qk_rope_head_dim
        gqa_token_decode_attention_flash_decoding(
            q_nope=q_nope,
            q_rope=q_rope,
            kv_nope=layer_inputs.kv_cache[:, :, : layer_config.kv_lora_rank],
            kv_rope=layer_inputs.kv_cache[:, :, layer_config.kv_lora_rank :],
            req_to_token_indexs=step_args.req_to_token_indexs,
            b_req_idx=step_args.b_req_idx,
            b_seq_len=step_args.b_seq_len,
            softmax_scale=layer_config.softmax_scale,
            schedule=step_args.schedule,
            out=out,
            alloc_tensor_func=alloc_func,
        )
        return out


class TritonMLAPrefillBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "triton_mla_prefill"

    @staticmethod
    def get_step_builder_cls() -> type[TritonMLAPrefillStepBuilder]:
        return TritonMLAPrefillStepBuilder

    @staticmethod
    def get_impl_cls() -> type[TritonMLAPrefillImpl]:
        return TritonMLAPrefillImpl


class TritonMLADecodeBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "triton_mla_decode"

    @staticmethod
    def get_step_builder_cls() -> type[TritonMLADecodeStepBuilder]:
        return TritonMLADecodeStepBuilder

    @staticmethod
    def get_impl_cls() -> type[TritonMLADecodeImpl]:
        return TritonMLADecodeImpl
