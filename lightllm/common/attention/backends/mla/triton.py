from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import torch

from lightllm.common.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionStepBuilder,
    AttentionMetadata,
    PrefillStateBuilder,
    StepBuilderContext,
    TensorAllocator,
)
from . import MLADecodeLayerInputs, MLADecodeLayerConfig, MLAPrefillLayerInputs, MLAPrefillLayerConfig

if TYPE_CHECKING:
    from lightllm.common.attention.kernels.mla.decode.gqa_flash_decoding import MLADecodeSchedule


class TritonMLAPrefillImpl(AttentionImpl[MLAPrefillLayerConfig, AttentionMetadata, MLAPrefillLayerInputs]):
    def __init__(self, layer_config: MLAPrefillLayerConfig) -> None:
        super().__init__(layer_config)
        # 256-dim queries (GLM-4 MoE Lite: 192 nope + 64 rope) run the contiguous-KV GQA prefill kernel
        # over keys expanded per head.
        self.expands_keys = layer_config.qk_nope_head_dim + layer_config.qk_rope_head_dim == 256
        if not layer_config.causal:
            raise NotImplementedError("Triton MLA prefill supports only causal attention")

    def forward(
        self,
        att_state: AttentionMetadata,
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

        assert att_state.b_q_start_loc is not None and att_state.cu_kv_seq_len is not None, "prefill fields required"
        assert att_state.b_ready_cache_len is not None, "prefill fields required"
        layer_config = self.layer_config
        nope_dim = layer_config.qk_nope_head_dim
        assert layer_inputs.q.shape[1:] == (
            layer_config.num_heads,
            nope_dim + layer_config.qk_rope_head_dim,
        ), "q does not match the layer_config"
        assert layer_inputs.q.dtype == layer_config.dtype, "q does not match the layer_config"
        if self.expands_keys:
            return self._expanded_key_forward(att_state, layer_inputs, out, alloc_func)
        context_attention_fwd_with_v(
            q_nope=layer_inputs.q[:, :, :nope_dim],
            q_rope=layer_inputs.q[:, :, nope_dim:],
            k_nope=layer_inputs.k_nope,
            k_rope=layer_inputs.k_rope,
            v=layer_inputs.v,
            o=out,
            b_start_loc=att_state.b_q_start_loc,
            b_kv_start_loc=att_state.cu_kv_seq_len,
            b_seq_len=att_state.b_seq_len,
            b_prompt_cache_len=att_state.b_ready_cache_len,
            max_input_len=att_state.max_q_seq_len,
            softmax_scale=layer_config.softmax_scale,
        )
        return out

    def _expanded_key_forward(self, att_state, layer_inputs: MLAPrefillLayerInputs, out, alloc_func):
        from lightllm.common.attention.kernels.gqa.bf16.prefill.context_flashattention_nopad import (
            context_attention_fwd_contiguous_kv,
        )

        layer_config, q = self.layer_config, layer_inputs.q
        nope_dim, heads = layer_config.qk_nope_head_dim, layer_config.num_heads
        k = alloc_func((layer_inputs.k_nope.shape[0], heads, q.shape[-1]), dtype=q.dtype, device=q.device)
        k[:, :, :nope_dim].copy_(layer_inputs.k_nope.expand(-1, heads, -1))
        k[:, :, nope_dim:].copy_(layer_inputs.k_rope.expand(-1, heads, -1))
        context_attention_fwd_contiguous_kv(
            q,
            k,
            layer_inputs.v,
            out,
            att_state.b_q_start_loc,
            att_state.cu_kv_seq_len,
            att_state.b_seq_len,
            att_state.max_q_seq_len,
            att_state.b_ready_cache_len,
            softmax_scale=layer_config.softmax_scale,
        )
        return out


@dataclass(frozen=True)
class TritonMLADecodeAttState:
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


class TritonMLADecodeStepBuilder(AttentionStepBuilder[TritonMLADecodeAttState]):
    def __init__(self, *, num_heads: int, kv_lora_rank: int, qk_rope_head_dim: int, dtype: torch.dtype) -> None:
        """Fix the layer group's per-rank configuration, which selects the split schedule's kernel config."""
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.dtype = dtype
        self._cuda_graph_schedules: dict[tuple[int, int], "MLADecodeSchedule"] = {}

    def build(self, batch: AttentionMetadata) -> TritonMLADecodeAttState:
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
        return TritonMLADecodeAttState(
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


class TritonMLADecodeImpl(AttentionImpl[MLADecodeLayerConfig, TritonMLADecodeAttState, MLADecodeLayerInputs]):
    def __init__(self, layer_config: MLADecodeLayerConfig) -> None:
        super().__init__(layer_config)
        # Each query row reads its own request prefix, which is causal across a request's rows.
        if not layer_config.causal:
            raise NotImplementedError("Triton MLA decode supports only causal attention")

    def forward(
        self,
        att_state: TritonMLADecodeAttState,
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
        assert (att_state.num_heads, att_state.kv_lora_rank, att_state.qk_rope_head_dim, att_state.dtype) == (
            layer_config.num_heads,
            layer_config.kv_lora_rank,
            layer_config.qk_rope_head_dim,
            layer_config.dtype,
        ), "attention state were built for a different MLA layer configuration"
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
            req_to_token_indexs=att_state.req_to_token_indexs,
            b_req_idx=att_state.b_req_idx,
            b_seq_len=att_state.b_seq_len,
            softmax_scale=layer_config.softmax_scale,
            schedule=att_state.schedule,
            out=out,
            alloc_tensor_func=alloc_func,
        )
        return out


class TritonMLAPrefillBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "triton_mla_prefill"

    @staticmethod
    def get_step_builder_cls() -> type[PrefillStateBuilder]:
        return PrefillStateBuilder

    @staticmethod
    def get_impl_cls() -> type[TritonMLAPrefillImpl]:
        return TritonMLAPrefillImpl

    @classmethod
    def create_step_builder(cls, layer_config: MLAPrefillLayerConfig, ctx: StepBuilderContext) -> PrefillStateBuilder:
        return PrefillStateBuilder()


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

    @classmethod
    def create_step_builder(
        cls, layer_config: MLADecodeLayerConfig, ctx: StepBuilderContext
    ) -> TritonMLADecodeStepBuilder:
        return TritonMLADecodeStepBuilder(**cls.step_builder_key(layer_config)._asdict())

    @classmethod
    def step_builder_key(cls, layer_config: MLADecodeLayerConfig) -> "_TritonMLADecodeKey":
        # The split schedule's kernel config depends on these.
        return _TritonMLADecodeKey(
            layer_config.num_heads, layer_config.kv_lora_rank, layer_config.qk_rope_head_dim, layer_config.dtype
        )


class _TritonMLADecodeKey(NamedTuple):
    num_heads: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    dtype: torch.dtype
