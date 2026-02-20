import torch
import torch.distributed as dist
from typing import Tuple

from lightllm.models.qwen3next.layer_infer.transformer_layer_infer import (
    Qwen3NextFullAttentionTransformerLayerInfer,
    Qwen3NextGatedDeltaNetTransformerLayerInfer,
)
from lightllm.models.qwen3next.layer_weights.transformer_layer_weight import (
    Qwen3NextFullAttentionTransformerLayerWeight,
    Qwen3NextGatedDeltaNetTransformerLayerWeight,
)
from lightllm.models.qwen2_vl.triton_kernel.mrope import mrope_triton_fused
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class Qwen35FullAttentionTransformerLayerInfer(Qwen3NextFullAttentionTransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        # Initialize mrope section from config
        rope_scaling = network_config.get("rope_scaling", {})
        mrope_section = rope_scaling.get("mrope_section", [11, 11, 10])
        self.mrope_section = torch.tensor(mrope_section, dtype=torch.int32, device="cuda")

    def _get_qkv(
        self,
        input: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: Qwen3NextFullAttentionTransformerLayerWeight,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input = input.view(-1, self.embed_dim_)

        # Q and gate projection
        if not infer_state.is_prefill:
            q_gate_buf = self._get_decode_buffer(
                "q_gate_out",
                (self._graph_max_batch_size, self.tp_q_gate_dim),
                input.dtype,
                input.device,
            )[: input.size(0)]
            q_gate = layer_weight.q_gate_proj.mm(input, out=q_gate_buf)
            kv_buf = self._get_decode_buffer(
                "kv_out",
                (self._graph_max_batch_size, self.tp_kv_dim),
                input.dtype,
                input.device,
            )[: input.size(0)]
            kv_out = layer_weight.kv_proj.mm(input, out=kv_buf)
        else:
            q_gate = layer_weight.q_gate_proj.mm(input)
            kv_out = layer_weight.kv_proj.mm(input)

        q_dim = self.tp_q_head_num_ * self.head_dim_
        q = q_gate[:, :q_dim].contiguous()
        # In-place sigmoid for gate
        infer_state.gate_value = q_gate[:, q_dim:].sigmoid_()
        cache_kv = kv_out.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)

        # Q normalization (in-place)
        from lightllm.models.qwen3next.triton_kernel.gemma_rmsnorm import gemma_rmsnorm_forward

        gemma_rmsnorm_forward(
            q.view(-1, self.head_dim_),
            layer_weight.q_norm_weight_.weight,
            eps=self.eps_,
            out=q.view(-1, self.head_dim_),
        )

        k_input = cache_kv[:, : self.tp_k_head_num_, :].reshape(-1, cache_kv.shape[-1])
        if not infer_state.is_prefill:
            k_normed = self._get_decode_buffer(
                "k_norm_out",
                (self._graph_max_batch_size * self.tp_k_head_num_, cache_kv.shape[-1]),
                k_input.dtype,
                k_input.device,
            )[: k_input.shape[0]]
            gemma_rmsnorm_forward(k_input, layer_weight.k_norm_weight_.weight, eps=self.eps_, out=k_normed)
        else:
            k_normed = gemma_rmsnorm_forward(k_input, layer_weight.k_norm_weight_.weight, eps=self.eps_)
        cache_kv[:, : self.tp_k_head_num_, :] = k_normed.view(-1, self.tp_k_head_num_, cache_kv.shape[-1])

        if hasattr(infer_state, "position_cos") and infer_state.position_cos is not None:
            rotary_dim = int(self.head_dim_ * self.partial_rotary_factor)

            q_rotary = q.view(-1, self.tp_q_head_num_, self.head_dim_)[:, :, :rotary_dim].contiguous()
            k_rotary = cache_kv[:, : self.tp_k_head_num_, :rotary_dim].contiguous()

            mrope_triton_fused(
                q_rotary,
                k_rotary,
                infer_state.position_cos,
                infer_state.position_sin,
                self.mrope_section,
                is_interleaved=True,  # Qwen3 uses interleaved mrope
            )

            q.view(-1, self.tp_q_head_num_, self.head_dim_)[:, :, :rotary_dim] = q_rotary
            cache_kv[:, : self.tp_k_head_num_, :rotary_dim] = k_rotary
        else:
            from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd

            rotary_emb_fwd(
                q.view(-1, self.tp_q_head_num_, self.head_dim_),
                cache_kv[:, : self.tp_k_head_num_, :],
                infer_state.position_cos,
                infer_state.position_sin,
                partial_rotary_factor=self.partial_rotary_factor,
            )

        return q, cache_kv


class Qwen35GatedDeltaNetTransformerLayerInfer(Qwen3NextGatedDeltaNetTransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        rope_scaling = network_config.get("rope_scaling", {})
        mrope_section = rope_scaling.get("mrope_section", [11, 11, 10])
        self.mrope_section = torch.tensor(mrope_section, dtype=torch.int32, device="cuda")
