import torch
from typing import Dict, Any

from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.fused_moe_weight_tp import FusedMoeWeightTP
from lightllm.common.quantization import Quantcfg
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


class GPTOSSFusedMoeWeightTP(FusedMoeWeightTP):
    def __init__(
        self,
        gate_up_proj_name: str,  # diff with FusedMoeWeightTP
        down_proj_name: str,
        e_score_correction_bias_name: str,
        weight_prefix: str,
        n_routed_experts: int,
        num_fused_shared_experts: int,
        split_inter_size: int,
        data_type: torch.dtype,
        network_config: Dict[str, Any],
        layer_num: int,
        world_size: int = 1,  # diff with FusedMoeWeightTP
        quant_cfg: Quantcfg = None,
    ) -> None:
        super().__init__(
            gate_up_proj_name,
            down_proj_name,
            gate_up_proj_name,
            e_score_correction_bias_name,
            weight_prefix,
            n_routed_experts,
            num_fused_shared_experts,
            split_inter_size,
            data_type,
            network_config,
            layer_num,
            quant_cfg,
        )
        self.hidden_size = network_config["hidden_size"]

        self.alpha = 1.702
        self.limit = 7.0
        self.tp_world_size_ = world_size

        self.w1_bias = None
        self.w2_bias = None

        self._down_bias_name = f"{weight_prefix}.{down_proj_name}_bias"
        self._down_blocks_name = f"{weight_prefix}.{down_proj_name}_blocks"
        self._down_scales_name = f"{weight_prefix}.{down_proj_name}_scales"
        self._gate_up_bias_name = f"{weight_prefix}.{gate_up_proj_name}_bias"
        self._gate_up_blocks_name = f"{weight_prefix}.{gate_up_proj_name}_blocks"
        self._gate_up_scales_name = f"{weight_prefix}.{gate_up_proj_name}_scales"
        return

    def _fuse_weight_scale(self):
        assert False, "Not implemented for GPT-OSS."

    def _fuse(self):
        assert False, "Not implemented for GPT-OSS."

    def load_hf_weights(self, weights):
        if (
            weights.get(self._down_blocks_name, None) is not None
            and weights.get(self._down_scales_name, None) is not None
        ):
            w2 = self._convert_moe_packed_tensors(
                blocks=weights[self._down_blocks_name],
                scales=weights[self._down_scales_name],
                dtype=torch.bfloat16,
            )[:, self.split_inter_size * self.tp_rank_ : self.split_inter_size * (self.tp_rank_ + 1), :]
            self.w2 = (self._cuda(w2.transpose(1, 2)), None)

        if (
            weights.get(self._gate_up_blocks_name, None) is not None
            and weights.get(self._gate_up_scales_name, None) is not None
        ):
            w1 = self._convert_moe_packed_tensors(
                blocks=weights[self._gate_up_blocks_name],
                scales=weights[self._gate_up_scales_name],
                dtype=torch.bfloat16,
            )[:, :, self.split_inter_size * self.tp_rank_ * 2 : self.split_inter_size * (self.tp_rank_ + 1) * 2]
            self.w1 = (self._cuda(w1.transpose(1, 2)), None)

        if weights.get(self._gate_up_bias_name, None) is not None:
            w1_bias = weights[self._gate_up_bias_name][
                :, self.split_inter_size * self.tp_rank_ * 2 : self.split_inter_size * (self.tp_rank_ + 1) * 2
            ]
            self.w1_bias = self._cuda(w1_bias)

        if weights.get(self._down_bias_name, None) is not None:
            w2_bias = weights[self._down_bias_name]
            self.w2_bias = self._cuda(w2_bias)

    def router(self, router_logits, top_k):
        router_top_value, router_indices = torch.topk(router_logits, top_k, dim=-1)
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        return router_top_value, router_indices

    def _native_forward(
        self, input_tensor, router_logits, top_k, renormalize, use_grouped_topk, topk_group, num_expert_group
    ):
        """PyTorch native implementation for GPT-OSS MoE forward pass."""
        topk_weights, topk_ids = self.router(router_logits, top_k)

        w1, w1_scale = self.w1
        w2, w2_scale = self.w2

        batch_size, hidden_size = input_tensor.shape

        output = torch.zeros_like(input_tensor)
        input_bf16 = input_tensor.to(torch.bfloat16)

        for i in range(batch_size):
            expert_output = torch.zeros(hidden_size, dtype=torch.bfloat16, device=input_tensor.device)
            for j in range(top_k):
                expert_idx = topk_ids[i, j].item()
                weight = topk_weights[i, j]

                w1_expert = w1[expert_idx]
                w2_expert = w2[expert_idx]

                x = input_bf16[i : i + 1]
                hidden = torch.mm(x, w1_expert.T)  # [1, intermediate_size * 2]
                if self.w1_bias is not None:
                    hidden = hidden + self.w1_bias[expert_idx : expert_idx + 1]

                gate = hidden[:, 0::2]
                up = hidden[:, 1::2]

                gate = torch.clamp(gate * self.alpha, -self.limit, self.limit)
                gate = torch.nn.functional.sigmoid(gate)
                hidden = gate * up

                expert_out = torch.mm(hidden, w2_expert.T)
                if self.w2_bias is not None:
                    expert_out = expert_out + self.w2_bias[expert_idx : expert_idx + 1] / self.tp_world_size_

                expert_output += weight * expert_out.squeeze(0)

            output[i] = expert_output

        input_tensor.copy_(output.to(input_tensor.dtype))
        return output

    def _cuda_forward(
        self, input_tensor, router_logits, top_k, renormalize, use_grouped_topk, topk_group, num_expert_group
    ):
        """CUDA optimized implementation for GPT-OSS MoE forward pass."""
        topk_weights, topk_ids = self.router(router_logits, top_k)

        w1, w1_scale = self.w1
        w2, w2_scale = self.w2
        use_fp8_w8a8 = self.quant_method is not None

        from lightllm.common.fused_moe.grouped_fused_moe import fused_experts

        output_tensor = fused_experts(
            hidden_states=input_tensor.to(torch.bfloat16),
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            use_fp8_w8a8=use_fp8_w8a8,
            w1_bias=self.w1_bias,
            w2_bias=self.w2_bias / self.tp_world_size_,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            layout="interleaved",
            alpha=self.alpha,
            limit=self.limit,
        )
        return output_tensor

    def experts(self, input_tensor, router_logits, top_k, renormalize, use_grouped_topk, topk_group, num_expert_group):
        """Backward compatible method that routes to platform-specific implementation."""
        return self._forward(
            input_tensor=input_tensor,
            router_logits=router_logits,
            top_k=top_k,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
        )

    def _convert_moe_packed_tensors(
        self,
        blocks,
        scales,
        *,
        dtype: torch.dtype = torch.bfloat16,
        rows_per_chunk: int = 32768 * 1024,
    ) -> torch.Tensor:
        """
        Convert the mxfp4 weights again, dequantizing and makes them compatible with the forward
        pass of GPT_OSS.
        """
        import math

        # Check if blocks and scales are on CPU, and move to GPU if so
        if not blocks.is_cuda and torch.cuda.is_available():
            blocks = blocks.cuda()
            scales = scales.cuda()

        scales = scales.to(torch.int32) - 127  # that's because 128=2**7

        assert blocks.shape[:-1] == scales.shape, f"{blocks.shape[:-1]=} does not match {scales.shape=}"

        lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

        *prefix_shape, G, B = blocks.shape
        rows_total = math.prod(prefix_shape) * G

        blocks = blocks.reshape(rows_total, B)
        scales = scales.reshape(rows_total, 1)

        out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

        for r0 in range(0, rows_total, rows_per_chunk):
            r1 = min(r0 + rows_per_chunk, rows_total)

            blk = blocks[r0:r1]
            exp = scales[r0:r1]

            # nibble indices -> int64
            idx_lo = (blk & 0x0F).to(torch.long)
            idx_hi = (blk >> 4).to(torch.long)

            sub = out[r0:r1]
            sub[:, 0::2] = lut[idx_lo]
            sub[:, 1::2] = lut[idx_hi]

            torch.ldexp(sub, exp, out=sub)
            del idx_lo, idx_hi, blk, exp, sub

        out = out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)
        del blocks, scales, lut
        return out.transpose(1, 2).contiguous()
