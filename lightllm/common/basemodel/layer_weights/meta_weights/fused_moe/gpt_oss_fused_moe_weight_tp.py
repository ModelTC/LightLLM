import os
import torch
import threading
from typing import Optional, Tuple, List, Dict, Any

from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.fused_moe_weight import FusedMoeWeight
from lightllm.utils.dist_utils import get_current_rank_in_dp, get_current_device_id
from lightllm.common.quantization import Quantcfg
from lightllm.common.quantization.quantize_method import QuantizationMethod
from lightllm.utils.log_utils import init_logger
from lightllm.common.basemodel.routing_manager import g_routing_capture_manager

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


class GPTOSSFusedMoeWeightTP(FusedMoeWeight):
    def __init__(
        self,
        gate_up_proj_name: str,
        down_proj_name: str,
        e_score_correction_bias_name: str,
        weight_prefix: str,
        n_routed_experts: int,
        hidden_size: int,
        moe_intermediate_size: int,
        data_type: torch.dtype,
        quant_method: QuantizationMethod = None,
        num_fused_shared_experts: int = 0,
        layer_num: int = 0,
        network_config: Dict[str, Any] = None,
        moe_layer_index: int = 0,
    ) -> None:
        network_config["norm_topk_prob"] = None
        super().__init__(
            gate_proj_name=gate_up_proj_name,
            down_proj_name=down_proj_name,
            up_proj_name=gate_up_proj_name,
            e_score_correction_bias_name=e_score_correction_bias_name,
            weight_prefix=weight_prefix,
            n_routed_experts=n_routed_experts,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            data_type=data_type,
            quant_method=quant_method,
            num_fused_shared_experts=num_fused_shared_experts,
            layer_num=layer_num,
            network_config=network_config,
            moe_layer_index=moe_layer_index,
        )

        self.hidden_size = network_config["hidden_size"]

        self.alpha = 1.702
        self.limit = 7.0

        self.w1_bias = None
        self.w2_bias = None

        self._down_bias_name = f"{weight_prefix}.{down_proj_name}_bias"
        self._down_blocks_name = f"{weight_prefix}.{down_proj_name}_blocks"
        self._down_scales_name = f"{weight_prefix}.{down_proj_name}_scales"
        self._gate_up_bias_name = f"{weight_prefix}.{gate_up_proj_name}_bias"
        self._gate_up_blocks_name = f"{weight_prefix}.{gate_up_proj_name}_blocks"
        self._gate_up_scales_name = f"{weight_prefix}.{gate_up_proj_name}_scales"
        return

    def _create_weight(self):
        """
        因为加载方式比较特殊，不在这里创建weight。
        """
        pass

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

    def verify_load(self):
        assert self.w1 is not None and self.w2 is not None
        return True

    def _router(self, router_logits, top_k):
        router_top_value, router_indices = torch.topk(router_logits, top_k, dim=-1)
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        return router_top_value, router_indices

    def experts(
        self,
        input_tensor: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: int,
        num_expert_group: int,
        is_prefill: Optional[bool] = None,
        microbatch_index: int = 0,
    ):

        topk_weights, topk_ids = self._router(router_logits, top_k)

        # Rollout router replay
        if g_routing_capture_manager is not None:
            g_routing_capture_manager.capture(self.moe_layer_index, topk_ids, microbatch_index)

        w1, w1_scale = self.w1
        w2, w2_scale = self.w2
        use_fp8_w8a8 = self.quant_method is not None
        use_fp8_w8a8 = False  # TODO: disable fp8 for GPT-OSS for now

        from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe import fused_experts

        output_tensor = fused_experts(
            hidden_states=input_tensor.to(w1.dtype),
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

    def _cuda(self, cpu_tensor):
        return cpu_tensor.contiguous().to(self.data_type_).cuda(get_current_device_id())
