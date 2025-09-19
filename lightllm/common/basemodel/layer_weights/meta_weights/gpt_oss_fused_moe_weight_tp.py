import os
import torch
import threading
from typing import Optional, Tuple, List, Dict, Any

from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe_weight_tp import FusedMoeWeightTP
from lightllm.utils.dist_utils import get_current_rank_in_dp, get_current_device_id
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
            self.w2 = (self._cuda(w2), None)

        if (
            weights.get(self._gate_up_blocks_name, None) is not None
            and weights.get(self._gate_up_scales_name, None) is not None
        ):
            w1 = self._convert_moe_packed_tensors(
                blocks=weights[self._gate_up_blocks_name],
                scales=weights[self._gate_up_scales_name],
                dtype=torch.bfloat16,
            )[:, :, self.split_inter_size * self.tp_rank_ * 2 : self.split_inter_size * (self.tp_rank_ + 1) * 2]
            self.w1 = (self._cuda(w1), None)

        if weights.get(self._gate_up_bias_name, None) is not None:
            w1_bias = weights[self._gate_up_bias_name][
                :, self.split_inter_size * self.tp_rank_ * 2 : self.split_inter_size * (self.tp_rank_ + 1) * 2
            ]
            self.w1_bias = self._cuda(w1_bias)

        if weights.get(self._down_bias_name, None) is not None:
            w2_bias = weights[self._down_bias_name]
            self.w2_bias = self._cuda(w2_bias)

    def experts(self, hidden_states: torch.Tensor, routing_weights, layer_num):
        w1, w1_scale = self.w1
        w2, w2_scale = self.w2
        assert w1_scale is None and w2_scale is None, "For now, we do not support quantized weight in GPT-OSS."

        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
        num_experts = routing_weights.shape[1]

        hidden_states = hidden_states.repeat(num_experts, 1)
        hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)
        gate_up = torch.bmm(hidden_states, w1) + self.w1_bias[..., None, :]
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        next_states = torch.bmm(((up + 1) * glu), w2)
        next_states = next_states + self.w2_bias[..., None, :] / self.tp_world_size_
        next_states = next_states.view(num_experts, batch_size, -1, self.hidden_size)
        next_states = next_states * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
        next_states = next_states.sum(dim=0)
        return next_states

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
