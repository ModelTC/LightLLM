import os
import torch
import numpy as np

from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.rowmm_weight import ROWMMWeight
from lightllm.common.basemodel.layer_weights.meta_weights.norm_weight import NormWeight
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
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


class GptOssTransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(
        self,
        layer_num,
        data_type,
        network_config,
        mode=[],
        quant_cfg=None,
    ):
        super().__init__(layer_num, data_type, network_config, mode, quant_cfg)
        return

    def _init_moe(self):
        moe_mode = os.getenv("MOE_MODE", "TP")
        assert moe_mode in ["TP"], "For now, GPT-OSS type model only support MOE TP mode."
        self.moe_gate = ROWMMWeight(
            weight_name=self._router_weight_name,
            data_type=self.data_type_,
            layer_num=self.layer_num_,
            bias_name=self._router_bias_name,
            name="moe_gate",
            tp_rank=0,
            tp_world_size=1,
        )

        # Current defination of experts
        self.down_proj_bias = NormWeight(self._down_bias_name, torch.bfloat16)
        self.down_proj_weight_blocks = NormWeight(self._down_blocks_name, torch.uint8)
        self.down_proj_weight_scales = NormWeight(self._down_scales_name, torch.uint8)

        self.gate_up_proj_bias = NormWeight(self._gate_up_bias_name, torch.bfloat16)
        self.gate_up_proj_weight_blocks = NormWeight(self._gate_up_blocks_name, torch.uint8)
        self.gate_up_proj_weight_scales = NormWeight(self._gate_up_scales_name, torch.uint8)
        self.attn_sinks = NormWeight(self._attn_sink_name, torch.bfloat16)

    def load_hf_weights(self, weights):
        super().load_hf_weights(weights)
        self._post_weight_process()

    def _init_weight_names(self):
        super()._init_weight_names()

        # Sinks
        self._attn_sink_name = f"model.layers.{self.layer_num_}.self_attn.sinks"

        # Bias
        self._q_bias_name = f"model.layers.{self.layer_num_}.self_attn.q_proj.bias"
        self._k_bias_name = f"model.layers.{self.layer_num_}.self_attn.k_proj.bias"
        self._v_bias_name = f"model.layers.{self.layer_num_}.self_attn.v_proj.bias"
        self._o_bias_name = f"model.layers.{self.layer_num_}.self_attn.o_proj.bias"

        # MOE Layers
        # model.layers.0.mlp.experts.down_proj_bias	[32, 2 880]
        # model.layers.0.mlp.experts.down_proj_blocks	[32, 2 880, 90, 16]
        # model.layers.0.mlp.experts.down_proj_scales	[32, 2 880, 90]
        # model.layers.0.mlp.experts.gate_up_proj_bias	[32, 5 760]
        # model.layers.0.mlp.experts.gate_up_proj_blocks	[32, 5 760, 90, 16]
        # model.layers.0.mlp.experts.gate_up_proj_scales	[32, 5 760, 90]
        # model.layers.0.mlp.router.bias	[32]
        # model.layers.0.mlp.router.weight	[32, 2 880]

        self._router_bias_name = f"model.layers.{self.layer_num_}.mlp.router.bias"
        self._router_weight_name = f"model.layers.{self.layer_num_}.mlp.router.weight"

        self._down_bias_name = f"model.layers.{self.layer_num_}.mlp.experts.down_proj_bias"
        self._down_blocks_name = f"model.layers.{self.layer_num_}.mlp.experts.down_proj_blocks"
        self._down_scales_name = f"model.layers.{self.layer_num_}.mlp.experts.down_proj_scales"
        self._down_weight_name = None

        self._gate_up_bias_name = f"model.layers.{self.layer_num_}.mlp.experts.gate_up_proj_bias"
        self._gate_up_blocks_name = f"model.layers.{self.layer_num_}.mlp.experts.gate_up_proj_blocks"
        self._gate_up_scales_name = f"model.layers.{self.layer_num_}.mlp.experts.gate_up_proj_scales"
        self._gate_up_weight_name = None

    def _init_ffn(self):
        self._init_moe()

    def _post_weight_process(self):
        self.moe_intermediate_size = self.network_config_["intermediate_size"]
        self.split_inter_size = self.moe_intermediate_size // self.tp_world_size_

        if self.down_proj_weight_blocks.verify_load() and self.down_proj_weight_scales.verify_load():
            self.down_proj_weight = self._convert_moe_packed_tensors(
                blocks=self.down_proj_weight_blocks.weight,
                scales=self.down_proj_weight_scales.weight,
                dtype=torch.bfloat16,
            )[
                :, self.split_inter_size * self.tp_rank_ : self.split_inter_size * (self.tp_rank_ + 1), :
            ]  # (32, 1440, 2880)

        if self.gate_up_proj_weight_blocks.verify_load() and self.gate_up_proj_weight_scales.verify_load():
            self.gate_up_proj_weight = self._convert_moe_packed_tensors(
                blocks=self.gate_up_proj_weight_blocks.weight,
                scales=self.gate_up_proj_weight_scales.weight,
                dtype=torch.bfloat16,
            )  # (32, 2880, 5760)
            expert_num = self.gate_up_proj_weight.shape[0]
            self.gate_up_proj_weight = self.gate_up_proj_weight.reshape(expert_num, -1, 2, self.moe_intermediate_size)[
                :, :, :, self.split_inter_size * self.tp_rank_ : self.split_inter_size * (self.tp_rank_ + 1)
            ].reshape(
                expert_num, -1, 2 * self.split_inter_size
            )  # (32, 2880, 2880)

        if self.gate_up_proj_bias.verify_load():
            self.gate_up_proj_bias.weight = self.gate_up_proj_bias.weight.reshape(
                expert_num, 2, self.moe_intermediate_size
            )[:, :, self.split_inter_size * self.tp_rank_ : self.split_inter_size * (self.tp_rank_ + 1)].reshape(
                expert_num, 2 * self.split_inter_size
            )  # (32, 2880)

    def _convert_moe_packed_tensors(
        self,
        blocks,
        scales,
        *,
        dtype: torch.dtype = torch.bfloat16,
        rows_per_chunk: int = 32768 * 1024,  # TODO these values are not here by mistake ;)
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

        scales = scales.to(torch.int32) - 127  # TODO that's because 128=2**7

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
