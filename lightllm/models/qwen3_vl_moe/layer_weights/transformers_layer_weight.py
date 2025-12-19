import os
from lightllm.models.qwen3_moe.layer_weights.transformer_layer_weight import Qwen3MOETransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight, FusedMoeWeightEP, create_tp_moe_wegiht_obj


class Qwen3VLMOETransformerLayerWeight(Qwen3MOETransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, mode, quant_cfg)

    def load_hf_weights(self, weights):
        moe_prefix = f"model.layers.{self.layer_num_}.mlp.experts"
        gate_up_name = f"{moe_prefix}.gate_up_proj"
        down_name = f"{moe_prefix}.down_proj"

        if gate_up_name in weights:
            gate_up = weights[gate_up_name]  # [E, H, 2I]
            E, H, twoI = gate_up.shape
            assert twoI % 2 == 0, f"gate_up_proj last dim must be even, but got {twoI}"
            I_dim = twoI // 2

            if down_name in weights:
                down = weights[down_name]  # [E, I, H]
            else:
                down = None

            for e in range(E):
                gate_up_e = gate_up[e]
                gate_e = gate_up_e[:, :I_dim].transpose(0, 1).contiguous()
                up_e = gate_up_e[:, I_dim:].transpose(0, 1).contiguous()

                gate_key = f"{moe_prefix}.{e}.gate_proj.weight"
                up_key = f"{moe_prefix}.{e}.up_proj.weight"
                weights[gate_key] = gate_e
                weights[up_key] = up_e

                if down is not None:
                    down_key = f"{moe_prefix}.{e}.down_proj.weight"
                    weights[down_key] = down[e].transpose(0, 1).contiguous()

            del weights[gate_up_name]
            if down_name in weights:
                del weights[down_name]
        super().load_hf_weights(weights)
