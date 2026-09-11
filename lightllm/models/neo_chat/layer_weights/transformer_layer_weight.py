from lightllm.common.basemodel.layer_weights.meta_weights import (
    QKRMSNORMWeight,
    QKVROWNMMWeight,
)
from lightllm.models.qwen3.layer_weights.transformer_layer_weight import (
    Qwen3TransformerLayerWeight,
)


class NeoChatTransformerLayerWeight(Qwen3TransformerLayerWeight):
    def _init_weight_names(self):
        super()._init_weight_names()
        prefix = f"model.layers.{self.layer_num_}.self_attn"
        self._q_norm_hw_name = f"{prefix}.q_norm_hw.weight"
        self._k_norm_hw_name = f"{prefix}.k_norm_hw.weight"

    def _init_qkv(self):
        self.qkv_proj = QKVROWNMMWeight(
            in_dim=self.n_embed,
            q_head_num=self.q_head_num_,
            kv_head_num=self.k_head_num_,
            head_dim=self.head_dim,
            weight_names=[
                self._q_weight_name,
                self._k_weight_name,
                self._v_weight_name,
            ],
            data_type=self.data_type_,
            bias_names=[self._q_bias_name, self._k_bias_name, self._v_bias_name],
            quant_method=self.get_quant_method("qkv_proj"),
        )

    def _init_norm(self):
        super()._init_norm()
        # Replace Qwen3's full-head Q/K norm with Neo's temporal half, then
        # add the independent spatial half.
        self.qk_norm_weight_ = QKRMSNORMWeight(
            dim=self.head_dim // 2,
            q_weight_name=self._q_norm_name,
            k_weight_name=self._k_norm_name,
            data_type=self.data_type_,
        )
        self.qk_hw_norm_weight_ = QKRMSNORMWeight(
            dim=self.head_dim // 2,
            q_weight_name=self._q_norm_hw_name,
            k_weight_name=self._k_norm_hw_name,
            data_type=self.data_type_,
        )
