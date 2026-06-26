from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight, QKVROWNMMWeight
from lightllm.models.qwen3_5.layer_weights.transformer_layer_weight import (
    Qwen35TransformerLayerWeight,
)
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class Qwen3_5MTPTransformerLayerWeight(Qwen35TransformerLayerWeight):
    # MTP draft-model weights live under the `mtp.layers.*` checkpoint namespace, so every
    # main-model layer name (`model.layers.*`) is retargeted to it at load time.

    _MAIN_PREFIX = "model.layers."
    _MTP_PREFIX = "mtp.layers."

    _ATTN_NORM_NAME_ATTRS = (
        "_q_weight_name",
        "_q_norm_name",
        "_q_bias_name",
        "_k_weight_name",
        "_k_norm_name",
        "_k_bias_name",
        "_v_weight_name",
        "_v_bias_name",
        "_kv_weight_name",
        "_kv_bias_name",
        "_o_weight_name",
        "_o_bias_name",
        "_att_norm_weight_name",
        "_att_norm_bias_name",
        "_ffn_norm_weight_name",
        "_ffn_norm_bias_name",
    )

    def _retarget(self, name):
        if name is None:
            return None
        return name.replace(self._MAIN_PREFIX, self._MTP_PREFIX, 1)

    def _retarget_attn_norm_names(self):
        for attr in self._ATTN_NORM_NAME_ATTRS:
            setattr(self, attr, self._retarget(getattr(self, attr)))

    def _init_qkv(self):
        in_dim = self.n_embed
        q_out_dim = self.q_head_num_ * self.head_dim
        self.qkv_proj = QKVROWNMMWeight(
            in_dim=in_dim,
            q_head_num=self.q_head_num_,
            kv_head_num=self.k_head_num_,
            head_dim=self.head_dim,
            weight_names=[self._q_weight_name, self._k_weight_name, self._v_weight_name],
            data_type=self.data_type_,
            bias_names=[self._q_bias_name, self._k_bias_name, self._v_bias_name],
            quant_method=self.get_quant_method("qkv_proj"),
        )
        self._o_gate_weight_name = f"{self._MTP_PREFIX}{self.layer_num_}.self_attn.o_gate_proj.weight"
        self._o_gate_proj = ROWMMWeight(
            in_dim=in_dim,
            out_dims=[q_out_dim],
            weight_names=[self._o_gate_weight_name],
            data_type=self.data_type_,
            bias_names=None,
            quant_method=self.get_quant_method("o_gate_proj"),
        )

    def _init_weight_names(self):
        super()._init_weight_names()
        # Retarget all main-model layer key names to the mtp.* namespace.
        self._retarget_attn_norm_names()
        # MLP (dense) projection names retargeted by Qwen35TransformerLayerWeight.
        self._gate_weight_name = self._retarget(self._gate_weight_name)
        self._gate_bias_name = self._retarget(self._gate_bias_name)
        self._up_weight_name = self._retarget(self._up_weight_name)
        self._up_bias_name = self._retarget(self._up_bias_name)
        self._gate_up_weight_name = self._retarget(self._gate_up_weight_name)
        self._gate_up_bias_name = self._retarget(self._gate_up_bias_name)
        self._down_weight_name = self._retarget(self._down_weight_name)
        self._down_bias_name = self._retarget(self._down_bias_name)
