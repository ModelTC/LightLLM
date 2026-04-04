from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.colmm_weight import COLMMWeight
from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.rowmm_weight import KVROWNMMWeight, ROWMMWeight
from lightllm.common.basemodel.layer_weights.meta_weights.norm_weight import RMSNormWeight
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight

"""
midlayer.hidden_norm.weight	[2048]	
midlayer.input_layernorm.weight	[2048]	
midlayer.mlp.down_proj.weight	[2048, 6144]	
midlayer.mlp.gate_proj.weight	[6144, 2048]	
midlayer.mlp.up_proj.weight	[6144, 2048]	
midlayer.post_attention_layernorm.weight	[2048]	
midlayer.self_attn.k_proj.weight	[512, 4096]	
midlayer.self_attn.o_proj.weight	[2048, 4096]	
midlayer.self_attn.q_proj.weight	[4096, 4096]	
midlayer.self_attn.v_proj.weight	[512, 4096]	
"""

class Qwen3EagleTransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, quant_cfg)
        hidden_size = self.network_config_["hidden_size"]
        self.fc_weight_ = ROWMMWeight(
            in_dim=hidden_size * 3,
            out_dims=[hidden_size],
            weight_names="fc.weight",
            quant_method=self.quant_cfg.get_quant_method(0, "fc"),
            data_type=self.data_type_,
            tp_rank=0,
            tp_world_size=1,
        )
        return

    def _init_weight_names(self):
        super()._init_weight_names()
        if self.network_config_["architectures"][0] == "Eagle3Speculator":
            weight_prefix = f"layers.0"
        else:
            weight_prefix = f"midlayer"
        self._q_weight_name = f"{weight_prefix}.self_attn.q_proj.weight"
        self._k_weight_name = f"{weight_prefix}.self_attn.k_proj.weight"
        self._v_weight_name = f"{weight_prefix}.self_attn.v_proj.weight"
        self._kv_weight_name = f"{weight_prefix}.self_attn.kv_proj.weight"
        self._o_weight_name = f"{weight_prefix}.self_attn.o_proj.weight"

        self._gate_weight_name = f"{weight_prefix}.mlp.gate_proj.weight"
        self._up_weight_name = f"{weight_prefix}.mlp.up_proj.weight"
        self._down_weight_name = f"{weight_prefix}.mlp.down_proj.weight"
        self._gate_up_bias_name = None

        self._att_norm_weight_name = f"{weight_prefix}.input_layernorm.weight"
        self._ffn_norm_weight_name = f"{weight_prefix}.post_attention_layernorm.weight"
        self._hidden_norm_weight_name = f"{weight_prefix}.hidden_norm.weight"
    
    def _init_qkv(self):
        in_dim = self.n_embed * 2
        q_out_dim = self.q_head_num_ * self.head_dim
        self.q_proj = ROWMMWeight(
            in_dim=in_dim,
            out_dims=[q_out_dim],
            weight_names=self._q_weight_name,
            data_type=self.data_type_,
            bias_names=self._q_bias_name,
            quant_method=self.get_quant_method("q_proj"),
        )
        self.kv_proj = KVROWNMMWeight(
            in_dim=in_dim,
            kv_head_num=self.k_head_num_,
            head_dim=self.head_dim,
            weight_names=[self._k_weight_name, self._v_weight_name],
            data_type=self.data_type_,
            bias_names=[self._k_bias_name, self._v_bias_name],
            quant_method=self.get_quant_method("kv_proj"),
        )

    def _init_o(self):
        in_dim = self.o_head_num_ * self.head_dim
        out_dim = self.n_embed
        self.o_proj = COLMMWeight(
            in_dim=in_dim,
            out_dims=[out_dim],
            weight_names=self._o_weight_name,
            data_type=self.data_type_,
            bias_names=self._o_bias_name,
            quant_method=self.get_quant_method("o_proj"),
        )

    def _init_ffn(self):
        self.gate_up_proj = ROWMMWeight(
            in_dim=self.n_embed,
            out_dims=[self.n_inter, self.n_inter],
            weight_names=[self._gate_weight_name, self._up_weight_name],
            data_type=self.data_type_,
            bias_names=[self._gate_bias_name, self._up_bias_name],
            quant_method=self.get_quant_method("gate_up_proj"),
        )
        self.down_proj = COLMMWeight(
            in_dim=self.n_inter,
            out_dims=[self.n_embed],
            weight_names=self._down_weight_name,
            data_type=self.data_type_,
            bias_names=self._down_bias_name,
            quant_method=self.get_quant_method("down_proj"),
        )

    def _init_norm(self):
        super()._init_norm()
        hidden_size = self.network_config_["hidden_size"]
        self.hidden_norm_weight_ = RMSNormWeight(
            dim=hidden_size,
            weight_name=self._hidden_norm_weight_name,
            data_type=self.data_type_,
        )
