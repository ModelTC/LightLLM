from lightllm.common.execution.formats.kv.spec import MLAKVStateSpec
from lightllm.models.deepseek2.layer_infer.transformer_layer_infer import Deepseek2TransformerLayerInfer
from lightllm.models.deepseek2.layer_weights.transformer_layer_weight import Deepseek2TransformerLayerWeight
from lightllm.models.base.adapter import RecipeModelAdapter
from lightllm.common.execution.compute.attention.mla.spec import MLAAttentionSpec
from lightllm.common.execution.contracts.state import StateSpec
from lightllm.models.llama.adapter import LlamaModelAdapter
from lightllm.models.llama.yarn_rotary_utils import get_deepseek_mscale
from copy import deepcopy


class Deepseek2ModelAdapter(RecipeModelAdapter):
    moe_group_score_topk = 1
    pre_and_post_weight_class = LlamaModelAdapter.pre_and_post_weight_class
    pre_layer_infer_class = LlamaModelAdapter.pre_layer_infer_class
    post_layer_infer_class = LlamaModelAdapter.post_layer_infer_class
    transformer_weight_class = Deepseek2TransformerLayerWeight
    transformer_layer_infer_class = Deepseek2TransformerLayerInfer

    def moe_specs(self):
        from lightllm.models.base.adapter import grouped_moe_specs

        return grouped_moe_specs(self.config, self.dtype, group_score_topk=self.moe_group_score_topk)

    def __init__(self, config, *, role, dtype, tp_world_size=1, layer_ids=None):
        self.config, self.dtype = deepcopy(config), dtype
        if config.get("model_type") in ("deepseek_v3", "deepseek_v32"):
            self.moe_group_score_topk = 2
        heads, depth = config["num_attention_heads"], config["num_hidden_layers"]
        if depth <= 0 or tp_world_size <= 0 or heads <= 0 or heads % tp_world_size:
            raise ValueError("MLA requires layers and whole query-head TP shards")
        scale = (config["qk_nope_head_dim"] + config["qk_rope_head_dim"]) ** -0.5
        rope = config.get("rope_scaling") or {}
        if rope.get("mscale_all_dim", 0):
            mscale = get_deepseek_mscale(rope["factor"], rope["mscale_all_dim"])
            scale = scale * mscale * mscale
        spec = MLAAttentionSpec(
            heads // tp_world_size,
            config["kv_lora_rank"],
            config["qk_nope_head_dim"],
            config["qk_rope_head_dim"],
            config["v_head_dim"],
            dtype,
            scale,
        )
        self.layer_ids = tuple(range(depth)) if layer_ids is None else tuple(layer_ids)
        if len(self.layer_ids) != depth or len(set(self.layer_ids)) != depth or min(self.layer_ids) < 0:
            raise ValueError("MLA execution layer IDs must uniquely describe every model layer")
        self._attention = {i: spec for i in self.layer_ids}
        self._states = tuple(
            StateSpec(role, f"mla.{i}", i, (i,), MLAKVStateSpec(latent_dim=spec.latent_dim, rope_dim=spec.rope_dim))
            for i in self.layer_ids
        )

    def validate_layer(self, layer):
        spec = self._attention[layer.layer_num_]
        if (
            layer.tp_q_head_num_,
            layer.kv_lora_rank,
            layer.qk_nope_head_dim,
            layer.qk_rope_head_dim,
            layer.v_head_dim,
            layer.softmax_scale,
        ) != (spec.q_heads, spec.latent_dim, spec.nope_dim, spec.rope_dim, spec.value_dim, spec.scale):
            raise ValueError("MLA recipe projections disagree with the declared semantics")

    def layer_attention_params(self, spec, weight, device):
        from lightllm.models.base.attention_params import mla_params

        return mla_params(spec, weight, device)
