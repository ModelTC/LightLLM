from lightllm.core.formats.recurrent.spec import ConvSSMStateSpec
from lightllm.core.formats.kv.spec import IndexerWindowStateSpec
from lightllm.core.formats.kv.spec import PooledKVStateSpec
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from lightllm.models.glm5_next.layer_infer.pre_layer_infer import Glm5NextPreLayerInfer
from lightllm.models.glm5_next.layer_infer.transformer_layer_infer import Glm5NextTransformerLayerInfer
from lightllm.models.glm5_next.layer_weights.pre_and_post_layer_weight import Glm5NextPreAndPostLayerWeight
from lightllm.models.glm5_next.layer_weights.transformer_layer_weight import Glm5NextTransformerLayerWeight
from lightllm.models.base.adapter import RecipeModelAdapter
from lightllm.core.compute.attention.kda.spec import KimiDeltaAttentionSpec
from lightllm.core.compute.attention.pooled_nope.spec import PooledNoPEAttentionSpec
from lightllm.core.contracts.state import StateSpec, LayerStates
from copy import deepcopy


class Glm5NextModelAdapter(RecipeModelAdapter):
    pre_and_post_weight_class = Glm5NextPreAndPostLayerWeight
    transformer_weight_class = Glm5NextTransformerLayerWeight
    pre_layer_infer_class = Glm5NextPreLayerInfer
    post_layer_infer_class = LlamaPostLayerInfer
    transformer_layer_infer_class = Glm5NextTransformerLayerInfer

    def moe_specs(self):
        from lightllm.models.base.adapter import grouped_moe_specs

        return grouped_moe_specs(
            self.config, self.dtype, limit=self.config["swiglu_limit"], clamp_up_add_one=False, group_score_topk=2
        )

    def __init__(self, config, *, role, dtype, tp_world_size=1, layer_ids=None):
        self.config, self.dtype = deepcopy(config), dtype
        depth, linear = config["num_hidden_layers"], config["linear_attn_config"]
        kinds = config["layer_types"]
        if depth <= 0 or len(kinds) != depth or tp_world_size <= 0:
            raise ValueError("GLM requires one explicit attention declaration per layer")
        if linear["num_heads"] % tp_world_size or config["num_attention_heads"] % tp_world_size:
            raise ValueError("GLM attention requires complete TP head shards")
        if config["qk_rope_head_dim"] != 0:
            raise ValueError("GLM pooled attention requires NoPE semantics")
        self.layer_ids = tuple(range(depth)) if layer_ids is None else tuple(layer_ids)
        if len(self.layer_ids) != depth or len(set(self.layer_ids)) != depth or min(self.layer_ids) < 0:
            raise ValueError("GLM execution layer IDs must uniquely describe every model layer")
        self._attention, self._state_inputs, states = {}, {}, []
        for i, kind in zip(self.layer_ids, kinds, strict=True):
            if kind == "linear_attention":
                spec = KimiDeltaAttentionSpec(
                    linear["num_heads"] // tp_world_size,
                    linear["head_dim"],
                    linear["short_conv_kernel_size"],
                    dtype,
                    linear.get("gate_lower_bound", -5.0),
                )
                params = ConvSSMStateSpec(
                    key_heads=linear["num_heads"],
                    value_heads=linear["num_heads"],
                    key_dim=spec.head_dim,
                    value_dim=spec.head_dim,
                    conv_history_length=spec.conv_size - 1,
                )
                states.append(StateSpec(role, f"kda.{i}", i, (i,), params, "recurrent"))
                self._state_inputs[i] = LayerStates(states[-1])

            elif kind == "deepseek_sparse_attention":
                spec = PooledNoPEAttentionSpec(
                    config["num_attention_heads"] // tp_world_size,
                    config["kv_lora_rank"],
                    config["qk_nope_head_dim"] ** -0.5,
                    config["index_n_heads"],
                    config["index_head_dim"],
                    config["index_topk"],
                    config["index_kpool"],
                    dtype,
                )
                states.append(
                    StateSpec(
                        role,
                        f"pooled_kv.{i}",
                        i,
                        (i,),
                        PooledKVStateSpec(latent_dim=spec.latent_dim, index_dim=spec.index_dim),
                    )
                )
                states.append(
                    StateSpec(
                        role,
                        f"indexer_tail.{i}",
                        i,
                        (i,),
                        IndexerWindowStateSpec(window=spec.pool_size, stride=spec.pool_size, index_dim=spec.index_dim),
                        "window",
                    )
                )
                self._state_inputs[i] = LayerStates(states[-2], (("indexer_tail", states[-1]),))
            else:
                raise ValueError(f"unsupported GLM attention semantics: {kind}")
            self._attention[i] = spec
        self._states = tuple(states)

    def validate_layer(self, layer):
        spec = self._attention[layer.layer_num_]
        if layer.is_linear_attention_layer != isinstance(spec, KimiDeltaAttentionSpec):
            raise ValueError("GLM recipe differs from the declared attention family")
        if isinstance(spec, KimiDeltaAttentionSpec):
            actual, expected = (layer.tp_linear_num_heads, layer.linear_head_dim), (spec.heads, spec.head_dim)
        else:
            actual, expected = (layer.tp_q_head_num_, layer.kv_lora_rank), (spec.q_heads, spec.latent_dim)
        if actual != expected:
            raise ValueError("GLM recipe dimensions differ from the declared semantics")

    def attention_states(self):
        return self._state_inputs.copy()

    def layer_attention_params(self, spec, weight, device):
        from lightllm.models.base.attention_params import kda_params, pooled_nope_params

        if isinstance(spec, KimiDeltaAttentionSpec):
            return kda_params(spec, weight, device)
        return pooled_nope_params(spec, weight, device)
