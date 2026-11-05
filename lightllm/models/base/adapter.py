"""Model declarations contain semantics, not allocated storage or backend objects."""

from lightllm.common.execution.contracts.model import ModelWeights
from lightllm.common.execution.contracts.model import ModelRecipe
from lightllm.common.execution.formats.recurrent.spec import ConvSSMStateSpec
from lightllm.common.execution.formats.kv.spec import DenseKVStateSpec


from copy import deepcopy
from typing import Any, Protocol

from lightllm.common.execution.compute.attention.dense.spec import DenseAttentionSpec
from lightllm.common.execution.compute.attention.gdn.spec import GatedDeltaNetSpec
from lightllm.common.execution.compute.attention.mla.spec import MLAAttentionSpec
from lightllm.common.execution.contracts.state import StateSpec, LayerStates
from lightllm.common.execution.compute.moe.spec import MoeSpec


class RecipeModelAdapter:
    attention_phases = ("prefill", "decode")
    pre_and_post_weight_class = None
    transformer_weight_class = None
    pre_layer_infer_class = None
    post_layer_infer_class = None
    transformer_layer_infer_class = None

    def moe_specs(self) -> dict[int, MoeSpec]:
        return {}

    def attention_specs(self) -> dict[int, Any]:
        return self._attention.copy()

    def state_specs(self) -> tuple[StateSpec, ...]:
        return self._states

    def attention_states(self) -> dict[int, LayerStates]:
        inputs = {}
        for state in self._states:
            for consumer in state.consumers:
                if consumer in inputs:
                    raise ValueError("models with multiple state inputs must declare their roles explicitly")
                inputs[consumer] = LayerStates(state)
        return inputs

    def attention_params(self, weights, device):
        return {i: self.layer_attention_params(spec, weights[i], device) for i, spec in self._attention.items()}

    def build_weights(self, quant_config) -> ModelWeights:
        """Weight classes own checkpoint names, TP mapping, quantization and tied weights."""
        pre_post = self.pre_and_post_weight_class(self.dtype, network_config=self.config)
        layers = [
            self.transformer_weight_class(i, self.dtype, network_config=self.config, quant_cfg=quant_config)
            for i in self.layer_ids
        ]
        return ModelWeights(pre_post, tuple(layers))

    def build_recipe(self) -> ModelRecipe:
        pre = self.pre_layer_infer_class(network_config=self.config)
        post = self.post_layer_infer_class(network_config=self.config)
        layers = [self.transformer_layer_infer_class(i, network_config=self.config) for i in self.layer_ids]
        for layer in layers:
            self.validate_layer(layer)
        return ModelRecipe(pre, post, tuple(layers))

    def load_weights(self, weights, directory, weight_dict=None):
        from lightllm.common.basemodel.layer_weights.hf_load_utils import load_hf_weights

        if weight_dict is not None and not isinstance(weight_dict, dict):
            raise TypeError("weight_dict must be a dict or None")
        pre_post, layers = weights.pre_post, weights.layers
        layers = list({id(layer): layer for layer in layers}.values())
        load_hf_weights(
            data_type=self.dtype,
            weight_dir=directory,
            pre_post_layer=pre_post,
            transformer_layer_list=layers,
            weight_dict=weight_dict,
        )
        pre_post.verify_load()
        for layer in layers:
            layer.verify_load()


class DenseModelAdapter(RecipeModelAdapter):
    def __init__(self, config, *, role, dtype, tp_world_size=1, layer_ids=None):
        self.config = deepcopy(config)
        self.dtype = dtype
        depth = config["num_hidden_layers"]
        q_heads, kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
        if depth <= 0 or tp_world_size <= 0 or q_heads <= 0 or q_heads % tp_world_size:
            raise ValueError("Dense model requires layers and complete TP query shards")
        if kv_heads <= 0 or (kv_heads % tp_world_size and tp_world_size % kv_heads):
            raise ValueError("Dense KV heads require complete TP shards or replicas")
        head_dim = config.get("head_dim", config["hidden_size"] // q_heads)
        self.layer_ids = tuple(range(depth)) if layer_ids is None else tuple(layer_ids)
        if len(self.layer_ids) != depth or len(set(self.layer_ids)) != depth or min(self.layer_ids) < 0:
            raise ValueError("execution layer IDs must uniquely describe every model layer")
        self._attention = {}
        layer_types = config.get("layer_types")
        if layer_types is not None and len(layer_types) != depth:
            raise ValueError("layer_types must describe every model layer")
        for offset, layer in enumerate(self.layer_ids):
            kind = "full_attention"
            if layer_types is not None:
                kind = layer_types[offset]
            elif config.get("use_sliding_window") and offset >= config.get("max_window_layers", 0):
                kind = "sliding_attention"
            window = (-1, -1)
            if kind == "sliding_attention":
                width = config.get("sliding_window")
                if not isinstance(width, int) or width <= 0:
                    raise ValueError("sliding attention requires a positive window size")
                window = (width - 1, 0)
            elif kind != "full_attention":
                raise ValueError(f"Dense adapter cannot describe layer semantics: {kind}")
            self._attention[layer] = DenseAttentionSpec(
                q_heads // tp_world_size,
                max(kv_heads // tp_world_size, 1),
                head_dim,
                window=window,
                dtype=dtype,
            )
        self._states = tuple(
            StateSpec(
                role,
                f"kv.{layer}",
                layer,
                (layer,),
                DenseKVStateSpec(kv_heads=kv_heads, head_dim=spec.head_dim),
            )
            for layer, spec in self._attention.items()
        )

    def layer_attention_params(self, spec, weight, device):
        from lightllm.models.base.attention_params import dense_params

        return dense_params(spec, weight, device)

    def validate_layer(self, layer):
        spec = self._attention[layer.layer_num_]
        if isinstance(spec, GatedDeltaNetSpec):
            if not layer.is_linear_attention_layer:
                raise ValueError("model recipe does not implement the declared recurrent layer")
            return
        if getattr(layer, "is_linear_attention_layer", False):
            raise ValueError("model recipe does not implement the declared Dense layer")
        if (layer.tp_q_head_num_, layer.tp_k_head_num_, layer.head_dim_) != (
            spec.q_heads,
            spec.kv_heads,
            spec.head_dim,
        ):
            raise ValueError("model recipe dimensions disagree with the declared attention semantics")


class GDNModelAdapter(DenseModelAdapter):
    def __init__(self, config, *, role, dtype, tp_world_size=1):
        from lightllm.models.qwen3next.config import is_linear_attention_layer

        dense_config = dict(config, layer_types=["full_attention"] * config["num_hidden_layers"])
        super().__init__(dense_config, role=role, dtype=dtype, tp_world_size=tp_world_size)
        self.config = deepcopy(config)
        kh, vh = config["linear_num_key_heads"], config["linear_num_value_heads"]
        if kh % tp_world_size or vh % tp_world_size:
            raise ValueError("GDN projections require whole key and value head TP shards")
        spec = GatedDeltaNetSpec(
            kh // tp_world_size,
            vh // tp_world_size,
            config["linear_key_head_dim"],
            config["linear_value_head_dim"],
            config["linear_conv_kernel_dim"],
            dtype,
            config.get("hidden_act", "silu"),
        )
        states = []
        for state in self._states:
            if is_linear_attention_layer(config, state.producer):
                self._attention[state.producer] = spec
                state = StateSpec(
                    role,
                    f"gdn.{state.producer}",
                    state.producer,
                    state.consumers,
                    ConvSSMStateSpec(
                        key_heads=kh,
                        value_heads=vh,
                        key_dim=spec.key_dim,
                        value_dim=spec.value_dim,
                        conv_history_length=spec.conv_size - 1,
                    ),
                    "recurrent",
                )
            states.append(state)
        self._states = tuple(states)

    def layer_attention_params(self, spec, weight, device):
        if isinstance(spec, GatedDeltaNetSpec):
            from lightllm.models.base.attention_params import gdn_params

            return gdn_params(spec, weight, device)
        return super().layer_attention_params(spec, weight, device)


def grouped_moe_specs(config, dtype, *, limit=None, clamp_up_add_one=True, layer_ids=None, group_score_topk=1):
    """DeepSeek/GLM recipe semantics, before weights or execution resources exist."""
    experts = config.get("n_routed_experts")
    if not experts:
        return {}
    layers = tuple(
        layer
        for layer in (range(config["num_hidden_layers"]) if layer_ids is None else layer_ids)
        if layer >= config["first_k_dense_replace"] and layer % config.get("moe_layer_freq", 1) == 0
    )
    if not layers:
        return {}
    groups, selected, top_k = config.get("n_group", 0), config.get("topk_group", 0), config["num_experts_per_tok"]
    method = config.get("topk_method")
    if method == "greedy":
        groups, selected, group_score_topk = 1, 1, 1
    elif method == "group_limited_greedy":
        group_score_topk = 1
    elif method == "noaux_tc":
        group_score_topk = 2
    elif method is not None:
        raise ValueError(f"unsupported MoE routing semantics: {method}")
    spec = MoeSpec(
        config["hidden_size"],
        config["moe_intermediate_size"],
        experts,
        top_k,
        dtype,
        renormalize=config.get("norm_topk_prob", False),
        scoring=config.get("scoring_func", "softmax"),
        groups=groups or 1,
        selected_groups=selected if groups else 1,
        group_score_topk=group_score_topk,
        routed_scale=config.get("routed_scaling_factor", 1.0),
        shared_experts=config.get("n_shared_experts") or 0,
        limit=limit,
        clamp_up_add_one=clamp_up_add_one,
    )
    return {layer: spec for layer in layers}
