from dataclasses import dataclass, replace

from lightllm.common.attention.api import AttentionFamily, AttentionPhase, CacheLayout, LayerSpec
from lightllm.common.attention.api.inputs import DenseResources, GDNResources, KDAResources, MLAResources
from lightllm.common.attention.api.spec import (
    AttentionFeature,
    CacheEncoding,
    GDNConfig,
    KDAConfig,
    MLAConfig,
    SparseMLAConfig,
)


@dataclass(frozen=True)
class AttentionDescription:
    family: AttentionFamily = AttentionFamily.DENSE
    features: frozenset[AttentionFeature] = frozenset()
    window: tuple[int, int] | None = None
    variant: str = ""


class AttentionAdapter:
    def describe(self, model, layer, phase, description):
        heads, kv_heads, head_dim, value_dim = self.dimensions(model, layer)
        state = model.att_state_manager.get_state(model.att_state_role, self.state_type, layer.layer_num_)
        states = tuple(
            state
            for state in (
                *model.att_state_manager.get_full_attention_states(model.att_state_role),
                *model.att_state_manager.get_bounded_attention_states(model.att_state_role),
            )
            if state.contains_logical_layer(layer.layer_num_)
        )
        features = set(description.features)
        draft_step = model.mtp_manager.get_decode_draft_step(model.is_mtp_draft_model)
        if draft_step > 0 and phase is AttentionPhase.DECODE:
            features.add(AttentionFeature.MTP)
        if draft_step > 0 and not model.is_mtp_draft_model and model.args.mtp_dynamic_verify:
            features.add(AttentionFeature.DYNAMIC_VERIFY)
        return LayerSpec(
            family=description.family,
            phase=phase,
            dtype=model.data_type,
            cache=self.cache(model, state, states, kv_heads, head_dim, value_dim),
            num_q_heads=heads,
            head_size=head_dim,
            value_head_size=value_dim,
            sliding_window=description.window,
            features=frozenset(features),
            variant=description.variant,
            softmax_scale=self.scale(layer),
            config=self.computation_config(model, layer, state),
        )

    def cache(self, model, state, states, heads, head_dim, value_dim):
        return CacheLayout(
            encoding=CacheEncoding.from_option(model.args.llm_kv_type),
            page_size=model.args.page_size,
            dtype=state.storage_dtype,
            key_heads=heads,
            key_dim=head_dim,
            value_dim=value_dim,
            storage_schema=tuple(state.describe_layout() for state in states),
        )

    def computation_config(self, model, layer, state):
        return None

    def scale(self, layer):
        return None


class DenseAdapter(AttentionAdapter):
    state_type = "gqa"

    def dimensions(self, model, layer):
        return layer.tp_q_head_num_, layer.tp_k_head_num_, layer.head_dim_, layer.head_dim_

    def cache(self, model, state, states, heads, head_dim, value_dim):
        return replace(
            super().cache(model, state, states, heads, head_dim, value_dim),
            quant_group_size=getattr(state.config, "group_size", 0),
            scale_mode=getattr(state.config, "scale_mode", "none"),
        )

    def bind(self, model, layer, weight):
        state = model.att_state_manager.get_state(model.att_state_role, self.state_type, layer.layer_num_)
        resources = state.layer_resources(layer.layer_num_)
        return DenseResources(
            **vars(resources),
            alibi=getattr(weight, "tp_alibi", None),
            sink=weight.attn_sinks.weight
            if AttentionFeature.SINK in layer.describe_attention(AttentionPhase.PREFILL).features
            else None,
        )


class MLAAdapter(AttentionAdapter):
    state_type = "mla"

    def computation_config(self, model, layer, state):
        return MLAConfig(layer.qk_rope_head_dim, layer.v_head_dim)

    def dimensions(self, model, layer):
        return layer.tp_q_head_num_, 1, layer.qk_nope_head_dim + layer.qk_rope_head_dim, layer.kv_lora_rank

    def scale(self, layer):
        return layer.softmax_scale

    def bind(self, model, layer, weight):
        state = model.att_state_manager.get_state(model.att_state_role, self.state_type, layer.layer_num_)
        return MLAResources(state.layer_resources(layer.layer_num_), layer.kv_lora_rank, layer.qk_rope_head_dim)


class SparseMLAAdapter(MLAAdapter):
    def computation_config(self, model, layer, state):
        return SparseMLAConfig(model.config.get("index_topk", 2048), model.config.get("index_kpool", 1))

    def dimensions(self, model, layer):
        return (
            layer.tp_q_head_num_,
            1,
            layer.qk_nope_head_dim + layer.network_config_.get("qk_rope_head_dim", 0),
            layer.kv_lora_rank,
        )

    def bind(self, model, layer, weight):
        state = model.att_state_manager.get_state(model.att_state_role, self.state_type, layer.layer_num_)
        return MLAResources(
            state.layer_resources(layer.layer_num_),
            layer.kv_lora_rank,
            layer.network_config_.get("qk_rope_head_dim", 0),
        )


class ConvSSMAdapter(AttentionAdapter):
    def cache(self, model, state, states, heads, head_dim, value_dim):
        return CacheLayout(
            encoding=CacheEncoding.PLAIN,
            page_size=model.args.page_size,
            dtype=state.config.conv_dtype,
            key_heads=heads,
            key_dim=head_dim,
            value_dim=value_dim,
            state_dtype=state.config.ssm_dtype,
            state_mode=state.state_mode,
            storage_schema=tuple(state.describe_layout() for state in states),
        )


class GDNAdapter(ConvSSMAdapter):
    state_type = "gdn"

    def computation_config(self, model, layer, state):
        return GDNConfig(state.config.num_v_heads, state.config.conv_width + 1, layer.network_config_["hidden_act"])

    def dimensions(self, model, layer):
        config = model.att_state_manager.get_state(model.att_state_role, self.state_type, layer.layer_num_).config
        return config.num_k_heads, config.num_k_heads, config.key_dim, config.value_dim

    def bind(self, model, layer, weight):
        state = model.att_state_manager.get_state(model.att_state_role, self.state_type, layer.layer_num_)
        storage = state.layer_resources(layer.layer_num_)
        return GDNResources(
            conv=storage.conv,
            ssm=storage.ssm,
            conv_weight=weight.linear_conv1d.mm_param.weight,
            conv_bias=weight.linear_conv1d.bias,
            a_log=weight.linear_A_log.weight,
            dt_bias=weight.linear_dt_bias.weight,
            mtp_state_index=storage.mtp_state_index,
            replay_state=storage.replay_state,
        )


class KDAAdapter(ConvSSMAdapter):
    state_type = "kda"

    def computation_config(self, model, layer, state):
        return KDAConfig(
            state.config.conv_width + 1, layer.network_config_["linear_attn_config"].get("gate_lower_bound", -5.0)
        )

    def dimensions(self, model, layer):
        config = model.att_state_manager.get_state(model.att_state_role, self.state_type, layer.layer_num_).config
        return config.num_k_heads, config.num_k_heads, config.key_dim, config.value_dim

    def bind(self, model, layer, weight):
        state = model.att_state_manager.get_state(model.att_state_role, self.state_type, layer.layer_num_)
        storage = state.layer_resources(layer.layer_num_)
        return KDAResources(
            conv=storage.conv,
            ssm=storage.ssm,
            conv_weight=weight.get_merged_kda_conv_weight(),
            a_log=weight.linear_A_log.weight,
            dt_bias=weight.linear_dt_bias.weight,
            mtp_state_index=storage.mtp_state_index,
            replay_state=storage.replay_state,
        )


ADAPTERS = {
    AttentionFamily.DENSE: DenseAdapter(),
    AttentionFamily.MLA: MLAAdapter(),
    AttentionFamily.SPARSE_MLA: SparseMLAAdapter(),
    AttentionFamily.GDN: GDNAdapter(),
    AttentionFamily.KDA: KDAAdapter(),
}
