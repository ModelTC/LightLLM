from lightllm.common.attention.base_backend import AttentionBackend
from lightllm.common.attention.backends.gqa import selector as gqa_selector
from lightllm.common.attention.backends.gqa import GQAAttConfig, get_gqa_kv_quant
from lightllm.models.gpt_oss.layer_infer.transformer_layer_infer import GptOssTransformerLayerInfer
from lightllm.models.gpt_oss.layer_weights.transformer_layer_weight import GptOssTransformerLayerWeight
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class GptOssTpPartModel(LlamaTpPartModel):
    # weight class
    transformer_weight_class = GptOssTransformerLayerWeight

    # infer class
    transformer_layer_infer_class = GptOssTransformerLayerInfer

    @classmethod
    def create_att_backends(cls, model: "GptOssTpPartModel") -> tuple[AttentionBackend, ...]:
        config = model.config
        backends = []
        for sliding in (True, False):
            layer_ids = tuple(
                layer.layer_num_
                for layer in model.layers_infer
                if (config["layer_types"][layer.layer_num_] == "sliding_attention") == sliding
            )
            if not layer_ids:
                continue
            window = (config["sliding_window"] - 1, 0) if sliding else (-1, -1)
            head_dim = config["head_dim"]
            state = model.att_state_manager.get_state(
                model_role=model.att_state_role, state_type_filter="gqa", logical_layer_id=layer_ids[0]
            )
            kv_quant, group_size, has_q_scale = get_gqa_kv_quant(
                llm_kv_type=model.args.llm_kv_type,
                kv_scales=state.get_layer_kv_scales(logical_layer_id=layer_ids[0]),
                head_dim=head_dim,
            )
            att_config = GQAAttConfig(
                num_heads=config["num_attention_heads"] // model.tp_world_size_,
                num_kv_heads=max(config["num_key_value_heads"] // model.tp_world_size_, 1),
                head_dim=head_dim,
                dtype=model.data_type,
                causal=not model.is_mtp_draft_model or model.args.mtp_mode not in ("dspark", "dflash", "dflash2"),
                kv_quant=kv_quant,
                kv_quant_group_size=group_size,
                has_q_scale=has_q_scale,
                has_sinks=True,
                sliding_window=window,
            )
            backends.append(
                gqa_selector.select_gqa_prefill_backend(att_config=att_config, model=model, logical_layer_ids=layer_ids)
            )
            backends.append(
                gqa_selector.select_gqa_decode_backend(att_config=att_config, model=model, logical_layer_ids=layer_ids)
            )
        return tuple(backends)

    def __init__(self, kvargs):
        super().__init__(kvargs)
