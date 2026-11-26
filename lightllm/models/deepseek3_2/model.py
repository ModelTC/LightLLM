from typing import TYPE_CHECKING
from lightllm.common.attention.backends.sparse_mla import SparseMLAAttConfig
from lightllm.common.attention.base_backend import AttentionBackend
from lightllm.common.attention.backends.sparse_mla import selector as sparse_mla_selector
from lightllm.models.deepseek2.model import Deepseek2TpPartModel
from lightllm.models.deepseek3_2.layer_weights.transformer_layer_weight import Deepseek3_2TransformerLayerWeight
from lightllm.models.deepseek3_2.layer_infer.transformer_layer_infer import Deepseek3_2TransformerLayerInfer
from lightllm.models.deepseek3_2.infer_struct import Deepseek3_2InferStateInfo


if TYPE_CHECKING:
    from easydict import EasyDict
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.common.attention_state.full_att import FullAttentionState


class Deepseek3_2TpPartModel(Deepseek2TpPartModel):
    # weight class
    transformer_weight_class = Deepseek3_2TransformerLayerWeight

    # infer class
    transformer_layer_infer_class = Deepseek3_2TransformerLayerInfer
    infer_state_class = Deepseek3_2InferStateInfo

    @classmethod
    def create_att_backends(cls, model: "Deepseek3_2TpPartModel") -> tuple[AttentionBackend, ...]:
        if not model.layers_infer:
            return ()
        layer_ids = tuple(layer.layer_num_ for layer in model.layers_infer)
        model_config = model.config
        softmax_scale = (model_config["qk_nope_head_dim"] + model_config["qk_rope_head_dim"]) ** -0.5
        rope_scaling = model_config.get("rope_scaling")
        if rope_scaling and rope_scaling.get("mscale_all_dim", 0):
            from lightllm.models.llama.yarn_rotary_utils import get_deepseek_mscale

            mscale = get_deepseek_mscale(scale=rope_scaling["factor"], mscale=rope_scaling["mscale_all_dim"])
            softmax_scale *= mscale * mscale
        config = SparseMLAAttConfig(
            num_heads=model_config["num_attention_heads"] // model.tp_world_size_,
            kv_lora_rank=model_config["kv_lora_rank"],
            qk_rope_head_dim=model_config["qk_rope_head_dim"],
            softmax_scale=softmax_scale,
            dtype=model.data_type,
            kv_cache_format="fp8_ds_mla" if model.args.llm_kv_type == "fp8kv_dsa" else "bf16",
            topk=model_config["index_topk"],
        )
        return (
            sparse_mla_selector.select_sparse_mla_prefill_backend(
                att_config=config, model=model, logical_layer_ids=layer_ids
            ),
            sparse_mla_selector.select_sparse_mla_decode_backend(
                att_config=config, model=model, logical_layer_ids=layer_ids
            ),
        )

    @classmethod
    def create_full_att_states(
        cls,
        model_config: dict[str, object],
        args: "StartArgs | EasyDict",
        *,
        model_logical_layer_ids: tuple[int, ...],
        tp_rank: int = 0,
        tp_world_size: int = 1,
    ) -> list["FullAttentionState"]:
        """DeepSeek V3.2 使用包含 indexer 字段的 packed MLA，编码由 MLA 创建入口选择。"""
        from lightllm.common.attention_state.full_att import PackedMLALogicalConfig, create_packed_mla_state

        config = cls.normalize_model_config(model_config)
        logical_layer_groups = tuple((layer_id,) for layer_id in model_logical_layer_ids)
        if not logical_layer_groups:
            return []
        if args.llm_kv_type not in ("None", "fp8kv_dsa"):
            raise ValueError(f"{cls.__name__} requires --llm_kv_type None or fp8kv_dsa, got {args.llm_kv_type!r}")
        state = create_packed_mla_state(
            config=PackedMLALogicalConfig(
                logical_layer_groups=logical_layer_groups,
                latent_dim=config["kv_lora_rank"] + config["qk_rope_head_dim"],
                tp_rank=tp_rank,
                tp_world_size=tp_world_size,
                format="packed_group128_fp8" if args.llm_kv_type == "fp8kv_dsa" else "packed_bf16",
                indexer_dim=config.get("indexer_dim", 128),
            ),
            page_size=args.page_size,
            device="meta",
        )
        return [state]
