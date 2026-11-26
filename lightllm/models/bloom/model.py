from typing import TYPE_CHECKING
from lightllm.common.attention.backends.gqa import selector as gqa_selector
from lightllm.common.attention.base_backend import AttentionBackend
from lightllm.common.attention.backends.gqa import GQAAttConfig, get_gqa_kv_quant
from lightllm.models.bloom.layer_infer.pre_layer_infer import BloomPreLayerInfer
from lightllm.models.bloom.layer_infer.post_layer_infer import BloomPostLayerInfer
from lightllm.models.bloom.layer_infer.transformer_layer_infer import BloomTransformerLayerInfer
from lightllm.models.bloom.layer_weights.pre_and_post_layer_weight import BloomPreAndPostLayerWeight
from lightllm.models.bloom.layer_weights.transformer_layer_weight import BloomTransformerLayerWeight
from lightllm.common.basemodel import InferStateInfo, TpPartBaseModel


if TYPE_CHECKING:
    from easydict import EasyDict
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.common.attention_state.full_att import FullAttentionState


class BloomTpPartModel(TpPartBaseModel):
    # weight class
    pre_and_post_weight_class = BloomPreAndPostLayerWeight
    transformer_weight_class = BloomTransformerLayerWeight

    # infer class
    pre_layer_infer_class = BloomPreLayerInfer
    post_layer_infer_class = BloomPostLayerInfer
    transformer_layer_infer_class = BloomTransformerLayerInfer

    # infer state class
    infer_state_class = InferStateInfo

    @classmethod
    def create_att_backends(cls, model: "BloomTpPartModel") -> tuple[AttentionBackend, ...]:
        if not model.layers_infer:
            return ()
        config = model.config
        layer_ids = tuple(layer.layer_num_ for layer in model.layers_infer)
        head_dim = config["n_embed"] // config["num_attention_heads"]
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
            num_kv_heads=config["num_attention_heads"] // model.tp_world_size_,
            head_dim=head_dim,
            dtype=model.data_type,
            causal=not model.is_mtp_draft_model or model.args.mtp_mode not in ("dspark", "dflash", "dflash2"),
            kv_quant=kv_quant,
            kv_quant_group_size=group_size,
            has_q_scale=has_q_scale,
            has_alibi=True,
        )
        return (
            gqa_selector.select_gqa_prefill_backend(
                att_config=att_config, requested="triton", model=model, logical_layer_ids=layer_ids
            ),
            gqa_selector.select_gqa_decode_backend(
                att_config=att_config, requested="triton", model=model, logical_layer_ids=layer_ids
            ),
        )

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

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
        """按本模型的全局逻辑层创建 GQA 布局，量化存储由算法入口选择。"""
        from lightllm.common.attention_state.full_att import GQAAttentionConfig, create_gqa_state
        from lightllm.utils.torch_dtype_utils import get_torch_dtype

        config = cls.normalize_model_config(model_config)
        logical_layer_groups = tuple((layer_id,) for layer_id in model_logical_layer_ids)
        if not logical_layer_groups:
            return []
        state = create_gqa_state(
            config=GQAAttentionConfig(
                logical_layer_groups=logical_layer_groups,
                global_head_num=config.get("num_key_value_heads", config["num_attention_heads"]),
                head_dim=config.get("head_dim", config["hidden_size"] // config["num_attention_heads"]),
                dtype=get_torch_dtype(args.data_type),
                tp_rank=tp_rank,
                tp_world_size=tp_world_size,
            ),
            kv_type=args.llm_kv_type,
            page_size=args.page_size,
            device="meta",
        )
        return [state]

    def _init_config(self):
        super()._init_config()
        # rename key
        # repair_config()
        self._reset_num_key_value_heads()
        return

    def _reset_num_key_value_heads(self):
        self.config["num_key_value_heads"] = self.config["num_attention_heads"]
        return
