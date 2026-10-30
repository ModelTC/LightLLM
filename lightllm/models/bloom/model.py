from typing import TYPE_CHECKING
from lightllm.models.bloom.layer_infer.pre_layer_infer import BloomPreLayerInfer
from lightllm.models.bloom.layer_infer.post_layer_infer import BloomPostLayerInfer
from lightllm.models.bloom.layer_infer.transformer_layer_infer import BloomTransformerLayerInfer
from lightllm.models.bloom.layer_weights.pre_and_post_layer_weight import BloomPreAndPostLayerWeight
from lightllm.models.bloom.layer_weights.transformer_layer_weight import BloomTransformerLayerWeight
from lightllm.common.basemodel import InferStateInfo, TpPartBaseModel
from lightllm.common.basemodel.attention.triton.fp import TritonAttBackend


if TYPE_CHECKING:
    from easydict import EasyDict
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.common.att_resources.full_att import FullAttentionState


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

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    @classmethod
    def _create_full_att_states(
        cls, model_config: dict[str, object], args: "StartArgs | EasyDict", *,
        model_logical_layer_ids: tuple[int, ...], tp_rank: int = 0, tp_world_size: int = 1,
    ) -> list["FullAttentionState"]:
        """按本模型的全局逻辑层创建 GQA 布局，量化存储由算法入口选择。"""
        from lightllm.common.att_resources.full_att import FullAttentionConfig, create_gqa_state
        from lightllm.common.basemodel.basemodel_utils import get_resource_config
        from lightllm.utils.torch_dtype_utils import get_torch_dtype

        config = get_resource_config(model_config, model_class=cls)
        logical_layer_groups = tuple((layer_id,) for layer_id in model_logical_layer_ids)
        if not logical_layer_groups:
            return []
        state = create_gqa_state(
            config=FullAttentionConfig(
                logical_layer_groups=logical_layer_groups,
                global_head_num=config.get("num_key_value_heads", config["num_attention_heads"]),
                head_dim=config.get("head_dim", config["hidden_size"] // config["num_attention_heads"]),
                dtype=get_torch_dtype(args.data_type),
                tp_rank=tp_rank, tp_world_size=tp_world_size,
            ),
            kv_type=args.llm_kv_type, page_size=args.page_size, device="meta",
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

    def _init_att_backend(self):
        self.prefill_att_backend = TritonAttBackend(self)
        self.decode_att_backend = TritonAttBackend(self)
        return
