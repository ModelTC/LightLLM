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
        """按当前模型的方案创建 full-attention 定义，不分配 KV 容量或注册管理器。"""
        from lightllm.common.basemodel.basemodel_utils import create_full_attention_definition

        state = create_full_attention_definition(
            model_class=cls, model_config=model_config, args=args,
            model_logical_layer_ids=model_logical_layer_ids, tp_rank=tp_rank, tp_world_size=tp_world_size,
        )
        return [] if state is None else [state]

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
