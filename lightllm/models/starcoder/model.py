from typing import TYPE_CHECKING
from lightllm.models.starcoder.layer_infer.transformer_layer_infer import StarcoderTransformerLayerInfer
from lightllm.models.starcoder.layer_infer.pre_layer_infer import StarcoderPreLayerInfer
from lightllm.models.starcoder.layer_weights.transformer_layer_weight import StarcoderTransformerLayerWeight
from lightllm.models.starcoder.layer_weights.pre_and_post_layer_weight import StarcoderPreAndPostLayerWeight
from lightllm.models.bloom.layer_infer.post_layer_infer import BloomPostLayerInfer
from lightllm.common.build_utils import repair_config
from lightllm.common.basemodel import TpPartBaseModel
from lightllm.common.basemodel import InferStateInfo


if TYPE_CHECKING:
    from easydict import EasyDict
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.common.att_resources.full_att import FullAttentionState


class StarcoderTpPartModel(TpPartBaseModel):
    # weight class
    pre_and_post_weight_class = StarcoderPreAndPostLayerWeight
    transformer_weight_class = StarcoderTransformerLayerWeight

    # infer class
    pre_layer_infer_class = StarcoderPreLayerInfer
    transformer_layer_infer_class = StarcoderTransformerLayerInfer
    post_layer_infer_class = BloomPostLayerInfer
    infer_state_class = InferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)

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
        repair_config(self.config, same_names=["rms_norm_eps", "layer_norm_epsilon"])
        self._reset_num_key_value_heads()
        return

    def _reset_num_key_value_heads(self):
        self.config["num_key_value_heads"] = 1
        return

    def _verify_params(self):
        assert self.load_way == "HF", "StarCoder only support HF format to load Now!"

    @classmethod
    def _full_attention_options(cls, config, args):
        from lightllm.common.basemodel.basemodel_utils import get_full_attention_options

        options = get_full_attention_options(config, args)
        options.update(heads=1, dim=config["hidden_size"] // config["num_attention_heads"], replicated=True)
        return options

    def _init_some_value(self):
        super()._init_some_value()
        self.tp_k_head_num_ = self.config["num_key_value_heads"]
        self.tp_v_head_num_ = self.config["num_key_value_heads"]
        return
