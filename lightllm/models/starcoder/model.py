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
        """StarCoder 的单份 KV 在各 TP rank 复制，逻辑层由本模型直接定义。"""
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
                global_head_num=1,
                head_dim=config["hidden_size"] // config["num_attention_heads"],
                dtype=get_torch_dtype(args.data_type),
                tp_rank=tp_rank, tp_world_size=tp_world_size,
                replicated=True,
            ),
            kv_type=args.llm_kv_type, page_size=args.page_size, device="meta",
        )
        return [state]

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

    def _init_some_value(self):
        super()._init_some_value()
        self.tp_k_head_num_ = self.config["num_key_value_heads"]
        self.tp_v_head_num_ = self.config["num_key_value_heads"]
        return
