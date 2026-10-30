from typing import TYPE_CHECKING
from lightllm.models.gemma_2b.layer_weights.transformer_layer_weight import Gemma_2bTransformerLayerWeight
from lightllm.models.gemma_2b.layer_weights.pre_and_post_layer_weight import Gemma_2bPreAndPostLayerWeight
from lightllm.models.gemma_2b.layer_infer.pre_layer_infer import Gemma_2bPreLayerInfer
from lightllm.models.gemma_2b.layer_infer.transformer_layer_infer import Gemma_2bTransformerLayerInfer
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.model import LlamaTpPartModel


if TYPE_CHECKING:
    from easydict import EasyDict
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.common.att_resources.full_att import FullAttentionState


class Gemma_2bTpPartModel(LlamaTpPartModel):
    # weight class
    pre_and_post_weight_class = Gemma_2bPreAndPostLayerWeight
    transformer_weight_class = Gemma_2bTransformerLayerWeight

    # infer class
    pre_layer_infer_class = Gemma_2bPreLayerInfer
    post_layer_infer_class = LlamaPostLayerInfer
    transformer_layer_infer_class = Gemma_2bTransformerLayerInfer

    # infer state class
    infer_state_class = LlamaInferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    def _init_config(self):
        super()._init_config()
        return

    def _verify_params(self):
        assert self.load_way in ["HF"], "gemma only supports HF format to load Now!"
        # assert self.config["num_key_value_heads"] % self.tp_world_size_ == 0
        assert self.config["num_attention_heads"] % self.tp_world_size_ == 0
        return

    @classmethod
    def create_full_att_states(
        cls, model_config: dict[str, object], args: "StartArgs | EasyDict", *,
        model_logical_layer_ids: tuple[int, ...], tp_rank: int = 0, tp_world_size: int = 1,
    ) -> list["FullAttentionState"]:
        """Gemma 2B 保留复制 KV 的布局及本模型定义的 head 维度。"""
        from lightllm.common.att_resources.full_att import FullAttentionConfig, create_gqa_state
        from lightllm.utils.torch_dtype_utils import get_torch_dtype

        config = cls.normalize_model_config(model_config)
        logical_layer_groups = tuple((layer_id,) for layer_id in model_logical_layer_ids)
        if not logical_layer_groups:
            return []
        state = create_gqa_state(
            config=FullAttentionConfig(
                logical_layer_groups=logical_layer_groups,
                global_head_num=config.get("num_key_value_heads", config["num_attention_heads"]),
                head_dim=config["hidden_size"] // config["num_attention_heads"],
                dtype=get_torch_dtype(args.data_type),
                tp_rank=tp_rank, tp_world_size=tp_world_size,
                replicated=True,
            ),
            kv_type=args.llm_kv_type, page_size=args.page_size, device="meta",
        )
        return [state]
