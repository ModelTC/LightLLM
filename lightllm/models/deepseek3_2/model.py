from typing import TYPE_CHECKING
from lightllm.models.deepseek2.model import Deepseek2TpPartModel
from lightllm.models.deepseek3_2.layer_weights.transformer_layer_weight import Deepseek3_2TransformerLayerWeight
from lightllm.models.deepseek3_2.layer_infer.transformer_layer_infer import Deepseek3_2TransformerLayerInfer
from lightllm.common.basemodel.attention import get_nsa_prefill_att_backend_class, get_nsa_decode_att_backend_class


if TYPE_CHECKING:
    from easydict import EasyDict
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.common.attention_state.full_att import FullAttentionState


class Deepseek3_2TpPartModel(Deepseek2TpPartModel):

    # weight class
    transformer_weight_class = Deepseek3_2TransformerLayerWeight

    # infer class
    transformer_layer_infer_class = Deepseek3_2TransformerLayerInfer

    @classmethod
    def create_full_att_states(
        cls, model_config: dict[str, object], args: "StartArgs | EasyDict", *,
        model_logical_layer_ids: tuple[int, ...], tp_rank: int = 0, tp_world_size: int = 1,
    ) -> list["FullAttentionState"]:
        """DeepSeek V3.2 使用包含 indexer 字段的 packed MLA，编码由 MLA 创建入口选择。"""
        from lightllm.common.attention_state.full_att import MLAAttentionConfig, create_mla_state
        from lightllm.utils.torch_dtype_utils import get_torch_dtype

        config = cls.normalize_model_config(model_config)
        logical_layer_groups = tuple((layer_id,) for layer_id in model_logical_layer_ids)
        if not logical_layer_groups:
            return []
        state = create_mla_state(
            config=MLAAttentionConfig(
                logical_layer_groups=logical_layer_groups,
                latent_dim=config["kv_lora_rank"] + config["qk_rope_head_dim"],
                dtype=get_torch_dtype(args.data_type),
                tp_rank=tp_rank, tp_world_size=tp_world_size,
            ),
            kv_type=args.llm_kv_type, packed=True, page_size=args.page_size, device="meta",
        )
        return [state]

    def _init_att_backend(self):
        self.prefill_att_backend = get_nsa_prefill_att_backend_class(index=0)(model=self)
        self.decode_att_backend = get_nsa_decode_att_backend_class(index=0)(model=self)
        return
