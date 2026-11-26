from typing import TYPE_CHECKING
import torch
from lightllm.common.attention.backends.gqa import selector as gqa_selector
from lightllm.common.attention.base_backend import AttentionBackend
from lightllm.common.attention.backends.gqa import GQAAttConfig, get_gqa_kv_quant
from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.mixtral.layer_infer.transformer_layer_infer import MixtralTransformerLayerInfer
from lightllm.models.mixtral.layer_weights.transformer_layer_weight import MixtralTransformerLayerWeight

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


if TYPE_CHECKING:
    from easydict import EasyDict
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.common.attention_state.full_att import FullAttentionState


class MixtralTpPartModel(TpPartBaseModel):
    # weight class
    pre_and_post_weight_class = LlamaPreAndPostLayerWeight
    transformer_weight_class = MixtralTransformerLayerWeight

    # infer class
    pre_layer_infer_class = LlamaPreLayerInfer
    post_layer_infer_class = LlamaPostLayerInfer
    transformer_layer_infer_class = MixtralTransformerLayerInfer

    # infer state class
    infer_state_class = LlamaInferStateInfo

    @classmethod
    def create_att_backends(cls, model: "MixtralTpPartModel") -> tuple[AttentionBackend, ...]:
        if not model.layers_infer:
            return ()
        config = model.config
        layer_ids = tuple(layer.layer_num_ for layer in model.layers_infer)
        head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
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
        )
        return (
            gqa_selector.select_gqa_prefill_backend(att_config=att_config, model=model, logical_layer_ids=layer_ids),
            gqa_selector.select_gqa_decode_backend(att_config=att_config, model=model, logical_layer_ids=layer_ids),
        )

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    def _init_config(self):
        super()._init_config()
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

    def _verify_params(self):
        assert self.load_way in ["HF"], "mixtral only supports HF format to load Now!"
        assert self.config["num_key_value_heads"] % self.tp_world_size_ == 0
        assert self.config["num_attention_heads"] % self.tp_world_size_ == 0
        return

    def _init_custom(self):
        self._init_to_get_rotary()
        return

    def _init_to_get_rotary(self, default_base=10000):
        if self.config.get("rope_scaling", {}) is None:
            rope_scaling_factor = 1.0
        else:
            rope_scaling_factor = self.config.get("rope_scaling", {}).get("factor", 1.0)

        base = self.config.get("rope_theta", float(default_base))

        if "max_sequence_length" in self.config:
            max_seq_len = self.config["max_sequence_length"]
        else:
            max_position_embeddings = self.config.get(
                "max_position_embeddings", 2048 if base <= 10000.0 + 1e-5 else 16384
            )
            max_seq_len = max_position_embeddings * rope_scaling_factor

        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.head_dim_, 2, device="cpu", dtype=torch.float32) / self.head_dim_)
        )
        t = torch.arange(max_seq_len + 1024 * 64, device="cpu", dtype=torch.float32) / rope_scaling_factor
        freqs = torch.outer(t, inv_freq)

        self._cos_cached = torch.cos(freqs).to(self.data_type).cuda()
        self._sin_cached = torch.sin(freqs).to(self.data_type).cuda()
        return
