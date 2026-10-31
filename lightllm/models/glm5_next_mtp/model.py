from typing import TYPE_CHECKING
from copy import deepcopy

from lightllm.common.basemodel import TpPartBaseModel
from lightllm.common.basemodel.attention.nsa.glm5_next import Glm5NextSparseAttBackend
from lightllm.common.basemodel.attention.nsa.fp8_glm5_next import Fp8Glm5NextSparseAttBackend
from lightllm.models.glm5_next.layer_infer.transformer_layer_infer import Glm5NextTransformerLayerInfer
from lightllm.models.glm5_next.layer_weights.transformer_layer_weight import Glm5NextTransformerLayerWeight
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from .layer_infer.pre_layer_infer import Glm5NextMTPPreLayerInfer
from .layer_weights.pre_and_post_layer_weight import Glm5NextMTPPreAndPostLayerWeight


if TYPE_CHECKING:
    from easydict import EasyDict
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.common.attention_state.full_att import FullAttentionState
    from lightllm.common.attention_state.att_tail import AttentionTailState


class Glm5NextMTPModel(TpPartBaseModel):
    is_mtp_draft_model = True
    pre_and_post_weight_class = Glm5NextMTPPreAndPostLayerWeight
    transformer_weight_class = Glm5NextTransformerLayerWeight
    pre_layer_infer_class = Glm5NextMTPPreLayerInfer
    post_layer_infer_class = LlamaPostLayerInfer
    transformer_layer_infer_class = Glm5NextTransformerLayerInfer

    def __init__(self, kvargs):
        self.main_model = kvargs.pop("main_model")
        self.mtp_previous_draft_models = kvargs.pop("mtp_previous_draft_models")
        super().__init__(kvargs)

    def _init_config(self):
        # Reuse the validated target configuration without changing its mHC layout.
        self.config = self.normalize_model_config(deepcopy(self.main_model.config))
        assert self.config.get("num_nextn_predict_layers") == 1, "GLM NextN requires one native MTP block"

    def _init_weights(self, start_layer_index=None):
        assert start_layer_index is None
        self.pre_post_weight = self.pre_and_post_weight_class(self.data_type, self.config, self.quant_cfg)
        self.pre_post_weight.wte_weight_ = self.main_model.pre_post_weight.wte_weight_
        self.pre_post_weight.lm_head_weight_ = self.main_model.pre_post_weight.lm_head_weight_
        self.pre_post_weight.main_norm_weight_ = self.main_model.pre_post_weight.final_norm_weight_
        self.trans_layers_weight = [
            self.transformer_weight_class(
                layer_num=self.config["mtp_checkpoint_layer_start"],
                data_type=self.data_type,
                network_config=self.config,
                quant_cfg=self.quant_cfg,
            )
        ]

    def _init_infer_layer(self, start_layer_index=None):
        assert start_layer_index is None
        self.pre_infer = self.pre_layer_infer_class(self.config)
        self.post_infer = self.post_layer_infer_class(self.config)
        # Chained modules reuse the native weights but own distinct cache layers.
        layer_index = len(self.main_model.layers_infer) + len(self.mtp_previous_draft_models)
        self.layers_infer = [self.transformer_layer_infer_class(layer_index, self.config)]

    def _init_some_value(self):
        self.layers_num = 1
        self.vocab_size = self.config["vocab_size"]
        self.tp_k_head_num_ = 1
        self.tp_v_head_num_ = 0
        self.qk_nope_head_dim = self.config["qk_nope_head_dim"]
        self.qk_rope_head_dim = self.config["qk_rope_head_dim"]
        self.q_lora_rank = self.config["q_lora_rank"]
        self.kv_lora_rank = self.config["kv_lora_rank"]
        self.v_head_dim = self.config.get("v_head_dim", self.qk_nope_head_dim)
        self.head_dim_ = self.kv_lora_rank + self.qk_rope_head_dim

    @classmethod
    def create_full_att_states(
        cls,
        model_config: dict[str, object],
        args: "StartArgs | EasyDict",
        model_logical_layer_ids: tuple[int, ...],
        tp_rank: int = 0,
        tp_world_size: int = 1,
    ) -> list["FullAttentionState"]:
        """原生 GLM draft 的自有层采用 NoPE packed MLA，与 main 分别持有存储。"""
        from lightllm.common.attention_state.full_att import FullAttentionConfig, create_mla_state
        from lightllm.utils.torch_dtype_utils import get_torch_dtype

        config = cls.normalize_model_config(model_config)
        logical_layer_groups = tuple((layer_id,) for layer_id in model_logical_layer_ids)
        if not logical_layer_groups:
            return []
        state = create_mla_state(
            config=FullAttentionConfig(
                logical_layer_groups=logical_layer_groups,
                global_head_num=1,
                head_dim=config["kv_lora_rank"],
                dtype=get_torch_dtype(args.data_type),
                tp_rank=tp_rank,
                tp_world_size=tp_world_size,
            ),
            kv_type=args.llm_kv_type,
            packed=True,
            fp8_format="group64_e8m0",
            page_size=args.page_size,
            device="meta",
        )
        return [state]

    @classmethod
    def create_att_tail_states(
        cls,
        model_config: dict[str, object],
        args: "StartArgs | EasyDict",
        model_logical_layer_ids: tuple[int, ...],
        request_capacity: int = 1,
        tp_rank: int = 0,
        tp_world_size: int = 1,
        device: str = "meta",
    ) -> list["AttentionTailState"]:
        """直接选择本模型需要 K-pool tail 的层，不依赖 full State 创建过程。"""
        from lightllm.common.attention_state.att_tail import create_kpool_state, KPoolStateConfig
        from lightllm.utils.torch_dtype_utils import get_torch_dtype

        config = cls.normalize_model_config(model_config)
        logical_layer_ids = model_logical_layer_ids
        if not logical_layer_ids:
            return []
        assert args.linear_att_hash_page_size % config["index_kpool"] == 0, "GLM cache pages must align to index_kpool"
        state = create_kpool_state(
            config=KPoolStateConfig(
                logical_layer_ids=logical_layer_ids,
                pool_size=config["index_kpool"],
                head_dim=config["index_head_dim"],
                dtype=get_torch_dtype(args.data_type),
                tp_rank=tp_rank,
                tp_world_size=tp_world_size,
            ),
            request_capacity=request_capacity,
            mtp_step=args.mtp_step,
            device=device,
        )
        return [state]

    @classmethod
    def normalize_model_config(cls, model_config: dict[str, object]) -> dict[str, object]:
        """区分 draft 自有层数与 checkpoint 中的 MTP 权重层起点。"""
        config = super().normalize_model_config(model_config)
        # checkpoint 中 MTP 权重接在 main 层之后；重复规范化不能覆盖这个起点。
        config.setdefault("mtp_checkpoint_layer_start", config["num_hidden_layers"])
        config.update(num_hidden_layers=1, n_layer=1, mhc=False)
        return config

    def _init_att_backend(self):
        backend_class = (
            Fp8Glm5NextSparseAttBackend if self.args.llm_kv_type == "fp8kv_dsa" else Glm5NextSparseAttBackend
        )
        self.prefill_att_backend = backend_class(model=self)
        self.decode_att_backend = self.prefill_att_backend

    def autotune_layers(self):
        return 1
