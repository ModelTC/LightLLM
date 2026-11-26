import torch
from typing import Optional, TYPE_CHECKING
import triton
from lightllm.common.attention.backends.gdn import GDNAttConfig
from lightllm.common.attention.backends.gqa import selector as gqa_selector
from lightllm.common.attention.base_backend import AttentionBackend
from lightllm.common.attention.backends.gqa import GQAAttConfig, get_gqa_kv_quant
from lightllm.common.attention.backends.gdn import selector as gdn_selector
from lightllm.models.qwen3_moe.model import Qwen3MOEModel
from lightllm.models.qwen3next.layer_weights.transformer_layer_weight import (
    Qwen3NextTransformerLayerWeight,
)
from lightllm.models.qwen3next.layer_weights.pre_and_post_layer_weight import Qwen3NextPreAndPostLayerWeight
from lightllm.models.qwen3next.layer_infer.transformer_layer_infer import (
    Qwen3NextTransformerLayerInfer,
)
from lightllm.models.qwen3next.infer_struct import Qwen3NextInferStateInfo
from lightllm.utils.log_utils import init_logger
from lightllm.models.qwen3next.config import Qwen3NextResourceConfig

if TYPE_CHECKING:
    from easydict import EasyDict
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.common.attention_state.full_att import FullAttentionState
    from lightllm.common.attention_state.bounded_att import BoundedAttentionState

logger = init_logger(__name__)


class Qwen3NextTpPartModel(Qwen3MOEModel):
    is_hybrid_att_model = True

    # weight class
    pre_and_post_weight_class = Qwen3NextPreAndPostLayerWeight
    transformer_weight_class = Qwen3NextTransformerLayerWeight

    # infer class
    transformer_layer_infer_class = Qwen3NextTransformerLayerInfer

    # infer state class
    infer_state_class = Qwen3NextInferStateInfo

    @classmethod
    def create_att_backends(cls, model: "Qwen3NextTpPartModel") -> tuple[AttentionBackend, ...]:
        linear_layers = [layer for layer in model.layers_infer if layer.is_linear_attention_layer]
        full_layers = [layer for layer in model.layers_infer if not layer.is_linear_attention_layer]
        backends = []
        if linear_layers:
            config = GDNAttConfig(
                num_k_heads=model.config["linear_num_key_heads"] // model.tp_world_size_,
                num_v_heads=model.config["linear_num_value_heads"] // model.tp_world_size_,
                head_k_dim=model.config["linear_key_head_dim"],
                head_v_dim=model.config["linear_value_head_dim"],
                conv_kernel_dim=model.config["linear_conv_kernel_dim"],
                activation=model.config["hidden_act"],
                dtype=model.data_type,
                state_mtp_step=model.args.mtp_step,
                replay=model.args.ssm_state_mode == "replay",
            )
            layer_ids = tuple(layer.layer_num_ for layer in linear_layers)
            backends.append(
                gdn_selector.select_gdn_prefill_backend(att_config=config, model=model, logical_layer_ids=layer_ids)
            )
            select_decode = (
                gdn_selector.select_gdn_replay_decode_backend
                if config.replay
                else gdn_selector.select_gdn_decode_backend
            )
            backends.append(select_decode(att_config=config, model=model, logical_layer_ids=layer_ids))
        if full_layers:
            config = model.config
            layer_ids = tuple(layer.layer_num_ for layer in full_layers)
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
            backends.append(
                gqa_selector.select_gqa_prefill_backend(att_config=att_config, model=model, logical_layer_ids=layer_ids)
            )
            backends.append(
                gqa_selector.select_gqa_decode_backend(att_config=att_config, model=model, logical_layer_ids=layer_ids)
            )
        return tuple(backends)

    def __init__(self, kvargs) -> None:
        self._init_triton()
        super().__init__(kvargs)

    @classmethod
    def create_bounded_att_states(
        cls,
        model_config: dict[str, object],
        args: "StartArgs | EasyDict",
        model_logical_layer_ids: tuple[int, ...],
        request_capacity: int = 1,
        tp_rank: int = 0,
        tp_world_size: int = 1,
        device: str = "meta",
    ) -> list["BoundedAttentionState"]:
        """创建当前模型的 bounded 运行态；CPU 布局查询传入 meta 设备。"""
        from lightllm.common.attention_state.bounded_att import create_gdn_state, ConvSSMStateConfig

        config = cls.normalize_model_config(model_config)
        assert config["num_attention_heads"] % tp_world_size == 0
        assert not (
            args.mtp_step and args.enable_prefill_decode_mixed
        ), "MTP linear attention does not support mixed prefill/decode"
        cfg = Qwen3NextResourceConfig.from_model_config(config, args)
        full_layer_indices = set(cfg.get_full_att_layer_indices())
        logical_layer_ids = tuple(
            layer for i, layer in enumerate(model_logical_layer_ids) if i not in full_layer_indices
        )
        if not logical_layer_ids:
            return []
        state = create_gdn_state(
            config=ConvSSMStateConfig(
                logical_layer_ids=logical_layer_ids,
                global_k_heads=cfg.global_linear_k_heads,
                global_v_heads=cfg.global_linear_v_heads,
                key_dim=cfg.head_linear_k_dim,
                value_dim=cfg.head_linear_v_dim,
                conv_width=cfg.conv_kernel_size - 1,
                conv_dtype=cfg.conv_state_dtype,
                ssm_dtype=cfg.ssm_state_dtype,
                tp_rank=tp_rank,
                tp_world_size=tp_world_size,
            ),
            request_capacity=request_capacity,
            mtp_step=args.mtp_step,
            device=device,
            ssm_state_mode=args.ssm_state_mode,
            ssm_replay_config=args.ssm_replay_config,
        )
        return [state]

    def autotune_layers(self):
        return self.config["full_attention_interval"]

    @classmethod
    def create_full_att_states(
        cls,
        model_config: dict[str, object],
        args: "StartArgs | EasyDict",
        model_logical_layer_ids: tuple[int, ...],
        tp_rank: int = 0,
        tp_world_size: int = 1,
    ) -> list["FullAttentionState"]:
        """按 Qwen 混合层周期选择 full-attention 层，保留全局逻辑层号。"""
        from lightllm.common.attention_state.full_att import GQAAttentionConfig, create_gqa_state
        from lightllm.utils.torch_dtype_utils import get_torch_dtype

        config = cls.normalize_model_config(model_config)
        assert args.llm_kv_type in ("None", "fp8kv_sph", "fp8kv_spt"), "unsupported Qwen3Next KV encoding"
        cfg = Qwen3NextResourceConfig.from_model_config(config, args)
        logical_layer_groups = tuple((model_logical_layer_ids[index],) for index in cfg.get_full_att_layer_indices())
        if not logical_layer_groups:
            return []
        state = create_gqa_state(
            config=GQAAttentionConfig(
                logical_layer_groups=logical_layer_groups,
                global_head_num=cfg.full_att_all_num_kv_heads,
                head_dim=cfg.full_att_head_dim,
                dtype=get_torch_dtype(args.data_type),
                tp_rank=tp_rank,
                tp_world_size=tp_world_size,
            ),
            kv_type=args.llm_kv_type,
            page_size=args.page_size,
            device="meta",
        )
        return [state]

    def _init_triton(self):
        def _triton_allocator(size: int, alignment: int, stream: Optional[int]) -> torch.Tensor:
            return torch.empty(size, device="cuda", dtype=torch.int8)

        # Set Triton allocator for TMA descriptors
        # This is required for kernels in common/.../linear_att/fla/ops/solve_tril.py
        triton.set_allocator(_triton_allocator)
        logger.info("Triton allocator set for Qwen3Next model")
        return

    def _init_config(self):
        super()._init_config()
        self.num_kv_heads = max(self.config["num_key_value_heads"] // self.tp_world_size_, 1)
