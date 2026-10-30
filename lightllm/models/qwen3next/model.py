import torch
from typing import Optional, TYPE_CHECKING
import triton
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
from lightllm.models.qwen3next.att_resource_config import Qwen3NextResourceConfig

if TYPE_CHECKING:
    from easydict import EasyDict
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.common.att_resources.full_att import FullAttentionState
    from lightllm.common.att_resources.bounded_att import BoundedAttentionState

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

    def __init__(self, kvargs) -> None:
        self._init_triton()
        super().__init__(kvargs)

    @classmethod
    def _create_bounded_att_states(
        cls, model_config: dict[str, object], args: "StartArgs | EasyDict", *,
        model_logical_layer_ids: tuple[int, ...], request_capacity: int = 1,
        tp_rank: int = 0, tp_world_size: int = 1, device: str = "meta",
    ) -> list["BoundedAttentionState"]:
        """创建当前模型的 bounded 运行态；CPU 布局查询传入 meta 设备。"""
        from lightllm.common.att_resources.bounded_att import create_gdn_state, ConvSSMStateConfig
        from lightllm.common.basemodel.basemodel_utils import get_resource_config

        config = get_resource_config(model_config, model_class=cls)
        assert config["num_attention_heads"] % tp_world_size == 0
        assert not (args.mtp_step and args.enable_prefill_decode_mixed), "MTP linear attention does not support mixed prefill/decode"
        cfg = Qwen3NextResourceConfig.from_model_config(config, args, include_draft=False)
        logical_layer_ids = tuple(layer for i, layer in enumerate(model_logical_layer_ids) if (i + 1) % cfg.full_attention_interval)
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
            request_capacity=request_capacity, mtp_step=args.mtp_step, device=device,
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

    def autotune_layers(self):
        return self.config["full_attention_interval"]

    def _init_config(self):
        super()._init_config()
        self.num_kv_heads = max(self.config["num_key_value_heads"] // self.tp_world_size_, 1)

    @classmethod
    def _create_full_att_states(
        cls, model_config: dict[str, object], args: "StartArgs | EasyDict", *,
        model_logical_layer_ids: tuple[int, ...], tp_rank: int = 0, tp_world_size: int = 1,
    ) -> list["FullAttentionState"]:
        """按 Qwen 混合层周期选择 full-attention 层，保留全局逻辑层号。"""
        from lightllm.common.att_resources.full_att import FullAttentionConfig, create_gqa_state
        from lightllm.common.basemodel.basemodel_utils import get_resource_config
        from lightllm.utils.torch_dtype_utils import get_torch_dtype

        config = get_resource_config(model_config, model_class=cls)
        assert args.llm_kv_type in ("None", "fp8kv_sph", "fp8kv_spt"), "unsupported Qwen3Next KV encoding"
        cfg = Qwen3NextResourceConfig.from_model_config(config, args, include_draft=False)
        logical_layer_groups = tuple((model_logical_layer_ids[index],) for index in cfg.get_full_att_logical_layer_ids())
        if not logical_layer_groups:
            return []
        state = create_gqa_state(
            config=FullAttentionConfig(
                logical_layer_groups=logical_layer_groups,
                global_head_num=cfg.full_att_all_num_kv_heads,
                head_dim=cfg.full_att_head_dim,
                dtype=get_torch_dtype(args.data_type),
                tp_rank=tp_rank, tp_world_size=tp_world_size,
            ),
            kv_type=args.llm_kv_type, page_size=args.page_size, device="meta",
        )
        return [state]

    def _init_att_backend1(self):
        role = self.att_resource_role
        if not self.att_resource_manager.get_bounded_attention_states(role):
            self.prefill_att_backend1 = None
            self.decode_att_backend1 = None
            return
        from lightllm.common.basemodel.attention.create_linear_utils import (
            get_qwen35_linear_prefill_att_backend_class,
            get_qwen35_linear_decode_att_backend_class,
        )

        linear_prefill_att_backend_class = get_qwen35_linear_prefill_att_backend_class(index=1)
        linear_decode_att_backend_class = get_qwen35_linear_decode_att_backend_class(index=1)
        self.prefill_att_backend1 = linear_prefill_att_backend_class(model=self)
        self.decode_att_backend1 = linear_decode_att_backend_class(model=self)
        return
