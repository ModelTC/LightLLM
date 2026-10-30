import torch
from typing import Optional
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
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.models.qwen3next.att_resource_config import Qwen3NextResourceConfig

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
    def create_bounded_attention_definition(cls, model_config, args, *, model_logical_layer_ids, tp_rank=0, tp_world_size=1):
        from lightllm.common.att_resources.bounded_att import create_gdn_state, ConvSSMStateConfig
        from lightllm.common.basemodel.basemodel_utils import get_resource_config

        config = get_resource_config(model_config, model_class=cls)
        cfg = Qwen3NextResourceConfig.from_model_config(config, args, include_draft=False)
        logical_layer_ids = tuple(layer for i, layer in enumerate(model_logical_layer_ids) if (i + 1) % cfg.full_attention_interval)
        if not logical_layer_ids:
            return None
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
            request_capacity=1, device="meta",
        )
        return state

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

    def _init_att_resource_manager(self):
        from .att_resource_manager import Qwen3NextAttResourceManager
        from lightllm.common.basemodel.basemodel_utils import init_model_full_attention, init_model_bounded_attention

        assert self.config["num_attention_heads"] % self.tp_world_size_ == 0
        start_args: StartArgs = get_env_start_args()
        self.linear_config = Qwen3NextResourceConfig.from_model_config(self.config, start_args, include_draft=False)

        if start_args.mtp_step and start_args.enable_prefill_decode_mixed:
            raise ValueError("MTP linear attention does not support mixed prefill/decode")
        if not self.is_mtp_draft_model:
            self.att_resource_manager = Qwen3NextAttResourceManager(
                self.linear_config, self.max_req_num + 1, start_args.mtp_step, tp_rank=self.tp_rank_, logical_layer_ids=()
            )
        init_model_bounded_attention(self)
        init_model_full_attention(self)

    @classmethod
    def _full_attention_options(cls, config, args):
        if args.llm_kv_type not in ("None", "fp8kv_sph", "fp8kv_spt"):
            raise ValueError(f"Qwen3Next does not support llm_kv_type={args.llm_kv_type}")
        cfg = Qwen3NextResourceConfig.from_model_config(config, args, include_draft=False)
        return dict(
            heads=cfg.full_att_all_num_kv_heads, dim=cfg.full_att_head_dim, logical_layer_groups=tuple((logical_layer_id,) for logical_layer_id in cfg.get_full_att_logical_layer_ids())
        )

    def _init_att_backend1(self):
        role = self.att_resource_role
        if getattr(self.att_resource_manager, f"{role}_gdn_state", None) is None:
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
