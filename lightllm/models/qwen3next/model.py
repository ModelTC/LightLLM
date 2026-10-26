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
from lightllm.common.kv_cache_mem_manager import (
    FP8StaticPerHeadQuantMemManager,
    FP8StaticPerTensorQuantMemManager,
    MemoryManager,
)
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.common.cache_coordinator import HybridCacheCoordinator
from lightllm.common.linear_state import LinearStateManager
from lightllm.common.state_cache_manager import LinearAttCacheConfig

logger = init_logger(__name__)


class Qwen3NextTpPartModel(Qwen3MOEModel):

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
    def get_mem_manager_class(cls):
        start_args = get_env_start_args()
        if start_args.llm_kv_type == "None":
            mem_manager_class = MemoryManager
        elif start_args.llm_kv_type == "fp8kv_sph":
            mem_manager_class = FP8StaticPerHeadQuantMemManager
        elif start_args.llm_kv_type == "fp8kv_spt":
            mem_manager_class = FP8StaticPerTensorQuantMemManager
        else:
            raise ValueError(f"Qwen3Next does not support llm_kv_type={start_args.llm_kv_type}")
        return mem_manager_class

    def _init_mem_manager(self):
        assert self.config["num_attention_heads"] % self.tp_world_size_ == 0
        self.mem_manager = self.get_mem_manager_class()(
            size=self.max_total_token_num,
            dtype=self.data_type,
            head_num=self.num_kv_heads,
            head_dim=self.config["head_dim"],
            layer_num=self.linear_config.get_full_att_kv_layer_num_with_draft_model(),
            mem_fraction=self.mem_fraction,
        )
        self.mem_manager.set_kv_layer_mapping(self.linear_config.get_full_att_kv_layer_mapping())

    def _init_cache_coordinator(self):
        if self.args.mtp_step > 0:
            assert self.args.enable_prefill_decode_mixed is False
        self.linear_config = LinearAttCacheConfig.from_model_config(self.config, self.args)
        state = LinearStateManager(self.max_req_num, self.linear_config, self.args)
        self.cache_coordinator = HybridCacheCoordinator(linear_state_manager=state)

    def _init_att_backend1(self):
        # MTP draft models only keep full-attention layers and reuse the main
        # model's managers, so they do not need LinearAttBackend.
        if getattr(self, "is_mtp_draft_model", False):
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
