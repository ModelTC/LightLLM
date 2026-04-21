import torch
from typing import Optional
import triton
from lightllm.models.registry import ModelRegistry
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
from lightllm.distributed.communication_op import dist_group_manager
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.models.qwen3next.mem_manager import Qwen3NextHybridMemManager
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.common.req_manager import ReqManagerForMamba
from lightllm.server.router.dynamic_prompt.hybrid_radix_cache import HybridRadixCache
from lightllm.common.linear_att_cache_manager.config_objs import LinearAttCacheConfig

logger = init_logger(__name__)


@ModelRegistry("qwen3_next")
class Qwen3NextTpPartModel(Qwen3MOEModel):

    # weight class
    pre_and_post_weight_class = Qwen3NextPreAndPostLayerWeight
    transformer_weight_class = Qwen3NextTransformerLayerWeight

    # infer class
    transformer_layer_infer_class = Qwen3NextTransformerLayerInfer

    # infer state class
    infer_state_class = Qwen3NextInferStateInfo

    def __init__(self, kvargs) -> None:
        self.mem_manager: Qwen3NextHybridMemManager = None

        def _triton_allocator(size: int, alignment: int, stream: Optional[int]) -> torch.Tensor:
            return torch.empty(size, device="cuda", dtype=torch.int8)

        # Set Triton allocator for TMA descriptors
        # This is required for kernels in qwen3next/triton_kernel/fla/ops/solve_tril.py
        triton.set_allocator(_triton_allocator)
        logger.info("Triton allocator set for Qwen3Next model")
        super().__init__(kvargs)

    def autotune_layers(self):
        return self.config["full_attention_interval"]

    def _init_config(self):
        super()._init_config()
        self.num_kv_heads = max(self.config["num_key_value_heads"] // self.tp_world_size_, 1)

    def _init_custom(self):
        super()._init_custom()
        # Only initialize DeepEP group for MoE models with num_experts
        if "num_experts" in self.config and self.config["num_experts"] > 0:
            dist_group_manager.new_deepep_group(self.config["num_experts"], self.config["hidden_size"])

    def _init_mem_manager(self):
        assert self.config["num_attention_heads"] % self.tp_world_size_ == 0
        start_args: StartArgs = get_env_start_args()
        ssm_dtype_dict = {"bfloat16": torch.bfloat16, "float32": torch.float32}
        self.linear_config = LinearAttCacheConfig(
            tp_world_size=self.tp_world_size_,
            full_att_dtype=self.data_type,
            full_att_num_kv_heads=self.num_kv_heads,
            full_att_head_dim=self.config["head_dim"],
            num_linear_k_heads=self.config["linear_num_key_heads"] // self.tp_world_size_,
            num_linear_v_heads=self.config["linear_num_value_heads"] // self.tp_world_size_,
            head_linear_k_dim=self.config["linear_key_head_dim"],
            head_linear_v_dim=self.config["linear_value_head_dim"],
            conv_kernel_size=self.config["linear_conv_kernel_dim"],
            linear_layer_num=self.config["n_layer"]
            - (self.config["n_layer"] // self.config["full_attention_interval"]),
            conv_state_dtype=self.data_type,
            ssm_state_dtype=ssm_dtype_dict[start_args.linear_att_ssm_data_type],
            full_attention_interval=self.config["full_attention_interval"],
            all_layer_num=self.config["n_layer"],
        )

        self.mem_manager = Qwen3NextHybridMemManager(
            size=self.max_total_token_num,
            dtype=self.data_type,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.config["head_dim"],
            full_att_layer_num=self.linear_config.all_layer_num - self.linear_config.linear_layer_num,
            linear_config=self.linear_config,
            mem_fraction=self.mem_fraction,
        )

    def _init_req_manager(self):
        create_max_seq_len = 0

        if self.batch_max_tokens is not None:
            create_max_seq_len = max(create_max_seq_len, self.batch_max_tokens)
        if self.max_seq_length is not None:
            create_max_seq_len = max(create_max_seq_len, self.max_seq_length)

        self.req_manager = ReqManagerForMamba(
            self.max_req_num, create_max_seq_len, self.mem_manager, linear_config=self.linear_config
        )
        return
