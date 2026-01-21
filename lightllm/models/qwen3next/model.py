import torch
from typing import Optional
from typing_extensions import override
import triton
from lightllm.models.registry import ModelRegistry
from lightllm.models.qwen3_moe.model import Qwen3MOEModel
from lightllm.models.qwen3next.layer_weights.transformer_layer_weight import (
    Qwen3NextFullAttentionTransformerLayerWeight,
    Qwen3NextGatedDeltaNetTransformerLayerWeight,
)
from lightllm.models.qwen3next.layer_infer.transformer_layer_infer import (
    Qwen3NextFullAttentionTransformerLayerInfer,
    Qwen3NextGatedDeltaNetTransformerLayerInfer,
)
from lightllm.models.qwen3next.layer_infer.post_layer_infer import Qwen3NextPostLayerInfer
from lightllm.models.qwen3next.infer_struct import Qwen3NextInferStateInfo
from lightllm.utils.log_utils import init_logger
from lightllm.distributed.communication_op import dist_group_manager
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.models.qwen3next.mem_manager import Qwen3NextHybridMemManager
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.common.req_manager import ReqManagerForMamba
from lightllm.common.basemodel.layer_weights.hf_load_utils import load_hf_weights

logger = init_logger(__name__)


@ModelRegistry("qwen3_next")
class Qwen3NextTpPartModel(Qwen3MOEModel):

    post_layer_infer_class = Qwen3NextPostLayerInfer
    infer_state_class = Qwen3NextInferStateInfo

    is_hybrid_attention = True  # Indicates model uses hybrid (full + linear) attention
    use_buffer_manager = True  # Indicates model needs per-request buffer management for linear attention states

    _triton_allocator_initialized = False  # Class-level flag to ensure allocator is set only once

    @classmethod
    def get_radix_cache_class(cls):
        """Return HybridRadixCache for hybrid attention models."""
        from lightllm.server.router.dynamic_prompt.hybrid_radix_cache import HybridRadixCache

        return HybridRadixCache

    @classmethod
    def _init_triton_allocator(cls):
        """Initialize Triton allocator once for all instances."""
        if cls._triton_allocator_initialized:
            return

        def _triton_allocator(size: int, alignment: int, stream: Optional[int]) -> torch.Tensor:
            return torch.empty(size, device="cuda", dtype=torch.int8)

        # Set Triton allocator for TMA descriptors
        # This is required for kernels in qwen3next/triton_kernel/fla/ops/solve_tril.py
        triton.set_allocator(_triton_allocator)
        cls._triton_allocator_initialized = True
        logger.info("Triton allocator set for Qwen3Next model")

    def __init__(self, kvargs) -> None:
        self.mem_manager: Qwen3NextHybridMemManager = None
        self._init_triton_allocator()
        super().__init__(kvargs)

    @override
    def autotune_layers(self):
        return self.config["full_attention_interval"]

    @override
    def _init_config(self):
        super()._init_config()
        self.num_kv_heads = max(self.config["num_key_value_heads"] // self.tp_world_size_, 1)

    @override
    def _init_custom(self):
        super()._init_custom()
        dist_group_manager.new_deepep_group(self.config["num_experts"], self.config["hidden_size"])

    @override
    def _init_mem_manager(self):
        assert self.config["num_attention_heads"] % self.tp_world_size_ == 0

        start_args: StartArgs = get_env_start_args()
        mamba_cache_size = start_args.mamba_cache_size
        if mamba_cache_size is not None:
            assert (
                mamba_cache_size >= start_args.running_max_req_size
            ), "mamba_cache_size must be greater than running_max_req_size"

        self.num_linear_k_heads = self.config["linear_num_key_heads"]
        self.num_linear_v_heads = self.config["linear_num_value_heads"]
        self.head_linear_k_dim = self.config["linear_key_head_dim"]
        self.head_linear_v_dim = self.config["linear_value_head_dim"]

        conv_kernel_size = self.config["linear_conv_kernel_dim"]
        conv_dim = (
            self.head_linear_k_dim * self.num_linear_k_heads * 2 + self.head_linear_v_dim * self.num_linear_v_heads
        )

        ssm_dtype_dict = {"bfloat16": torch.bfloat16, "float32": torch.float32}
        if start_args.mamba_ssm_data_type not in ssm_dtype_dict:
            raise ValueError(
                f"Invalid mamba_ssm_data_type: {start_args.mamba_ssm_data_type}."
                f" Must be one of {list(ssm_dtype_dict.keys())}"
            )

        self.mem_manager = Qwen3NextHybridMemManager(
            full_attn_cache_size=self.max_total_token_num,
            linear_attn_cache_size=mamba_cache_size,
            dtype=self.data_type,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.config["head_dim"],
            layer_num=self.config["n_layer"],
            mtp_layer_num=start_args.mtp_step,
            full_attention_interval=self.config["full_attention_interval"],
            conv_state_dtype=self.data_type,
            conv_state_shape=(conv_dim // self.tp_world_size_, conv_kernel_size - 1),
            ssm_state_dtype=ssm_dtype_dict[start_args.mamba_ssm_data_type],
            ssm_state_shape=(
                self.num_linear_v_heads // self.tp_world_size_,
                self.head_linear_k_dim,
                self.head_linear_v_dim,
            ),
            max_req_num=self.max_req_num,
            mem_fraction=self.mem_fraction,
        )

    @override
    def _init_req_manager(self):
        create_max_seq_len = 0

        if self.batch_max_tokens is not None:
            create_max_seq_len = max(create_max_seq_len, self.batch_max_tokens)
        if self.max_seq_length is not None:
            create_max_seq_len = max(create_max_seq_len, self.max_seq_length)

        self.req_manager = ReqManagerForMamba(self.max_req_num, create_max_seq_len, self.mem_manager)

    @override
    def _init_weights(self):
        self.pre_post_weight = self.pre_and_post_weight_class(self.data_type, network_config=self.config)
        num_full_attention_layers = self.config["full_attention_interval"]
        self.trans_layers_weight = [
            Qwen3NextFullAttentionTransformerLayerWeight(
                i,
                self.data_type,
                network_config=self.config,
                quant_cfg=self.quant_cfg,
            )
            if (i + 1) % num_full_attention_layers == 0
            else Qwen3NextGatedDeltaNetTransformerLayerWeight(
                i,
                self.data_type,
                network_config=self.config,
                quant_cfg=self.quant_cfg,
            )
            for i in range(self.config["n_layer"])
        ]
        load_hf_weights(
            self.data_type,
            weight_dir=self.weight_dir_,
            pre_post_layer=self.pre_post_weight,
            transformer_layer_list=self.trans_layers_weight,
            weight_dict=self.weight_dict,
        )
        self.pre_post_weight.verify_load()
        [weight.verify_load() for weight in self.trans_layers_weight]

    def _init_infer_layer(self):
        self.pre_infer = self.pre_layer_infer_class(network_config=self.config)
        self.post_infer = self.post_layer_infer_class(network_config=self.config)
        num_full_attention_layers = self.config["full_attention_interval"]
        self.layers_infer = [
            Qwen3NextFullAttentionTransformerLayerInfer(i, network_config=self.config)
            if (i + 1) % num_full_attention_layers == 0
            else Qwen3NextGatedDeltaNetTransformerLayerInfer(i, network_config=self.config)
            for i in range(self.config["n_layer"])
        ]
