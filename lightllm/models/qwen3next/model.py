import torch
from typing_extensions import override
from lightllm.models.registry import ModelRegistry
from lightllm.models.qwen3_moe.model import Qwen3MOEModel
from lightllm.models.qwen3next.layer_weights.transformer_layer_weight import Qwen3NextTransformerLayerWeight
from lightllm.models.qwen3next.layer_infer.transformer_layer_infer import Qwen3NextTransformerLayerInfer
from lightllm.utils.log_utils import init_logger
from lightllm.distributed.communication_op import dist_group_manager
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.models.qwen3next.mem_manager import Qwen3NextMemoryManager, MambaStateBufferConfig
from lightllm.models.qwen3next.req_manager import Qwen3NextReqManager

logger = init_logger(__name__)


@ModelRegistry("qwen3_next")
class Qwen3NextTpPartModel(Qwen3MOEModel):
    # weight class
    transformer_weight_class = Qwen3NextTransformerLayerWeight

    # infer class
    transformer_layer_infer_class = Qwen3NextTransformerLayerInfer

    def __init__(self, kvargs) -> None:
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
        mtp_step = get_env_start_args().mtp_step
        self.num_linear_k_heads = self.config["linear_num_key_heads"]
        self.num_linear_v_heads = self.config["linear_num_value_heads"]
        self.head_linear_k_dim = self.config["linear_key_head_dim"]
        self.head_linear_v_dim = self.config["linear_value_head_dim"]
        conv_kernel_size = self.config["linear_conv_kernel_dim"]

        conv_dim = (
            self.head_linear_k_dim * self.num_linear_k_heads * 2 + self.head_linear_v_dim * self.num_linear_v_heads
        )

        mamba_state_buffer_config = MambaStateBufferConfig(
            conv_state_dtype=self.data_type,
            conv_state_shape=(conv_kernel_size - 1 + mtp_step, conv_dim // self.tp_world_size_),
            ssm_state_dtype=self.data_type,
            ssm_state_shape=(
                self.num_linear_v_heads // self.tp_world_size_,
                self.head_linear_k_dim,
                self.head_linear_v_dim,
            ),
        )

        self.mem_manager = Qwen3NextMemoryManager(
            size=self.max_total_token_num,
            dtype=self.data_type,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.config["head_dim"],
            layer_num=self.config["n_layer"],
            full_attention_interval=self.config["full_attention_interval"],
            max_req_num=self.max_req_num,
            mamba_state_buffer_config=mamba_state_buffer_config,
            mem_fraction=self.mem_fraction,
        )

    @override
    def _init_req_manager(self):
        create_max_seq_len = 0

        if self.batch_max_tokens is not None:
            create_max_seq_len = max(create_max_seq_len, self.batch_max_tokens)
        if self.max_seq_length is not None:
            create_max_seq_len = max(create_max_seq_len, self.max_seq_length)

        self.req_manager = Qwen3NextReqManager(self.max_req_num, create_max_seq_len, self.mem_manager)
