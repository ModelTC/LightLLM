import os
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

    # radix cache class
    radix_cache_class = HybridRadixCache

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
        self.num_linear_k_heads = self.config["linear_num_key_heads"] // self.tp_world_size_
        self.num_linear_v_heads = self.config["linear_num_value_heads"] // self.tp_world_size_
        self.head_linear_k_dim = self.config["linear_key_head_dim"]
        self.head_linear_v_dim = self.config["linear_value_head_dim"]
        conv_kernel_size = self.config["linear_conv_kernel_dim"]
        ssm_dtype_dict = {"bfloat16": torch.bfloat16, "float32": torch.float32}
        self.mem_manager = Qwen3NextHybridMemManager(
            full_attn_cache_size=self.max_total_token_num,
            linear_attn_cache_size=start_args.mamba_cache_size,
            dtype=self.data_type,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.config["head_dim"],
            layer_num=self.config["n_layer"],
            full_attention_interval=self.config["full_attention_interval"],
            conv_state_dtype=self.data_type,
            ssm_state_dtype=ssm_dtype_dict[start_args.mamba_ssm_data_type],
            conv_kernel_size=conv_kernel_size,
            num_linear_k_heads=self.num_linear_k_heads,
            num_linear_v_heads=self.num_linear_v_heads,
            head_linear_k_dim=self.head_linear_k_dim,
            head_linear_v_dim=self.head_linear_v_dim,
            max_req_num=self.max_req_num,
            mem_fraction=self.mem_fraction,
        )

    def _init_req_manager(self):
        create_max_seq_len = 0

        if self.batch_max_tokens is not None:
            create_max_seq_len = max(create_max_seq_len, self.batch_max_tokens)
        if self.max_seq_length is not None:
            create_max_seq_len = max(create_max_seq_len, self.max_seq_length)

        self.req_manager = ReqManagerForMamba(self.max_req_num, create_max_seq_len, self.mem_manager)

    @torch.no_grad()
    def _check_max_len_infer(self):
        """Extended memory check for hybrid attention models.

        In addition to the base prefill check, this validates that:
        1. The mamba cache has enough capacity for graph_max_batch_size concurrent requests.
        2. Mamba buffers are allocated during the decode stress test so GDN layers
           exercise their full memory footprint.
        """
        # Run the standard prefill check first.
        super()._check_max_len_infer()

        # Validate mamba cache capacity vs max concurrent requests.
        mamba_capacity = self.mem_manager.mamba_cache_mem_manager.size
        mtp_step = self.args.mtp_step
        buffers_per_req = mtp_step + 1
        max_concurrent = mamba_capacity // buffers_per_req
        if max_concurrent < self.graph_max_batch_size:
            logger.warning(
                f"mamba_cache_size ({mamba_capacity}) supports at most {max_concurrent} concurrent requests "
                f"(with mtp_step={mtp_step}), but graph_max_batch_size is {self.graph_max_batch_size}. "
                f"Decode check will use {max_concurrent} requests."
            )

    @torch.no_grad()
    def _check_decode_infer(self):
        """Decode check with mamba buffer allocation for hybrid models.

        Unlike the base _check_decode_infer, this override keeps mamba buffers
        allocated during the forward pass so GDN layers exercise their actual
        memory footprint (conv state updates, recurrent state reads/writes).
        """
        disable_check = os.getenv("DISABLE_CHECK_MAX_LEN_INFER", None) is not None
        if disable_check:
            return

        torch.distributed.barrier()

        batch_size = self.graph_max_batch_size
        mamba_mgr = self.mem_manager.mamba_cache_mem_manager
        buffers_per_req = self.args.mtp_step + 1
        max_mamba_reqs = mamba_mgr.size // buffers_per_req
        actual_batch = min(batch_size, max_mamba_reqs)

        if actual_batch < 2:
            logger.info("skip hybrid decode check: not enough mamba buffer slots")
            return

        try:
            logger.info(f"begin hybrid decode check with batch_size={actual_batch}")

            # Allocate request slots and mamba buffers
            req_idxs = []
            for _ in range(actual_batch):
                idx = self.req_manager.alloc()
                if idx is None:
                    break
                req_idxs.append(idx)
            actual_batch = len(req_idxs)
            if actual_batch < 2:
                self.req_manager.free_all()
                return

            req_idx_gpu = torch.tensor(req_idxs, device="cuda", dtype=torch.int64)
            self.req_manager.alloc_buffer_for_req(req_idx_gpu)

            # Allocate KV cache tokens spread across requests
            b_req_idx = torch.tensor(req_idxs, dtype=torch.int32, device="cuda")
            tokens_per_req = min(
                self.batch_max_tokens,
                max(1, self.max_total_token_num // actual_batch),
            )
            total_tokens = tokens_per_req * actual_batch
            total_tokens = min(total_tokens, self.mem_manager.can_use_mem_size)
            tokens_per_req = max(1, total_tokens // actual_batch)
            total_tokens = tokens_per_req * actual_batch

            dummy_input_ids = torch.ones(actual_batch, dtype=torch.int32, device="cuda")
            mem_indexes = self.mem_manager.alloc(total_tokens).cuda()
            b_seq_len = torch.full((actual_batch,), tokens_per_req, dtype=torch.int32, device="cuda")
            b_ready_cache_len = torch.zeros(actual_batch, dtype=torch.int32, device="cuda")
            b_mtp_index = torch.zeros(actual_batch, dtype=torch.int32, device="cuda")

            from lightllm.common.basemodel.batch_objs import ModelInput

            model_input = ModelInput(
                batch_size=actual_batch,
                total_token_num=total_tokens,
                max_q_seq_len=1,
                max_kv_seq_len=tokens_per_req,
                max_cache_len=tokens_per_req - 1,
                prefix_total_token_num=0,
                input_ids=dummy_input_ids,
                mem_indexes=mem_indexes[:actual_batch],
                b_req_idx=b_req_idx,
                b_seq_len=b_seq_len,
                b_mtp_index=b_mtp_index,
                is_prefill=False,
                b_ready_cache_len=b_ready_cache_len,
                multimodal_params=[{"images": [], "audios": []}] * actual_batch,
            )

            # Forward pass with mamba buffers allocated — GDN layers will access them
            model_output = self.forward(model_input)
            prob_out = torch.softmax(model_output.logits, dim=-1)
            del model_output
            # Simulate top_p/top_k sampling which calls probs.sort()
            prob_out.sort(dim=-1, descending=True)
            prob_out = None

            self.req_manager.free_all()
            self.mem_manager.free_all()
            logger.info(f"check hybrid decode infer batch_size={actual_batch} tokens_per_req={tokens_per_req} ok")
        except (RuntimeError, torch.OutOfMemoryError) as e:
            logger.exception(str(e))
            self.req_manager.free_all()
            self.mem_manager.free_all()
            exception_str = (
                "check hybrid decode infer fail (OOM during decode with concurrent requests), you can try:\n"
                "1. Set --graph_max_batch_size to a smaller value.\n"
                "2. Set --running_max_req_size to a smaller value.\n"
                "3. Set --mem_fraction or --max_total_token_num to a smaller value.\n"
                "4. Try increasing --mamba_cache_ratio or reducing --mamba_cache_size."
            )
            logger.error(exception_str)
            raise Exception(exception_str)
