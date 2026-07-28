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
from lightllm.utils.envs_utils import get_added_mtp_kv_layer_num, get_env_start_args
from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import Qwen3NextMemManager
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.common.req_manager import ReqManagerForMamba
from lightllm.common.linear_att_cache_manager.config_objs import (
    LinearAttCacheConfig,
    get_mtp_draft_full_att_layer_num,
)
from lightllm.common.basemodel.batch_objs import ModelOutput
from lightllm.distributed import all_reduce, all_reduce_residual_rmsnorm

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
        self._init_triton()
        super().__init__(kvargs)

    def _init_triton(self):
        def _triton_allocator(size: int, alignment: int, stream: Optional[int]) -> torch.Tensor:
            return torch.empty(size, device="cuda", dtype=torch.int8)

        # Set Triton allocator for TMA descriptors
        # This is required for kernels in qwen3next/triton_kernel/fla/ops/solve_tril.py
        triton.set_allocator(_triton_allocator)
        logger.info("Triton allocator set for Qwen3Next model")
        return

    def autotune_layers(self):
        return self.config["full_attention_interval"]

    def _autotune_extra_warmup(self):
        if not self.trans_layers_weight:
            return

        norm_weight = self.trans_layers_weight[0].ffn_norm_weight_
        add_rmsnorm = getattr(norm_weight, "add_rmsnorm", None)
        if add_rmsnorm is None:
            return

        hidden_dim = norm_weight.weight.shape[0]
        max_batch_size = min(self.graph_max_batch_size, self.batch_max_tokens)
        warmup_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        warmup_batch_sizes = [bs for bs in warmup_batch_sizes if bs <= max_batch_size]
        if max_batch_size not in warmup_batch_sizes:
            warmup_batch_sizes.append(max_batch_size)

        for batch_size in sorted(set(warmup_batch_sizes)):
            x = torch.zeros((batch_size, hidden_dim), dtype=self.data_type, device="cuda")
            residual = torch.zeros_like(x)
            out = torch.empty_like(x)
            add_rmsnorm(input=x, residual=residual, eps=self.layers_infer[0].eps_, out=out)
        return

    def _init_config(self):
        super()._init_config()
        self.num_kv_heads = max(self.config["num_key_value_heads"] // self.tp_world_size_, 1)

    def _init_linear_config(self):
        start_args: StartArgs = get_env_start_args()
        ssm_dtype_dict = {"bfloat16": torch.bfloat16, "float32": torch.float32}
        draft_full_att_layers = get_added_mtp_kv_layer_num()
        self.linear_config = LinearAttCacheConfig(
            tp_world_size=self.tp_world_size_,
            full_att_all_num_kv_heads=self.config["num_key_value_heads"],
            full_att_dtype=self.data_type,
            full_att_num_kv_heads=self.num_kv_heads,
            full_att_head_dim=self.config["head_dim"],
            global_linear_k_heads=self.config["linear_num_key_heads"],
            global_linear_v_heads=self.config["linear_num_value_heads"],
            num_linear_k_heads=max(1, self.config["linear_num_key_heads"] // self.tp_world_size_),
            num_linear_v_heads=max(1, self.config["linear_num_value_heads"] // self.tp_world_size_),
            head_linear_k_dim=self.config["linear_key_head_dim"],
            head_linear_v_dim=self.config["linear_value_head_dim"],
            conv_kernel_size=self.config["linear_conv_kernel_dim"],
            linear_layer_num=self.config["n_layer"]
            - (self.config["n_layer"] // self.config["full_attention_interval"]),
            conv_state_dtype=self.data_type,
            ssm_state_dtype=ssm_dtype_dict[start_args.linear_att_ssm_data_type],
            full_attention_interval=self.config["full_attention_interval"],
            all_layer_num=self.config["n_layer"],
            draft_full_att_layer_num=draft_full_att_layers,
        )
        return

    def _init_mem_manager(self):
        assert self.config["num_attention_heads"] % self.tp_world_size_ == 0
        main_full_att = self.linear_config.get_main_full_att_layer_num()
        persisted_full_att = self.linear_config.get_persisted_full_att_layer_num()

        self.mem_manager = Qwen3NextMemManager(
            size=self.max_total_token_num,
            dtype=self.data_type,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.config["head_dim"],
            full_att_layer_num=persisted_full_att,
            linear_config=self.linear_config,
            mem_fraction=self.mem_fraction,
        )
        self.mem_manager.main_full_att_layer_num = main_full_att
        self.mem_manager.draft_full_att_layers = self.linear_config.draft_full_att_layer_num

    def _init_req_manager(self):
        create_max_seq_len = 0

        if self.batch_max_tokens is not None:
            create_max_seq_len = max(create_max_seq_len, self.batch_max_tokens)
        if self.max_seq_length is not None:
            create_max_seq_len = max(create_max_seq_len, self.max_seq_length)

        self._init_linear_config()
        self.req_manager = ReqManagerForMamba(
            self.max_req_num, create_max_seq_len, None, linear_config=self.linear_config
        )
        return

    def _token_forward(self, infer_state: Qwen3NextInferStateInfo):
        input_ids = infer_state.input_ids
        input_embs = self.pre_infer.token_forward(input_ids, infer_state, self.pre_post_weight)
        input_embs = self.pre_infer._tpsp_sp_split(input=input_embs, infer_state=infer_state)

        next_att_normed = None
        for i in range(self.layers_num):
            layer: Qwen3NextTransformerLayerInfer = self.layers_infer[i]
            layer_weight: Qwen3NextTransformerLayerWeight = self.trans_layers_weight[i]

            if next_att_normed is None:
                input1 = layer._att_norm(input_embs, infer_state, layer_weight)
            else:
                input1 = next_att_normed
                next_att_normed = None

            if layer.is_linear_attention_layer:
                o = layer.token_attention_forward(input1, infer_state, layer_weight)
                input1 = layer._add_residual_ffn_norm(input_embs, o, infer_state, layer_weight)
                o = None
            else:
                q, cache_kv = layer._get_qkv(input1, infer_state, layer_weight)
                layer._post_cache_kv(cache_kv, infer_state, layer_weight)
                o = layer._token_attention_kernel(q, infer_state, layer_weight)
                q = None
                o = layer._get_o_local(o, infer_state, layer_weight)
                fused = None
                if layer.tp_world_size_ > 1:
                    fused = all_reduce_residual_rmsnorm(
                        o,
                        residual=input_embs.view(-1, layer.embed_dim_),
                        norm_weight=layer_weight.ffn_norm_weight_.weight,
                        eps=layer.eps_,
                        group=infer_state.dist_group,
                        alloc_func=layer.alloc_tensor,
                    )
                if fused is None:
                    if layer.tp_world_size_ > 1:
                        all_reduce(o, group=infer_state.dist_group)
                    input1 = layer._add_residual_ffn_norm(input_embs, o, infer_state, layer_weight)
                else:
                    input_embs, input1 = fused
                o = None

            ffn_out = layer._ffn(input1, infer_state, layer_weight)
            ffn_out = ffn_out.view(-1, layer.embed_dim_)

            if i + 1 < self.layers_num:
                next_layer: Qwen3NextTransformerLayerInfer = self.layers_infer[i + 1]
                next_layer_weight: Qwen3NextTransformerLayerWeight = self.trans_layers_weight[i + 1]
                add_rmsnorm = getattr(next_layer_weight.att_norm_weight_, "add_rmsnorm", None)
                if add_rmsnorm is not None:
                    next_att_normed = add_rmsnorm(
                        input=input_embs,
                        residual=ffn_out,
                        eps=next_layer.eps_,
                        alloc_func=next_layer.alloc_tensor,
                    )
                    continue

            input_embs.add_(ffn_out)

        last_input_embs = self.post_infer._tpsp_allgather(input=input_embs, infer_state=infer_state)
        predict_logits: torch.Tensor = self.post_infer.token_forward(
            last_input_embs, infer_state=infer_state, layer_weight=self.pre_post_weight
        )

        model_output = ModelOutput(logits=predict_logits.contiguous())
        if self.is_mtp_mode:
            input_embs = self.pre_infer._tpsp_allgather(input=input_embs, infer_state=infer_state)
            model_output.mtp_main_output_hiddens = input_embs.contiguous()

        if infer_state.is_cuda_graph:
            model_output.to_no_ref_tensor()

        return model_output
