import torch
import triton
from lightllm.common.basemodel.tp_reduce import ordered_tp_sum
from lightllm.common.basemodel.cuda_graph import CudaGraph
from lightllm.common.basemodel.hidden_collector import FinalHiddenCollector, NoopHiddenCollector
from lightllm.common.basemodel.triton_kernel.fused_moe.topk_select import fused_topk
from lightllm.distributed import dist_group_manager
from lightllm.models.qwen4_exp.layer_infer.transformer_layer_infer import Qwen4ExpTransformerLayerInfer
from ..triton_kernel.replicated_moe import expert_down, expert_up, pack_routes, sum_partials, sum_routes
from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import silu_and_mul_fwd


_REPLICATED_DECODE_CONFIG = {
    "hidden_size": 2560,
    "num_experts": 512,
    "moe_intermediate_size": 640,
    "shared_expert_intermediate_size": 640,
    "num_experts_per_tok": 10,
    "norm_topk_prob": True,
    "hidden_act": "silu",
}


def supports_fixed_decode(model):
    args = model.args
    return (
        model.tp_world_size_ == 8
        and args.dp == 1
        and args.nnodes == 1
        and model.data_type == torch.bfloat16
        and args.llm_kv_type == "fp8kv_spt"
        and args.enable_ep_moe
        and args.enable_tpsp_mix_mode
        and args.mtp_mode == "eagle_with_att"
        and args.mtp_step == 3
        and not args.mtp_dynamic_verify
        and not args.enable_decode_microbatch_overlap
        and args.ssm_state_mode == "replay"
        and args.linear_att_ssm_data_type == "float32"
        and args.ssm_replay_config.get("cache_len", 16) == 16
        and args.ssm_replay_config.get("compute_mode", "precompute") == "precompute"
        and model.quant_cfg.quant_type == "none"
        and not model.quant_cfg.quantized_weight
        and all(method == "none" for layer in model.quant_cfg.quant_cfg.values() for method in layer.values())
        and all(model.config.get(key) == value for key, value in _REPLICATED_DECODE_CONFIG.items())
        and torch.cuda.get_device_capability() == (9, 0)
    )


def supports_replicated_decode(model):
    args = model.args
    if (
        not model.fixed_decode_config
        or model.is_mtp_draft_model
        or args.enable_torch_memory_saver
        or args.enable_return_routed_experts
        or args.auto_update_redundancy_expert
        or type(model.hidden_collector_prototype) not in (FinalHiddenCollector, NoopHiddenCollector)
    ):
        return False
    symmetric = dist_group_manager.get_group(0).symm_mem_reduce
    if (
        symmetric is None
        or symmetric.disabled
        or symmetric.dtype != torch.bfloat16
        or symmetric.world_size != 8
        or symmetric.max_size < 64 * 2560 * 2 * 8
        or not hasattr(torch.ops.symm_mem, "multimem_all_gather_out")
    ):
        return False
    local_ids = list(range(model.tp_rank_ * 64, (model.tp_rank_ + 1) * 64))
    for layer in model.trans_layers_weight:
        experts = getattr(layer, "experts", None)
        if (
            experts is None
            or experts.quant_method.method_name != "none"
            or experts.local_logic_expert_ids_list != local_ids
            or experts.num_fused_shared_experts
            or experts.redundancy_expert_num
            or experts.scoring_func != "softmax"
            or experts.routed_scaling_factor != 1.0
            or experts.per_expert_scale is not None
            or experts.e_score_correction_bias is not None
        ):
            return False
        for pack, shape in ((experts.w13, (64, 1280, 2560)), (experts.w2, (64, 2560, 640))):
            if pack.weight.dtype != torch.bfloat16 or pack.weight.shape != shape or not pack.weight.is_contiguous():
                return False
        if layer.gate_up_proj.quant_method.method_name != "none" or layer.down_proj.quant_method.method_name != "none":
            return False
    return True


class ReplicatedDecoderLayer(Qwen4ExpTransformerLayerInfer):
    def _tpsp_sp_split(self, input, infer_state):
        return input

    def _compute_shared_expert(self, input, infer_state, layer_weight):
        shape = (input.shape[0], input.shape[1])
        if getattr(self, "shared_shape", None) != shape:
            self.shared_shape = shape
            self.shared_up = torch.empty(input.shape[0], 1280, dtype=input.dtype, device=input.device)
            self.shared_mid = torch.empty(input.shape[0], 640, dtype=input.dtype, device=input.device)
            self.shared_output = torch.empty_like(input)
            self.shared_gate = torch.empty(input.shape[0], 1, dtype=input.dtype, device=input.device)
        layer_weight.gate_up_proj.mm(input, out=self.shared_up)
        silu_and_mul_fwd(self.shared_up, self.shared_mid)
        layer_weight.down_proj.mm(self.shared_mid, out=self.shared_output)
        layer_weight.shared_expert_gate.mm(input, out=self.shared_gate)
        return self.shared_output

    def _tpsp_allgather(self, input, infer_state):
        return input

    def _tpsp_reduce(self, input, infer_state):
        return ordered_tp_sum(input, infer_state.dist_group, self.tp_world_size_)

    def _moe_ffn_edp(self, input, infer_state, layer_weight):
        tokens, hidden = input.shape
        experts, inter, topk = 64, 640, 10
        if getattr(self, "workspace_tokens", None) != tokens:
            self.shared_stream = torch.cuda.Stream()
            self.workspace_tokens = tokens
            self.routes = torch.empty(experts, tokens * topk, dtype=torch.int32, device="cuda")
            self.masked_ids = torch.empty(tokens, topk, dtype=torch.int32, device="cuda")
            self.counts = torch.empty(experts, dtype=torch.int32, device="cuda")
            self.mid = torch.empty(tokens * topk, inter, dtype=input.dtype, device="cuda")
            self.down = torch.empty(tokens * topk, hidden, dtype=input.dtype, device="cuda")
            self.partial = torch.empty(tokens, hidden, dtype=input.dtype, device="cuda")
        self.shared_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.shared_stream):
            shared = self._compute_shared_expert(input, infer_state, layer_weight)
        weights, ids = fused_topk(input, layer_weight.moe_gate.mm(input), topk, True)
        mask = infer_state.qwen4_exp_metadata.token_mask
        count = tokens * topk
        pack_routes[(experts,)](
            ids,
            mask,
            self.masked_ids,
            self.routes,
            self.counts,
            tokens,
            topk,
            self.tp_rank_ * experts,
            triton.next_power_of_2(count),
            num_warps=4 if count > 256 else 1,
        )
        ids = self.masked_ids
        expert_up[(experts, triton.cdiv(inter, 32))](
            input,
            layer_weight.experts.w13.weight,
            self.mid,
            self.routes,
            self.counts,
            hidden,
            inter,
            tokens,
            topk,
            16,
            32,
            256,
            num_warps=4,
            num_stages=3,
        )
        expert_down[(experts, triton.cdiv(hidden, 64))](
            self.mid,
            layer_weight.experts.w2.weight,
            self.down,
            self.routes,
            self.counts,
            inter,
            hidden,
            tokens,
            topk,
            1,
            16,
            64,
            128,
            num_warps=4,
            num_stages=3,
        )
        sum_routes[(tokens, triton.cdiv(hidden, 256))](
            self.down,
            weights,
            ids,
            self.partial,
            hidden,
            topk,
            self.tp_rank_ * experts,
            experts,
            triton.next_power_of_2(topk),
            num_warps=4,
        )
        gathered = infer_state.dist_group.symm_mem_reduce.ordered_all_gather(self.partial)
        combined = torch.empty_like(input)
        torch.cuda.current_stream().wait_stream(self.shared_stream)
        sum_partials[(tokens, triton.cdiv(hidden, 512))](
            gathered,
            ids,
            combined,
            hidden,
            tokens,
            topk,
            experts,
            512,
            10,
            num_warps=4,
            enable_fp_fusion=False,
            Shared=shared,
            Gate=self.shared_gate,
        )
        return combined


class ReplicatedDecodeGraph(CudaGraph):
    @staticmethod
    def gen_cuda_graph_batch_sizes(*args, **kwargs):
        sizes = CudaGraph.gen_cuda_graph_batch_sizes(*args, **kwargs)
        return sorted(set(sizes) | set(range(4, min(max(sizes), 64) + 1, 4)))
