import os
from typing import Dict, Optional, Tuple

import torch

from lightllm.common.quantization.quantize_method import QuantizationMethod, WeightPack
from lightllm.distributed import dist_group_manager
from lightllm.utils.envs_utils import get_deepep_num_max_dispatch_tokens_per_rank
from lightllm.utils.log_utils import init_logger

from .deepgemm_impl import FuseMoeDeepGEMM

logger = init_logger(__name__)

try:
    import deep_gemm
    from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8

    HAS_DEEPGEMM = True
except ImportError:
    HAS_DEEPGEMM = False

try:
    import torch.distributed._symmetric_memory  # noqa: F401

    HAS_SYMM_MEM = True
except Exception:
    HAS_SYMM_MEM = False


def _env_flag_is_on(name: str, default: str = "False") -> bool:
    return os.getenv(name, default).upper() in ["1", "TRUE", "ON"]


def is_mega_moe_available(quant_method: QuantizationMethod, enable_ep_moe: bool) -> bool:
    if not enable_ep_moe or not _env_flag_is_on("LIGHTLLM_ENABLE_MEGA_MOE"):
        return False
    if not HAS_DEEPGEMM or not HAS_SYMM_MEM or not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] < 10:
        return False
    return quant_method.method_name in {
        "none",
        "deepgemm-fp8w8a8-b128",
        "vllm-fp8w8a8-b128",
        "fp8w8a8-b128",
    }


class FuseMoeMegaGEMM(FuseMoeDeepGEMM):
    def __init__(
        self,
        n_routed_experts: int,
        num_fused_shared_experts: int,
        routed_scaling_factor: float,
        quant_method: QuantizationMethod,
        redundancy_expert_num: int,
        redundancy_expert_ids_tensor: torch.Tensor,
        routed_expert_counter_tensor: torch.Tensor,
        auto_update_redundancy_expert: bool,
    ):
        super().__init__(
            n_routed_experts=n_routed_experts,
            num_fused_shared_experts=num_fused_shared_experts,
            routed_scaling_factor=routed_scaling_factor,
            quant_method=quant_method,
            redundancy_expert_num=redundancy_expert_num,
            redundancy_expert_ids_tensor=redundancy_expert_ids_tensor,
            routed_expert_counter_tensor=routed_expert_counter_tensor,
            auto_update_redundancy_expert=auto_update_redundancy_expert,
        )
        self._mega_symm_buffer_cache: Dict[Tuple[int, int, int, int, int, int], object] = {}
        self._mega_weight_cache: Dict[
            Tuple[int, int, int, int], Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        ] = {}
        self._mega_log_once = False

    def _can_use_mega_moe_runtime(
        self,
        input_tensor: torch.Tensor,
        w13: WeightPack,
        w2: WeightPack,
        topk_ids: torch.Tensor,
    ) -> bool:
        if not is_mega_moe_available(self.quant_method, enable_ep_moe=True):
            return False
        if self.redundancy_expert_num > 0 or self.num_fused_shared_experts > 0:
            return False
        if not torch.distributed.is_initialized():
            return False
        if input_tensor.dtype != torch.bfloat16 or input_tensor.ndim != 2:
            return False
        if topk_ids.shape[1] > 32:
            return False
        if input_tensor.shape[1] % 128 != 0:
            return False
        if w13.weight.ndim != 3 or w2.weight.ndim != 3:
            return False
        if w13.weight.shape[1] % 128 != 0 or w13.weight.shape[2] % 128 != 0:
            return False
        if w2.weight.shape[1] % 128 != 0 or w2.weight.shape[2] % 128 != 0:
            return False
        return True

    def _dequantize_block_fp8_weight(self, weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
        block_size = 128
        expanded_scale = weight_scale.repeat_interleave(block_size, dim=-2).repeat_interleave(block_size, dim=-1)
        expanded_scale = expanded_scale[..., : weight.shape[-2], : weight.shape[-1]]
        return (weight.float() * expanded_scale).to(torch.bfloat16).contiguous()

    def _get_grouped_bf16_weight(self, weight_pack: WeightPack) -> torch.Tensor:
        method_name = self.quant_method.method_name
        if method_name == "none":
            return weight_pack.weight.to(torch.bfloat16).contiguous()
        if method_name in {"deepgemm-fp8w8a8-b128", "vllm-fp8w8a8-b128", "fp8w8a8-b128"}:
            return self._dequantize_block_fp8_weight(weight_pack.weight, weight_pack.weight_scale)
        raise RuntimeError(f"Unsupported quantization method for mega moe: {method_name}")

    def _cast_grouped_weights_to_fp4(self, bf16_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_groups, n, k = bf16_weights.shape
        packed_weight = torch.empty((num_groups, n, k // 2), device=bf16_weights.device, dtype=torch.int8)
        scale = torch.empty((num_groups, n, k // 32), device=bf16_weights.device, dtype=torch.float32)
        for expert_idx in range(num_groups):
            packed_weight[expert_idx], scale[expert_idx] = per_token_cast_to_fp4(
                bf16_weights[expert_idx].contiguous(),
                use_ue8m0=True,
                gran_k=32,
            )
        scale = deep_gemm.transform_sf_into_required_layout(scale, n, k, (1, 32), num_groups)
        return packed_weight, scale

    def _get_transformed_mega_weights(
        self, w13: WeightPack, w2: WeightPack
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        cache_key = (
            w13.weight.data_ptr(),
            0 if w13.weight_scale is None else w13.weight_scale.data_ptr(),
            w2.weight.data_ptr(),
            0 if w2.weight_scale is None else w2.weight_scale.data_ptr(),
        )
        if cache_key in self._mega_weight_cache:
            return self._mega_weight_cache[cache_key]

        l1_weights = self._cast_grouped_weights_to_fp4(self._get_grouped_bf16_weight(w13))
        l2_weights = self._cast_grouped_weights_to_fp4(self._get_grouped_bf16_weight(w2))
        transformed = deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights)
        self._mega_weight_cache[cache_key] = transformed
        return transformed

    def _get_symm_buffer(self, device: torch.device, num_topk: int, hidden: int, intermediate_hidden: int):
        num_max_tokens_per_rank = get_deepep_num_max_dispatch_tokens_per_rank()
        cache_key = (
            device.index,
            self.n_routed_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
        )
        if cache_key in self._mega_symm_buffer_cache:
            return self._mega_symm_buffer_cache[cache_key]

        mega_group = dist_group_manager.get_mega_moe_group()
        symm_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            mega_group,
            self.n_routed_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
        )
        self._mega_symm_buffer_cache[cache_key] = symm_buffer
        return symm_buffer

    def _run_mega_moe(
        self,
        input_tensor: torch.Tensor,
        w13: WeightPack,
        w2: WeightPack,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens, hidden = input_tensor.shape
        intermediate_hidden = w2.weight.shape[2]
        if num_tokens > get_deepep_num_max_dispatch_tokens_per_rank():
            raise RuntimeError("Mega MoE only supports token count <= NUM_MAX_DISPATCH_TOKENS_PER_RANK in V1")

        symm_buffer = self._get_symm_buffer(input_tensor.device, topk_ids.shape[1], hidden, intermediate_hidden)
        l1_weights, l2_weights = self._get_transformed_mega_weights(w13, w2)
        qinput, qinput_scale = per_token_cast_to_fp8(
            input_tensor,
            use_ue8m0=True,
            gran_k=32,
            use_packed_ue8m0=True,
        )
        symm_buffer.x[:num_tokens].copy_(qinput)
        symm_buffer.x_sf[:num_tokens].copy_(qinput_scale)
        symm_buffer.topk_idx[:num_tokens].copy_(topk_ids)
        symm_buffer.topk_weights[:num_tokens].copy_(topk_weights)

        output = torch.empty((num_tokens, hidden), dtype=input_tensor.dtype, device=input_tensor.device)
        deep_gemm.fp8_fp4_mega_moe(
            output,
            l1_weights,
            l2_weights,
            symm_buffer,
            cumulative_local_expert_recv_stats=None,
            activation="swiglu",
            fast_math=True,
        )
        if not self._mega_log_once:
            logger.info("Enable DeepGEMM Mega MoE fused path for EP-MoE standard execution.")
            self._mega_log_once = True
        return output

    def _fused_experts(
        self,
        input_tensor: torch.Tensor,
        w13: WeightPack,
        w2: WeightPack,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        router_logits: Optional[torch.Tensor] = None,
        is_prefill: Optional[bool] = None,
    ):
        if not self._can_use_mega_moe_runtime(input_tensor, w13, w2, topk_ids):
            return super()._fused_experts(
                input_tensor=input_tensor,
                w13=w13,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                router_logits=router_logits,
                is_prefill=is_prefill,
            )

        try:
            return self._run_mega_moe(input_tensor, w13, w2, topk_weights, topk_ids.to(torch.long))
        except Exception as exc:
            logger.warning(f"DeepGEMM Mega MoE fallback to grouped path because of runtime error: {exc}")
            return super()._fused_experts(
                input_tensor=input_tensor,
                w13=w13,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                router_logits=router_logits,
                is_prefill=is_prefill,
            )
