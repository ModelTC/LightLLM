import os
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from types import SimpleNamespace

import torch
from typing import Callable, Optional
from lightllm.common.quantization.no_quant import WeightPack
from lightllm.common.quantization.quantize_method import QuantizationMethod
from .base_impl import FuseMoeBaseImpl


def _use_sglang_triton_moe() -> bool:
    return os.getenv("LIGHTLLM_USE_SGLANG_TRITON_MOE", "0").upper() in {
        "ON",
        "TRUE",
        "1",
    }


@lru_cache(maxsize=1)
def _get_sglang_fused_experts_impl():
    """Load SGLang's tuned Triton MoE without its server RuntimeContext.

    LightLLM and SGLang share the same standard block-FP8 expert layout, but
    SGLang's standalone kernel helpers consult two unrelated process-global
    server flags.  Supply their disabled defaults locally so the kernel can be
    used as an optional backend without initializing an SGLang server.
    """

    from sglang.srt.layers.moe.moe_runner.triton_utils import (
        fused_moe as sglang_fused_moe,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils import (
        fused_moe_triton_config as sglang_fused_moe_config,
    )

    # Despite the upstream flag name, for this in-place standalone invocation
    # the optimization only folds the top-k expert sum into the down GEMM.  It
    # does not perform the TP all-reduce, which remains owned by LightLLM.
    enable_fused_sum = os.getenv("LIGHTLLM_SGLANG_FUSED_MOE_SUM", "1").upper() in {
        "ON",
        "TRUE",
        "1",
    }
    standalone_exec = SimpleNamespace(
        moe=SimpleNamespace(enable_fused_moe_sum_all_reduce=enable_fused_sum),
        deterministic=SimpleNamespace(enable_deterministic_inference=False),
    )
    sglang_fused_moe.get_exec = lambda: standalone_exec
    sglang_fused_moe_config.get_exec = lambda: standalone_exec
    from sglang.srt.layers.moe.moe_runner.triton_utils import override_config

    return sglang_fused_moe, override_config


@contextmanager
def _override_sglang_moe_configs(
    sglang_fused_moe,
    override_config,
    up_config: dict,
    down_config: Optional[dict],
):
    """Temporarily select independently measured up/down MoE configs."""

    if down_config is None:
        with override_config(up_config):
            yield
        return

    original = sglang_fused_moe.try_get_optimal_moe_config

    def resolve_config(*args, return_down_config=False, **kwargs):
        if return_down_config:
            return up_config, (down_config, None)
        return up_config

    sglang_fused_moe.try_get_optimal_moe_config = resolve_config
    try:
        yield
    finally:
        sglang_fused_moe.try_get_optimal_moe_config = original


@lru_cache(maxsize=None)
def _get_sglang_triton_moe_configs(
    w13_shape: tuple[int, ...],
    w2_shape: tuple[int, ...],
    topk: int,
    is_prefill: bool,
    token_count: int,
):
    """Return measured H100 (up, down) configs for GLM-5's TP8 shape."""

    if "H100" not in torch.cuda.get_device_name(torch.cuda.current_device()):
        return None
    if is_prefill:
        if token_count < 8192:
            up_config = {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 32,
                "num_warps": 4,
                "num_stages": 3,
            }
            return up_config, None

        # The two GLM expert projections have different output widths and
        # diverge at large prefill batches.  Measurements on H100 TP8 show
        # that N=128 remains best for the 4096->512 up projection, while the
        # 256->4096 down projection crosses over to N=64 at roughly 24K
        # tokens.  Selecting them independently avoids making the faster up
        # projection pay for the down projection's narrower tile.
        up_config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 64,
            "num_warps": 4,
            "num_stages": 3,
        }
        down_config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64 if token_count >= 24576 else 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 32 if token_count >= 24576 else 8,
            "num_warps": 4,
            "num_stages": 3,
        }
        return up_config, down_config
    if w13_shape != (289, 512, 4096):
        return None
    if w2_shape != (289, 4096, 256) or topk != 9:
        return None
    up_config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 16,
        "num_warps": 4,
        "num_stages": 3,
    }
    # The draft graph has eight physical rows; the MTP2 verify graph has 24.
    # Separate down-projection sweeps found different winners for those two
    # hot shapes while retaining BLOCK_SIZE_M=16 for shared route alignment.
    if token_count <= 8:
        down_config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 3,
        }
    else:
        down_config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 16,
            "num_warps": 4,
            "num_stages": 2,
        }
    return up_config, down_config


class FuseMoeTriton(FuseMoeBaseImpl):
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

    def create_workspace(self):
        return None

    def _select_experts(
        self,
        input_tensor: torch.Tensor,
        router_logits: torch.Tensor,
        correction_bias: Optional[torch.Tensor],
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: int,
        num_expert_group: int,
        scoring_func: str,
        per_expert_scale: Optional[torch.Tensor] = None,
        shared_expert_gate: Optional[torch.Tensor] = None,
    ):
        """Select experts and return topk weights and ids."""
        from lightllm.common.basemodel.triton_kernel.fused_moe.topk_select import select_experts

        topk_weights, topk_ids = select_experts(
            hidden_states=input_tensor,
            router_logits=router_logits,
            correction_bias=correction_bias,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            scoring_func=scoring_func,
        )
        if self.routed_scaling_factor != 1.0:
            topk_weights.mul_(self.routed_scaling_factor)
        if per_expert_scale is not None:
            topk_weights = topk_weights * per_expert_scale[topk_ids.to(torch.long)].to(topk_weights.dtype)
        origin_topk_ids = topk_ids
        if self.num_fused_shared_experts > 0:
            from lightllm.common.basemodel.triton_kernel.fused_moe.append_shared_expert_topk import (
                append_fused_shared_experts,
            )

            topk_weights, topk_ids = append_fused_shared_experts(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_expert_start_id=self.n_routed_experts,
                num_fused_shared_experts=self.num_fused_shared_experts,
                shared_expert_gate=shared_expert_gate,
            )
        return topk_weights, topk_ids, origin_topk_ids

    def _fused_experts(
        self,
        input_tensor: torch.Tensor,
        w13: WeightPack,
        w2: WeightPack,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        router_logits: Optional[torch.Tensor] = None,
        is_prefill: bool = False,
    ):
        w13_weight, w13_scale = w13.weight, w13.weight_scale
        w2_weight, w2_scale = w2.weight, w2.weight_scale
        use_fp8_w8a8 = w13_weight.dtype == torch.float8_e4m3fn

        if _use_sglang_triton_moe():
            block_size = getattr(self.quant_method, "block_size", None)
            if not use_fp8_w8a8 or block_size != 128:
                raise RuntimeError(
                    "LIGHTLLM_USE_SGLANG_TRITON_MOE currently requires "
                    "block-wise FP8 expert weights with block size 128"
                )
            if getattr(self, "swiglu_limit", None) is not None and getattr(self, "swiglu_clamp_up_add_one", True):
                raise RuntimeError("SGLang Triton MoE does not support clamp_up_add_one=True")

            sglang_fused_moe, override_config = _get_sglang_fused_experts_impl()
            tuned_configs = _get_sglang_triton_moe_configs(
                tuple(w13_weight.shape),
                tuple(w2_weight.shape),
                topk_ids.shape[1],
                bool(is_prefill),
                input_tensor.shape[0],
            )
            config_context = (
                _override_sglang_moe_configs(
                    sglang_fused_moe,
                    override_config,
                    *tuned_configs,
                )
                if tuned_configs is not None
                else nullcontext()
            )
            with config_context:
                sglang_fused_moe.fused_experts_impl(
                    hidden_states=input_tensor,
                    w1=w13_weight,
                    w2=w2_weight,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    inplace=True,
                    use_fp8_w8a8=True,
                    w1_scale=w13_scale,
                    w2_scale=w2_scale,
                    block_shape=[block_size, block_size],
                    routed_scaling_factor=1.0,
                    filter_expert=False,
                    swiglu_limit=getattr(self, "swiglu_limit", None),
                    gate_up_interleaved=False,
                )
            return input_tensor

        from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe import fused_experts

        fused_experts(
            hidden_states=input_tensor,
            w1=w13_weight,
            w2=w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            use_fp8_w8a8=use_fp8_w8a8,
            w1_scale=w13_scale,
            w2_scale=w2_scale,
            limit=getattr(self, "swiglu_limit", None),
            alpha=(getattr(self, "swiglu_alpha", 1.0) if getattr(self, "swiglu_limit", None) is not None else None),
            clamp_up_add_one=getattr(self, "swiglu_clamp_up_add_one", True),
        )
        return input_tensor

    def __call__(
        self,
        input_tensor: torch.Tensor,
        router_logits: torch.Tensor,
        w13: WeightPack,
        w2: WeightPack,
        correction_bias: Optional[torch.Tensor],
        scoring_func: str,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: int,
        num_expert_group: int,
        is_prefill: Optional[bool] = None,
        # Callback to capture MoE topk expert ids (routed experts metadata).
        moe_capture_callback: Optional[Callable[[torch.Tensor], None]] = None,
        per_expert_scale: Optional[torch.Tensor] = None,
        shared_expert_gate: Optional[torch.Tensor] = None,
    ):
        topk_weights, topk_ids, origin_topk_ids = self._select_experts(
            input_tensor=input_tensor,
            router_logits=router_logits,
            correction_bias=correction_bias,
            top_k=top_k,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            scoring_func=scoring_func,
            per_expert_scale=per_expert_scale,
            shared_expert_gate=shared_expert_gate,
        )

        if moe_capture_callback is not None:
            moe_capture_callback(origin_topk_ids)

        output = self._fused_experts(
            input_tensor=input_tensor,
            w13=w13,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=router_logits,
            is_prefill=is_prefill,
        )
        return output
