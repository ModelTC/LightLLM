from lightllm.utils.log_utils import init_logger
from typing import Callable, List, Optional, Tuple

logger = init_logger(__name__)

try:
    from vllm._custom_ops import *
    from vllm.model_executor.layers.fused_moe import FusedMoE

    select_experts = FusedMoE.select_experts
except ImportError:
    try:
        from lightllm.common.vllm_kernel._ops import *

        def select_experts(
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
            top_k: int,
            use_grouped_topk: bool,
            renormalize: bool,
            topk_group: Optional[int] = None,
            num_expert_group: Optional[int] = None,
            custom_routing_function: Optional[Callable] = None,
        ):
            from lightllm.common.fused_moe.topk_select import fused_topk, grouped_topk

            # DeekSeekv2 uses grouped_top_k
            if use_grouped_topk:
                assert topk_group is not None
                assert num_expert_group is not None
                topk_weights, topk_ids = grouped_topk(
                    hidden_states=hidden_states,
                    gating_output=router_logits,
                    topk=top_k,
                    renormalize=renormalize,
                    num_expert_group=num_expert_group,
                    topk_group=topk_group,
                )
            elif custom_routing_function is None:
                topk_weights, topk_ids = fused_topk(
                    hidden_states=hidden_states, gating_output=router_logits, topk=top_k, renormalize=renormalize
                )
            else:
                topk_weights, topk_ids = custom_routing_function(
                    hidden_states=hidden_states, gating_output=router_logits, topk=top_k, renormalize=renormalize
                )

            return topk_weights, topk_ids

    except ImportError:
        logger.error("vllm or lightllm_kernel is not installed, you can't use custom ops")

try:
    from lightllm.common.vllm_kernel._ops import init_custom_gather_ar
    from lightllm.common.vllm_kernel._ops import all_gather
    from lightllm.common.vllm_kernel._ops import allgather_dispose
    from lightllm.common.vllm_kernel._ops import allgather_register_buffer
    from lightllm.common.vllm_kernel._ops import allgather_get_graph_buffer_ipc_meta
    from lightllm.common.vllm_kernel._ops import allgather_register_graph_buffers
except ImportError:
    logger.error("lightllm_kernel is not installed, you can't use custom allgather")
