import torch
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple
from lightllm.common.quantization.quantize_method import (
    WeightPack,
    QuantizationMethod,
)


class FuseMoeBaseImpl(ABC):
    """将逻辑专家路由与实际执行布局分离。

    融合 MoE 的调用流程如下::

        _select_experts
            -> topk_weights + logical_topk_ids
            -> moe_capture_callback(logical_topk_ids)
            -> _prepare_expert_execution
                -> 追加 shared expert，或者
                -> 将 EPLB logical ID 映射为 physical expert ID
            -> _fused_experts(topk_weights, execution_topk_ids)

    ``_select_experts`` 对所有实现都遵循同一套稳定接口：只在模型的逻辑专家
    空间中选择专家，并返回原始 logical ID。该阶段不能应用任何与实际执行相关
    的布局转换，例如追加 shared expert 行，或者映射到 EPLB 冗余 physical 行。

    capture callback 在任何布局转换之前执行，因此采集到的路由元数据始终描述
    模型的 logical expert。所有实现都必须将 logical ID 视为只读数据。
    ``_prepare_expert_execution`` 可以为实际执行布局分配新的 ID tensor，但不能
    原地修改 logical ID tensor。

    因此，``_fused_experts`` 接收的是 execution ID：普通路径中仍是未经修改的
    logical ID；融合 shared expert 路径中是追加了 shared expert 行的 ID；EPLB
    路径中则是完成映射后的 physical ID。
    """

    def __init__(
        self,
        n_routed_experts: int,
        num_fused_shared_experts: int,
        routed_scaling_factor: float,
        quant_method: QuantizationMethod,
    ):
        self.n_routed_experts = n_routed_experts
        self.num_fused_shared_experts = num_fused_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.quant_method = quant_method

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
        # Qwen3Next/Qwen3.5-MoE 在 TP 模式下将 shared expert 融合进 routed MoE。
        # 该参数是 shared_expert_gate(hidden_states) 产生的逐 token 门控 logit；
        # 追加 shared expert 时使用 sigmoid(logit) 作为其聚合权重。
        shared_expert_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        topk_weights, topk_ids = self._select_experts(
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
        )
        if moe_capture_callback is not None:
            moe_capture_callback(topk_ids)
        topk_weights, topk_ids = self._prepare_expert_execution(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_expert_gate=shared_expert_gate,
        )
        return self._fused_experts(
            input_tensor=input_tensor,
            w13=w13,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=router_logits,
            is_prefill=is_prefill,
        )

    @abstractmethod
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """在模型的逻辑专家空间中完成 top-k 路由选择。

        返回形状一致的 ``topk_weights`` 和 ``logical_topk_ids``。这里的
        ``topk_ids`` 必须始终表示模型配置中的原始 logical expert，不能追加
        shared expert，也不能映射到 EPLB physical expert。返回的 ID tensor 会
        先交给 capture callback，后续实现必须将其视为只读数据。
        """
        pass

    @abstractmethod
    def _prepare_expert_execution(
        self,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_expert_gate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """将逻辑路由结果转换为 MoE kernel 实际需要的执行布局。

        该方法在 capture callback 之后执行。每个实现都必须明确处理自己的执行
        布局：普通路径保持权重和 logical ID 不变，shared-expert 路径追加对应
        expert，EPLB 路径将 logical ID 修复为 physical ID。发生布局转换时应
        返回新的 ID tensor，不能原地修改传入的 logical ``topk_ids``；如果专家
        数量发生变化，``topk_weights`` 必须同步调整。

        ``shared_expert_gate`` 当前由 Qwen3Next/Qwen3.5-MoE 使用，其内容是
        ``shared_expert_gate(hidden_states)`` 计算出的逐 token 门控 logit。在 TP
        fused shared-expert 路径中，``sigmoid(logit)`` 会作为追加 shared expert
        的权重，使其输出按 token 动态参与 routed expert 输出的聚合；传入 ``None``
        时，普通 fused shared expert 的追加权重为 1。EP 路径不使用该参数，而是
        单独计算 shared expert，并在应用相同门控后与 routed MoE 输出相加。
        """
        pass

    @abstractmethod
    def _fused_experts(
        self,
        input_tensor: torch.Tensor,
        w13: WeightPack,
        w2: WeightPack,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        router_logits: Optional[torch.Tensor] = None,
        is_prefill: Optional[bool] = None,
    ) -> torch.Tensor:
        """根据准备完成的路由结果执行融合 MoE 计算。

        这里的 ``topk_ids`` 已处于实际执行所需的 ID 空间：普通路径为 logical
        ID，shared expert 路径包含追加的专家 ID，EPLB 路径则为 physical ID。
        实现只能读取路由 ID，不能原地修改其内容。``is_prefill`` 只用于选择底层
        执行策略，不应再影响专家选择或 logical-to-physical 映射语义。
        """
        pass
