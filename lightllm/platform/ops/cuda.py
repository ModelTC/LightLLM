import torch
from typing import Optional, Tuple
from lightllm.common.basemodel.triton_kernel.embedding import embedding
from lightllm.common.basemodel.triton_kernel.norm.gated_rmsnorm import gated_rmsnorm_forward
from lightllm.common.basemodel.triton_kernel.norm.layernorm import layernorm_forward
from lightllm.common.basemodel.triton_kernel.norm.qk_norm import qk_rmsnorm_fused_forward
from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_forward
from lightllm.platform.base.ops import BackendOps


class CudaOps(BackendOps):

    def _embedding_impl(
        self,
        *,
        input_ids: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        vob_start_id: int,
        vob_end_id: int,
    ) -> torch.Tensor:
        embedding(
            input_ids=input_ids,
            weight=weight,
            vob_start_id=vob_start_id,
            vob_end_id=vob_end_id,
            out=out,
        )
        return out

    def _lm_head_impl(
        self,
        *,
        input: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        torch.mm(weight, input, out=out)
        return out

    def _rms_norm_impl(
        self,
        *,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        out: torch.Tensor,
        gate_value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if gate_value is None:
            rmsnorm_forward(x=input, weight=weight, eps=eps, out=out)
        else:
            gated_rmsnorm_forward(
                x=input, weight=weight, bias=None, eps=eps, z=gate_value, out=out
            )
        return out

    def _layer_norm_impl(
        self,
        *,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
        out: torch.Tensor,
    ) -> torch.Tensor:
        out_ = layernorm_forward(x=input, weight=weight, bias=bias, eps=eps)
        out.copy_(out_)
        return out

    def _qk_rms_norm_impl(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        w_q: torch.Tensor,
        w_k: torch.Tensor,
        eps: float,
        fp32_multiply: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return qk_rmsnorm_fused_forward(q=q, k=k, w_q=w_q, w_k=w_k, eps=eps, fp32_multiply=fp32_multiply)
