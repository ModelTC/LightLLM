import torch
from lightllm.platform.base.ops import WeightOps
from lightllm.common.basemodel.triton_kernel.embedding import npu_embedding
from typing import Optional, Tuple


class AscendWeightOps(WeightOps):

    def _embedding_impl(
        self,
        *,
        input_ids: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        vob_start_id: int,
        vob_end_id: int,
    ) -> torch.Tensor:
        npu_embedding(input_ids, weight, vob_start_id, vob_end_id, out)
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
        if gate_value is not None:
            raise NotImplementedError("gate_value is not supported for rms_norm on ascend")

        import torch_npu

        _out = torch_npu.npu_rms_norm(input, weight, epsilon=eps)[0]
        out.copy_(_out)
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
        raise NotImplementedError("layer_norm is not supported on ascend")

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
        import torch_npu

        head_dim_q = w_q.shape[0]
        head_dim_k = w_k.shape[0]
        flat_q = q.reshape(-1, head_dim_q)
        flat_k = k.reshape(-1, head_dim_k)
        _q = torch_npu.npu_rms_norm(flat_q, w_q, epsilon=eps)[0]
        _k = torch_npu.npu_rms_norm(flat_k, w_k, epsilon=eps)[0]
        _q = _q.view(q.shape)
        _k = _k.view(k.shape)
        q.copy_(_q)
        k.copy_(_k)
        return q, k