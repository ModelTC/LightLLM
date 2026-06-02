import torch
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Union


class BackendOps(ABC):

    @staticmethod
    def _ensure_out(
        out: Optional[torch.Tensor],
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: Union[str, torch.device],
        alloc_func: Callable = torch.empty,
    ) -> torch.Tensor:
        if out is None:
            return alloc_func(shape, dtype=dtype, device=device)
        return out

    def embedding(
        self,
        *,
        input_ids: torch.Tensor,
        weight: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        alloc_func: Callable = torch.empty,
        vob_start_id: int = 0,
        vob_end_id: Optional[int] = None,
    ) -> torch.Tensor:
        if vob_end_id is None:
            vob_end_id = weight.shape[0]
        out = self._ensure_out(
            out,
            shape=(input_ids.shape[0], weight.shape[1]),
            dtype=weight.dtype,
            device=weight.device,
            alloc_func=alloc_func,
        )
        return self._embedding_impl(
            input_ids=input_ids,
            weight=weight,
            out=out,
            vob_start_id=vob_start_id,
            vob_end_id=vob_end_id,
        )

    @abstractmethod
    def _embedding_impl(
        self,
        *,
        input_ids: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        vob_start_id: int,
        vob_end_id: int,
    ) -> torch.Tensor:
        pass

    def lm_head(
        self,
        *,
        input: torch.Tensor,
        weight: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        alloc_func: Callable = torch.empty,
    ) -> torch.Tensor:
        out = self._ensure_out(
            out,
            shape=(weight.shape[0], input.shape[1]),
            dtype=input.dtype,
            device=input.device,
            alloc_func=alloc_func,
        )
        return self._lm_head_impl(input=input, weight=weight, out=out)

    @abstractmethod
    def _lm_head_impl(
        self,
        *,
        input: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        pass

    def rms_norm(
        self,
        *,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        out: Optional[torch.Tensor] = None,
        alloc_func: Callable = torch.empty,
        gate_value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self._ensure_out(
            out,
            shape=input.shape,
            dtype=input.dtype,
            device=input.device,
            alloc_func=alloc_func,
        )
        return self._rms_norm_impl(
            input=input, weight=weight, eps=eps, out=out, gate_value=gate_value
        )

    @abstractmethod
    def _rms_norm_impl(
        self,
        *,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        out: torch.Tensor,
        gate_value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pass

    def layer_norm(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
        out: Optional[torch.Tensor] = None,
        alloc_func: Callable = torch.empty,
    ) -> torch.Tensor:
        out = self._ensure_out(
            out,
            shape=input.shape,
            dtype=input.dtype,
            device=input.device,
            alloc_func=alloc_func,
        )
        return self._layer_norm_impl(input=input, weight=weight, bias=bias, eps=eps, out=out)

    @abstractmethod
    def _layer_norm_impl(
        self,
        *,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
        out: torch.Tensor,
    ) -> torch.Tensor:
        pass

    def qk_rms_norm(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        w_q: torch.Tensor,
        w_k: torch.Tensor,
        eps: float,
        fp32_multiply: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._qk_rms_norm_impl(q=q, k=k, w_q=w_q, w_k=w_k, eps=eps, fp32_multiply=fp32_multiply)

    @abstractmethod
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
        pass
