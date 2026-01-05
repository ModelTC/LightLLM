import dataclasses
import torch
from .base_att import BaseAttBackend, BasePrefillAttState, BaseDecodeAttState, AttControl
from typing import Optional, Tuple


class Int8kvTritonAttBackend(BaseAttBackend):
    def __init__(self, quant_group_size: int):
        self.quant_group_size: int = quant_group_size

    def create_att_prefill_state(self, infer_state) -> "Int8kvTritonPrefillAttState":
        return Int8kvTritonPrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state) -> "Int8kvTritonDecodeAttState":
        return Int8kvTritonDecodeAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class Int8kvTritonPrefillAttState(BasePrefillAttState):
    def init_state(self):
        pass

    def copy_for_prefill_cuda_graph(self, new_state: "Int8kvTritonPrefillAttState"):
        pass

    def prefill_att(
        self,
        q: torch.Tensor,
        k: Tuple[torch.Tensor, torch.Tensor],
        v: Tuple[torch.Tensor, torch.Tensor],
        layer_weight,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        assert att_control.use_alibi is False

        return self._nomarl_prefill_att(q=q, k=k, v=v, layer_weight=layer_weight, alloc_func=alloc_func)

    def _nomarl_prefill_att(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_weight, alloc_func=torch.empty
    ):
        raise NotImplementedError("not impl")


@dataclasses.dataclass
class Int8kvTritonDecodeAttState(BaseDecodeAttState):
    def init_state(self):
        pass

    def copy_for_decode_cuda_graph(self, new_state: "Int8kvTritonDecodeAttState"):
        pass

    def decode_att(
        self,
        q: torch.Tensor,
        k: Tuple[torch.Tensor, torch.Tensor],
        v: Tuple[torch.Tensor, torch.Tensor],
        layer_weight,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ):
        assert att_control.use_alibi is False
        q = q
        k, k_scale = k
        v, v_scale = v
        if k_scale.ndim == 3 and k_scale.shape[2] == 1:
            return self._per_head_quant_decode_stage3_att(
                q=q,
                k=k,
                k_scale=k_scale,
                v=v,
                v_scale=v,
                layer_weight=layer_weight,
                alloc_func=alloc_func,
            )
        else:
            raise NotImplementedError("not support decode att")

    def _per_head_quant_decode_stage3_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        k_scale: torch.Tensor,
        v: torch.Tensor,
        v_scale: torch.Tensor,
        layer_weight,
        alloc_func=torch.empty,
    ):
        raise NotImplementedError("error")
