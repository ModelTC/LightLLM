import dataclasses
import torch
from .base_att import BaseAttBackend, BasePrefillAttState, BaseDecodeAttState, AttControl
from typing import Tuple


class MlaTritonAttBackend(BaseAttBackend):
    def create_att_prefill_state(self, infer_state) -> "MlaTritonPrefillAttState":
        return MlaTritonPrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state) -> "MlaTritonDecodeAttState":
        return MlaTritonDecodeAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class MlaTritonPrefillAttState(BasePrefillAttState):
    def init_state(self):
        pass

    def copy_for_prefill_cuda_graph(self, new_state: "MlaTritonPrefillAttState"):
        pass

    def prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        assert att_control.use_sliding_window is False and att_control.use_att_sink is False
        return self._mla_prefill_att(q=q, k=k, v=v, att_control=att_control, alloc_func=alloc_func)

    def _mla_prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl,
        alloc_func=torch.empty,
    ):
        pass


@dataclasses.dataclass
class MlaTritonDecodeAttState(BaseDecodeAttState):
    def init_state(self):
        pass

    def copy_for_decode_cuda_graph(self, new_state: "MlaTritonDecodeAttState"):
        pass

    def decode_att(
        self,
        q: Tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ):
        assert (
            att_control.use_sliding_window is False
            and att_control.use_att_sink is False
            and att_control.use_alibi is False
        )
        assert v is None
        q_nope, q_rope = q
        return self._mla_decode_att(
            q_nope=q_nope,
            q_rope=q_rope,
            kv=k,
            att_control=att_control,
            alloc_func=alloc_func,
        )

    def _mla_decode_att(
        self,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        kv: torch.Tensor,
        att_control: AttControl,
        alloc_func=torch.empty,
    ):
        assert att_control.mla_decode
        softmax_scale = att_control.mla_prefill_dict["softmax_scale"]

        from ..triton_kernel.mla_att.decode_att import gqa_token_decode_attention_flash_decoding

        qk_rope_head_dim = 64

        out = gqa_token_decode_attention_flash_decoding(
            q_nope=q_nope,
            q_rope=q_rope,
            kv_nope=kv[:, :, :qk_rope_head_dim],
            kv_rope=kv[:, :, -qk_rope_head_dim:],
            infer_state=self.infer_state,
            softmax_scale=softmax_scale,
            alloc_tensor_func=alloc_func,
        )
        return out
