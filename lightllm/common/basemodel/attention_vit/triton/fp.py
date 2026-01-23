import dataclasses
import torch
from ..base_att import BaseVitAttBackend, BaseVitAttState


class TritonVitAttBackend(BaseVitAttBackend):
    def create_vit_att_state(self, infer_state) -> "TritonVitAttState":
        return TritonVitAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class TritonVitAttState(BaseVitAttState):
    def init_state(self):
        pass

    def _vit_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        alloc_func=torch.empty,
    ):
        from lightllm.models.vit.triton_kernel.flashattention_nopad import _flash_attention_triton_fwd

        _flash_attention_triton_fwd(
            q,
            k,
            v,
            o,
            cu_seqlens,  # q k v cu_seqlens,
            max_seqlen,
        )
        return
