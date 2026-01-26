import torch
from lightllm.common.basemodel.attention_vit.base_att import BaseVitAttBackend
from lightllm.models.vit.triton_kernel.flashattention_nopad import _flash_attention_triton_fwd


class TritonVitAttBackend(BaseVitAttBackend):
    def _vit_att_fwd(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ):
        _flash_attention_triton_fwd(
            q,
            k,
            v,
            o,
            cu_seqlens,  # q k v cu_seqlens,
            max_seqlen,
        )
        return o
