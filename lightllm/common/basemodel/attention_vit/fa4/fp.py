import torch

from lightllm.common.basemodel.attention_vit.base_att import BaseVitAttBackend
from lightllm.utils.fa4_utils import (
    ensure_fa4_available,
    ensure_fa4_supported_gpu,
    _flash_attn_fwd,
)


class Fa4VitAttBackend(BaseVitAttBackend):
    def __init__(self):
        ensure_fa4_available()
        ensure_fa4_supported_gpu()

    @staticmethod
    def _vit_att_fwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> None:
        head_dim = q.shape[-1]
        return _flash_attn_fwd(
            out=o,
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=head_dim ** -0.5,
            causal=False,
            return_lse=False,
        )
