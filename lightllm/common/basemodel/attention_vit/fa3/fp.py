import dataclasses
import torch
from lightllm.common.basemodel.attention_vit.base_att import BaseVitAttBackend
from lightllm.utils.sgl_utils import flash_attn_varlen_func


class Fa3VitAttBackend(BaseVitAttBackend):
    @staticmethod
    def _vit_att_fwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> None:

        flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=False,
            out=o,
        )
        return o
