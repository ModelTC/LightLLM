from lightllm.utils.device_utils import is_metax
import torch
import torch.nn.functional as F

try:
    from xformers import ops as xformers_ops
    from xformers.ops import fmha
except ImportError:
    xformers_ops = None
    fmha = None

from lightllm.common.basemodel.attention_vit.base_att import BaseVitAttBackend


class XformersVitAttBackend(BaseVitAttBackend):
    @torch.no_grad()
    @staticmethod
    def _vit_att_fwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        assert q.ndim == k.ndim == v.ndim == o.ndim == 3
        assert cu_seqlens is not None and cu_seqlens.ndim == 1
        assert q.shape == k.shape == v.shape == o.shape

        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int64).tolist()
        seqlens = [int(L) for L in seqlens if int(L) > 0]

        if len(seqlens) == 0:
            return o
        if max_seqlen:
            assert max(seqlens) <= max_seqlen

        # The version of xformers on metex is 0.0.22 (nv is 0.0.32.post1), no device param
        if is_metax():
            attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        else:
            attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens, device=q.device)

        q_ = q.unsqueeze(0)  # [1, T, H, D]
        k_ = k.unsqueeze(0)  # [1, T, H, D]
        v_ = v.unsqueeze(0)  # [1, T, H, D]

        out = xformers_ops.memory_efficient_attention(q_, k_, v_, attn_bias=attn_bias, p=0.0)
        o.copy_(out.squeeze(0))  # [T, H, D]
        return o
