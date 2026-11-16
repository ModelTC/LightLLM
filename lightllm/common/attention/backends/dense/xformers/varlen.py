import torch

from lightllm.common.attention.api.varlen import validate_varlen_output
from lightllm.common.attention.backends.dense.varlen import BaseVarlenAttentionOp


class XformersVarlenAttentionOp(BaseVarlenAttentionOp):
    provider = "xformers"
    dependency = "xformers.ops"

    @torch.no_grad()
    def forward(self, inputs, *, out):
        validate_varlen_output(inputs, out)
        inputs.validate()
        from xformers import ops as xformers_ops
        from xformers.ops import fmha

        q, k, v, cu_seqlens, max_seqlen = inputs
        o = out
        assert q.ndim == k.ndim == v.ndim == o.ndim == 3
        assert cu_seqlens is not None and cu_seqlens.ndim == 1
        assert q.shape == k.shape == v.shape == o.shape
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int64).tolist()
        seqlens = [int(L) for L in seqlens if int(L) > 0]

        if len(seqlens) == 0:
            return o
        if max_seqlen:
            assert max(seqlens) <= max_seqlen

        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens, device=q.device)

        q_ = q.unsqueeze(0)  # [1, T, H, D]
        k_ = k.unsqueeze(0)  # [1, T, H, D]
        v_ = v.unsqueeze(0)  # [1, T, H, D]

        out = xformers_ops.memory_efficient_attention(q_, k_, v_, attn_bias=attn_bias, p=0.0)
        o.copy_(out.squeeze(0))  # [T, H, D]
        return o
