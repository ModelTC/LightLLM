from lightllm.common.attention.backends.dense.triton.kernels.prefill.varlen import _flash_attention_triton_fwd
from lightllm.common.attention.backends.dense.varlen import BaseVarlenAttentionBackend


class TritonVarlenAttentionBackend(BaseVarlenAttentionBackend):
    provider = "triton"
    dependency = "triton"
    supports_graph = True

    def forward(self, inputs, *, out):
        q, k, v, cu_seqlens, max_seqlen = inputs
        if q.shape[0] == 0:
            return out
        _flash_attention_triton_fwd(
            q,
            k,
            v,
            out,
            cu_seqlens,  # q k v cu_seqlens,
            max_seqlen,
        )
        return out
