import torch

from lightllm.common.attention.api.varlen import validate_varlen_output
from lightllm.common.attention.backends.dense.varlen import BaseVarlenAttentionOp


class Fa3VarlenAttentionOp(BaseVarlenAttentionOp):
    provider = "fa3"
    dependency = "sgl_kernel.flash_attn"
    min_device_capability = (9, 0)
    activation_dtypes = frozenset({torch.float16, torch.bfloat16})
    max_head_size = 256
    head_alignment = 8
    supports_graph = True

    def __init__(self):
        self._op = None

    def forward(self, inputs, *, out):
        validate_varlen_output(inputs, out)
        inputs.validate()
        q, k, v, cu_seqlens, max_seqlen = inputs
        tokens, _, head_dim = q.shape
        if tokens == 0:
            return out
        if self._op is None:
            import sgl_kernel.flash_attn

            self._op = torch.ops.sgl_kernel.fwd.default
        softmax_scale = head_dim**-0.5
        window_size = (-1, -1)
        self._op(
            q=q,
            k=k,
            v=v,
            k_new=None,
            v_new=None,
            q_v=None,
            out=out,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            cu_seqlens_k_new=None,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            page_table=None,
            kv_batch_idx=None,
            leftpad_k=None,
            rotary_cos=None,
            rotary_sin=None,
            seqlens_rotary=None,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            softmax_scale=softmax_scale,
            is_causal=False,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            attention_chunk=0,
            softcap=0.0,
            is_rotary_interleaved=False,
            scheduler_metadata=None,
            num_splits=1,
            pack_gqa=None,
            sm_margin=0,
            sinks=None,
            sparse_mask_fine=None,
            only_qv=False,
        )

        return out
