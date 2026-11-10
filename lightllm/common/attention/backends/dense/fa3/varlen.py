import torch
from lightllm.common.attention.backends.dense.varlen import BaseVarlenAttentionBackend


class Fa3VarlenAttentionBackend(BaseVarlenAttentionBackend):
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
            q,
            k,
            v,
            None,  # k_new
            None,  # v_new
            None,  # qv
            out,
            cu_seqlens,
            cu_seqlens,
            None,  # cu_seqlens_k_new
            None,
            None,
            max_seqlen,
            max_seqlen,
            None,  # page_table,
            None,  # kv_batch_idx
            None,  # leftpad_k
            None,  # rotary cos
            None,  # rotary sin
            None,  # seqlens_rotary
            None,
            None,
            None,
            softmax_scale,
            False,
            window_size[0],
            window_size[1],
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
