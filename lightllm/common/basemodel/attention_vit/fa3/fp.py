import dataclasses
import torch
from lightllm.common.basemodel.attention_vit.base_att import BaseVitAttBackend


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

        head_dim = q.shape[-1]
        softmax_scale = head_dim ** -0.5
        window_size = (-1, -1)
        torch.ops.sgl_kernel.fwd.default(
            q,
            k,
            v,
            None,  # k_new
            None,  # v_new
            None,  # qv
            o,  # out
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
            0.0,
            is_rotary_interleaved=False,
            scheduler_metadata=None,
            num_splits=1,
            pack_gqa=None,
            sm_margin=0,
            sinks=None,
        )

        return o
