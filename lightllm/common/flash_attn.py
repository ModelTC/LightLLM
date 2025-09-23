import torch
from typing import List, Optional, Tuple, Union
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def get_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


try:
    import flash_attn_3._C  # Registers operators with PyTorch

    flash_attn_3_mtp = torch.ops.flash_attn_3

    def flash_attn_with_kvcache_mtp(
        q,
        k,
        v,
        k_new: Optional[torch.Tensor] = None,
        v_new: Optional[torch.Tensor] = None,
        q_v: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        cu_seqlens_k_new: Optional[torch.Tensor] = None,
        seqused_q: Optional[torch.Tensor] = None,
        seqused_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        page_table: Optional[torch.Tensor] = None,
        cache_batch_idx: Optional[torch.Tensor] = None,
        cache_leftpad: Optional[torch.Tensor] = None,
        rotary_cos: Optional[torch.Tensor] = None,
        rotary_sin: Optional[torch.Tensor] = None,
        rotary_seqlens: Optional[torch.Tensor] = None,
        q_descale: Optional[torch.Tensor] = None,
        k_descale: Optional[torch.Tensor] = None,
        v_descale: Optional[torch.Tensor] = None,
        softmax_scale=None,
        is_causal=False,
        window_size=(-1, -1),
        softcap=0.0,  # 0.0 means deactivated
        is_rotary_interleaved=True,
        scheduler_metadata=None,
        num_splits=0,
        pack_gqa=None,
        sm_margin=0,
        mtp_step=0,
    ):
        assert k.stride(-1) == 1, "k must have contiguous last dimension"
        assert v.stride(-1) == 1, "v must have contiguous last dimension"
        if softmax_scale is None:
            softmax_scale = (q.shape[-1] + (q_v.shape[-1] if q_v is not None else 0)) ** (-0.5)
        seqused_k = get_contiguous(seqused_k)

        q, k, k_new, v_new = [get_contiguous(x) for x in (q, k, k_new, v_new)]
        v = v.contiguous() if v.stride(-1) != 1 and v.stride(-3) != 1 else v
        cu_seqlens_q, cu_seqlens_k_new = [get_contiguous(x) for x in (cu_seqlens_q, cu_seqlens_k_new)]
        page_table = get_contiguous(page_table)
        out, softmax_lse, *rest = flash_attn_3_mtp.fwd(
            q,
            k,
            v,
            k_new,
            v_new,
            q_v,
            None,  # out
            cu_seqlens_q,
            None,  # cu_seqlens_k
            cu_seqlens_k_new,
            None,  # seqused_q
            seqused_k,
            max_seqlen_q,
            None,  # max_seqlen_k
            page_table,
            cache_batch_idx,
            cache_leftpad,
            rotary_cos,
            rotary_sin,
            rotary_seqlens,
            q_descale,
            k_descale,
            v_descale,
            softmax_scale,
            is_causal,
            window_size[0],
            window_size[1],
            0,
            softcap,
            is_rotary_interleaved,
            scheduler_metadata,
            num_splits,
            pack_gqa,
            sm_margin,
            mtp_step,
        )
        return out

except:
    flash_attn_3_mtp = None
    flash_attn_with_kvcache_mtp = None
    logger.warning("flash_attn_3._C is not available, please install flash-attention-3 package.")
