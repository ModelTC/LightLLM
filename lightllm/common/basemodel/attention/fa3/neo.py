import dataclasses
from logging import warning
import torch
import warnings

from .fp import Fa3AttBackend
from ..base_att import BasePrefillAttState, AttControl
from typing import Optional, TYPE_CHECKING, Tuple


try:
    from flash_attn_interface import flash_attn_with_kvcache as flash_attn_with_kvcache_neo
    import inspect

    # Verify this is the neo-patched FA3 build (with image_token_tag support),
    _sig = inspect.signature(flash_attn_with_kvcache_neo)
    if "image_token_tag" not in _sig.parameters:
        raise ImportError("flash_attn_interface found but missing image_token_tag support (need neo build)")

    HAS_FLASH_ATTN_INTERFACE = True
except ImportError:
    warnings.warn(
        f"flash_attn_interface (fa3-neo) is not found which is required for image_token_tag bidirectional attention, "
        f"will fallback to triton attention even if requested fa3 for neo"
    )
    flash_attn_with_kvcache_neo = None
    HAS_FLASH_ATTN_INTERFACE = False


class NeoFa3AttBackend(Fa3AttBackend):
    def create_att_prefill_state(self, infer_state) -> "NeoFa3PrefillAttState":
        return NeoFa3PrefillAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class NeoFa3PrefillAttState(BasePrefillAttState):
    def init_state(self):
        pass

    def prefill_att(
        self,
        q: torch.Tensor,
        k: Tuple[torch.Tensor, torch.Tensor],
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        from lightllm.models.neo_chat_moe.infer_struct import NeoChatInferStateInfo

        self.infer_state: NeoChatInferStateInfo

        # neo_chat*: image-token bidirectional attention requires flash_attn_interface
        # (sgl_kernel's flash_attn_with_kvcache does not support image_token_tag).
        if self.infer_state.image_token_end is None or not HAS_FLASH_ATTN_INTERFACE:
            raise ImportError(
                "flash_attn_interface (fa3-neo) is required for image_token_tag bidirectional "
                "attention. Install it or to use the triton attention fallback."
            )
        o = flash_attn_with_kvcache_neo(
            q=q,
            k_cache=k.view(k.shape[0], 1, k.shape[1], k.shape[2]),
            v_cache=v.view(v.shape[0], 1, v.shape[1], v.shape[2]),
            page_table=self.page_table,
            cache_seqlens=self.infer_state.b_seq_len,
            cu_seqlens_q=self.cu_seqlens_q,
            cu_seqlens_k_new=self.cu_seqlens_k,
            max_seqlen_q=self.infer_state.max_q_seq_len,
            softmax_scale=1.0 / (q.shape[-1] ** 0.5),
            causal=True,
            window_size=(-1, -1),
            softcap=0.0,
            k_descale=None,
            v_descale=None,
            return_softmax_lse=False,
            **{"image_token_tag": self.infer_state.b_image_token_end},
        )
        return o
