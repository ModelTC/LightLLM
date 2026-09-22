import dataclasses
import inspect
import warnings

import torch

from .fp import Fa3AttBackend, Fa3PrefillAttState
from ..base_att import AttControl


_NEO_FA3_INSTALL_HINT = (
    "Install the Neo FA3 build from source in the current Python environment:\n"
    "  git clone --recursive https://github.com/WANDY666/flash-attention.git\n"
    "  cd flash-attention/hopper\n"
    "  FLASH_ATTENTION_FORCE_BUILD=TRUE python -m pip install "
    "--no-build-isolation --no-deps --force-reinstall .\n"
    "This replaces a standalone flash_attn_3 / flash_attn_interface installation; "
    "LightLLM's regular FA3 uses the separate sgl_kernel package."
)


try:
    from flash_attn_interface import flash_attn_with_kvcache as flash_attn_with_kvcache_neo

    # Neo FA3 accepts the exclusive visible KV end for each image query.
    _sig = inspect.signature(flash_attn_with_kvcache_neo)
    if "image_token_end" not in _sig.parameters:
        raise ImportError("flash_attn_interface is missing image_token_end support (need the Neo build)")

    HAS_FLASH_ATTN_INTERFACE = True
except ImportError as exc:
    warnings.warn(
        f"Neo FA3 is unavailable: {exc}. "
        "Automatic Neo prefill selection will fall back to Triton. "
        "To select Triton explicitly, use --llm_prefill_att_backend triton.\n" + _NEO_FA3_INSTALL_HINT
    )
    flash_attn_with_kvcache_neo = None
    HAS_FLASH_ATTN_INTERFACE = False


class NeoFa3AttBackend(Fa3AttBackend):
    def create_att_prefill_state(self, infer_state) -> "NeoFa3PrefillAttState":
        return NeoFa3PrefillAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class NeoFa3PrefillAttState(Fa3PrefillAttState):
    def prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        # neo_chat*: image-token bidirectional attention requires flash_attn_interface
        # (sgl_kernel's flash_attn_with_kvcache does not support image_token_end).
        if not HAS_FLASH_ATTN_INTERFACE:
            raise ImportError("Neo prefill requires FA3 with image_token_end support.\n" + _NEO_FA3_INSTALL_HINT)
        if self.infer_state.b_image_token_end is None:
            raise ValueError("Neo prefill requires b_image_token_end to describe image attention spans")
        o = flash_attn_with_kvcache_neo(
            q=q,
            k_cache=k.view(-1, self.backend.infer_page_size, k.shape[1], k.shape[2]),
            v_cache=v.view(-1, self.backend.infer_page_size, v.shape[1], v.shape[2]),
            page_table=self.page_table,
            cache_seqlens=self.infer_state.b_seq_len,
            cu_seqlens_q=self.cu_seqlens_q,
            cu_seqlens_k_new=self.cu_seqlens_k,
            max_seqlen_q=self.infer_state.max_q_seq_len,
            softmax_scale=1.0 / (q.shape[-1] ** 0.5),
            causal=self.causal,
            window_size=(-1, -1),
            softcap=0.0,
            k_descale=None,
            v_descale=None,
            return_softmax_lse=False,
            image_token_end=self.infer_state.b_image_token_end,
        )
        return o
