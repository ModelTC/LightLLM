import dataclasses
import inspect
import warnings

import torch

from lightllm.common.attention.api import UnsupportedAttention
from lightllm.common.attention.api.backend import LayerAttentionBinding
from lightllm.common.attention.api.inputs import DenseInputs
from lightllm.common.attention.backends.dense.fa3.backend import Fa3AttBackend, Fa3PrefillContext

_NEO_FA3_INSTALL_HINT = (
    "Install the Neo FA3 build from source in the current Python environment:\n"
    "  git clone --recursive https://github.com/WANDY666/flash-attention.git\n"
    "  cd flash-attention/hopper\n"
    "  FLASH_ATTENTION_FORCE_BUILD=TRUE python -m pip install --no-build-isolation --no-deps --force-reinstall .\n"
    "This replaces a standalone flash_attn_3 / flash_attn_interface installation; LightLLM's regular"
    " FA3 uses the separate sgl_kernel package."
)
try:
    from flash_attn_interface import flash_attn_with_kvcache as flash_attn_with_kvcache_neo

    _sig = inspect.signature(flash_attn_with_kvcache_neo)
    if "image_token_end" not in _sig.parameters:
        raise ImportError("flash_attn_interface is missing image_token_end support (need the Neo build)")
    HAS_FLASH_ATTN_INTERFACE = True
except ImportError as exc:
    warnings.warn(
        f"Neo FA3 is unavailable: {exc}. Automatic Neo prefill selection will fall back to Triton. "
        "To select Triton explicitly, use --llm_prefill_att_backend triton.\n" + _NEO_FA3_INSTALL_HINT
    )
    flash_attn_with_kvcache_neo = None
    HAS_FLASH_ATTN_INTERFACE = False


@dataclasses.dataclass(slots=True)
class NeoFa3PrefillContext(Fa3PrefillContext):
    pass


class NeoFa3AttBackend(Fa3AttBackend):
    @classmethod
    def check_support(cls, spec, caps=None, execution=None) -> None:
        super().check_support(spec, caps, execution)
        if spec.phase.value == "prefill" and not HAS_FLASH_ATTN_INTERFACE:
            raise UnsupportedAttention("Neo FA3 requires the image-token flash attention interface")

    provider = "fa3"
    cache_formats = frozenset({"None"})

    def _allocate_prefill_context(self, capacity, capture, workspace):
        return NeoFa3PrefillContext(capacity, capture, workspace)

    def _forward_prefill(self, layer: LayerAttentionBinding, context, inputs: DenseInputs, *, out=None) -> torch.Tensor:
        q = inputs.query
        k = layer.params.key
        v = layer.params.value
        if not HAS_FLASH_ATTN_INTERFACE:
            raise ImportError("Neo prefill requires FA3 with image_token_end support.\n" + _NEO_FA3_INSTALL_HINT)
        if context.batch.b_image_token_end is None:
            raise ValueError("Neo prefill requires b_image_token_end to describe image attention spans")
        o = flash_attn_with_kvcache_neo(
            q=q,
            k_cache=k.view(-1, self.infer_page_size, k.shape[1], k.shape[2]),
            v_cache=v.view(-1, self.infer_page_size, v.shape[1], v.shape[2]),
            page_table=context.page_table,
            cache_seqlens=context.batch.b_seq_len,
            cu_seqlens_q=context.cu_seqlens_q,
            cu_seqlens_k_new=context.cu_seqlens_k,
            max_seqlen_q=context.batch.max_q_seq_len,
            softmax_scale=1.0 / q.shape[-1] ** 0.5,
            causal=context.causal,
            window_size=(-1, -1),
            softcap=0.0,
            k_descale=None,
            v_descale=None,
            return_softmax_lse=False,
            image_token_end=context.batch.b_image_token_end,
        )
        return o
