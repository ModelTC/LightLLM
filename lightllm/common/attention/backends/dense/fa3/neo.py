from lightllm.common.attention.api.spec import CacheEncoding, AttentionFamily, AttentionPhase
import dataclasses
import inspect
from functools import lru_cache

import torch

from lightllm.common.attention.api import UnsupportedAttention
from lightllm.common.attention.api.backend import LayerAttention
from lightllm.common.attention.api.inputs import DenseInputs
from lightllm.common.attention.backends.dense.fa3.backend import (
    Fa3DensePrefillBackend,
    Fa3DenseDecodeBackend,
    Fa3PrefillContext,
)

_NEO_FA3_INSTALL_HINT = (
    "Install the Neo FA3 build from source in the current Python environment:\n"
    "  git clone --recursive https://github.com/WANDY666/flash-attention.git\n"
    "  cd flash-attention/hopper\n"
    "  FLASH_ATTENTION_FORCE_BUILD=TRUE python -m pip install --no-build-isolation --no-deps --force-reinstall .\n"
    "This replaces a standalone flash_attn_3 / flash_attn_interface installation; LightLLM's regular"
    " FA3 uses the separate sgl_kernel package."
)


@lru_cache(maxsize=1)
def get_neo_attention():
    try:
        from flash_attn_interface import flash_attn_with_kvcache

        if "image_token_end" not in inspect.signature(flash_attn_with_kvcache).parameters:
            raise ImportError("flash_attn_interface is missing image_token_end support (need the Neo build)")
    except (ImportError, OSError) as exc:
        raise UnsupportedAttention(f"Neo FA3 is unavailable: {exc}.\n" + _NEO_FA3_INSTALL_HINT) from exc
    return flash_attn_with_kvcache


@dataclasses.dataclass(slots=True)
class NeoFa3PrefillContext(Fa3PrefillContext):
    pass


class Fa3DenseNeoCommon:
    family = AttentionFamily.DENSE
    variant = "neo"

    provider = "fa3"

    cache_encodings = frozenset({CacheEncoding.PLAIN})


class Fa3DenseNeoPrefillBackend(Fa3DenseNeoCommon, Fa3DensePrefillBackend):
    phase = AttentionPhase.PREFILL

    @classmethod
    def check_environment(cls, caps):
        super().check_environment(caps)
        get_neo_attention()

    def allocate_context(self, capacity, *, capture, workspace) -> NeoFa3PrefillContext:
        return NeoFa3PrefillContext(capacity, capture, workspace)

    def forward(
        self, layer: LayerAttention, context: NeoFa3PrefillContext, inputs: DenseInputs, *, out=None
    ) -> torch.Tensor:
        q = inputs.query
        k = layer.resources.key
        v = layer.resources.value
        flash_attn_with_kvcache_neo = get_neo_attention()
        if context.batch.b_image_token_end is None:
            raise ValueError("Neo prefill requires b_image_token_end to describe image attention spans")
        o = flash_attn_with_kvcache_neo(
            q=q,
            k_cache=k.view(-1, self.kernel_page_size, k.shape[1], k.shape[2]),
            v_cache=v.view(-1, self.kernel_page_size, v.shape[1], v.shape[2]),
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


class Fa3DenseNeoDecodeBackend(Fa3DenseNeoCommon, Fa3DenseDecodeBackend):
    phase = AttentionPhase.DECODE
