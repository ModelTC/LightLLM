"""Generation-side adapter from dense prefix KV to attention calls."""

from __future__ import annotations

import math

import torch

from lightllm.server.generationserver.kv_materializer import DensePrefixKV


def _paged_prefix_attention(*args, **kwargs):
    from lightllm.common.basemodel.attention.paged_prefix import (
        paged_prefix_attention,
    )

    return paged_prefix_attention(*args, **kwargs)


class DensePrefixAttentionProvider:
    """Attention interface expected by the NeoPP LightX2V runner."""

    def __init__(self, prefix: DensePrefixKV) -> None:
        if prefix.batch_size != 1:
            raise ValueError("an X2V Prefix provider requires exactly one prefix")
        self._prefix = prefix
        self.seq_len = prefix.seq_len
        self._cu_seqlens_q: dict[tuple[torch.device, int], torch.Tensor] = {}

    def attend(self, *, layer_index: int, q: torch.Tensor, softmax_scale: float):
        expected_scale = q.shape[-1] ** -0.5
        if not math.isclose(
            softmax_scale,
            expected_scale,
            rel_tol=1e-6,
            abs_tol=0.0,
        ):
            raise ValueError("consumer and LightLLM Prefix attention scales differ")
        cache_key = (q.device, q.shape[0])
        cu_seqlens_q = self._cu_seqlens_q.get(cache_key)
        if cu_seqlens_q is None:
            cu_seqlens_q = torch.tensor(
                [0, q.shape[0]],
                dtype=torch.int32,
                device=q.device,
            )
            self._cu_seqlens_q[cache_key] = cu_seqlens_q
        layer = self._prefix.layer(layer_index)

        return _paged_prefix_attention(
            q,
            layer.key,
            layer.value,
            self._prefix.page_table,
            self._prefix.prefix_lengths,
            cu_seqlens_q,
            max_q_len=q.shape[0],
        )


__all__ = ["DensePrefixAttentionProvider"]
