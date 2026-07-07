import dataclasses
import torch

from ..base_att import AttControl
from ..paged_fa3.fp import PagedFa3AttBackend, PagedFa3PrefillAttState, PagedFa3DecodeAttState
from lightllm.utils.fa4_utils import (
    ensure_fa4_available,
    ensure_fa4_supported_gpu,
    flash_attn_varlen_func,
    sm90_fa4_paged_kv_tile_n,
    unwrap_fa4_output,
)


class Fa4AttBackend(PagedFa3AttBackend):
    def __init__(self, model):
        ensure_fa4_available()
        ensure_fa4_supported_gpu()
        super().__init__(model=model)

    def create_att_prefill_state(self, infer_state) -> "Fa4PrefillAttState":
        return Fa4PrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state) -> "Fa4DecodeAttState":
        return Fa4DecodeAttState(backend=self, infer_state=infer_state)


def _sm90_fa4_paged_kv_tile_n(
    head_dim: int,
    head_dim_v: int,
    window_size: tuple[int, int],
) -> int | None:
    return sm90_fa4_paged_kv_tile_n(head_dim=head_dim, head_dim_v=head_dim_v, window_size=window_size)


def _ensure_fa4_paged_kv_supported(
    head_dim: int,
    head_dim_v: int,
    window_size: tuple[int, int],
    page_size: int,
) -> None:
    tile_n = _sm90_fa4_paged_kv_tile_n(head_dim, head_dim_v, window_size)
    if tile_n is None or page_size == tile_n or tile_n >= 128:
        return

    raise RuntimeError(
        "FA4 SM90 paged KV requires page_size == tile_n for this shape; "
        f"current page_size={page_size}, required_page_size={tile_n}, "
        f"head_dim={head_dim}, head_dim_v={head_dim_v}, window_size={window_size}. "
        "LightLLM's current FA4 wrapper uses token-granular KV pages, so this shape would need "
        "the removed repack fallback to run. Please set the FA4 KV cache page size to "
        f"{tile_n} tokens for this model/shape, or switch --llm_prefill_att_backend/"
        "--llm_decode_att_backend to another backend."
    )


@dataclasses.dataclass
class Fa4PrefillAttState(PagedFa3PrefillAttState):
    def _normal_prefill_att(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, att_control: AttControl, alloc_func=torch.empty
    ) -> torch.Tensor:
        if att_control.use_sliding_window:
            window_size = att_control.sliding_window
        else:
            window_size = (-1, -1)

        if att_control.use_att_sink:
            sink_weight = att_control.sink_weight
        else:
            sink_weight = None

        head_dim = q.shape[-1]
        head_dim_v = v.shape[-1]
        softmax_scale = 1.0 / (head_dim ** 0.5)
        _ensure_fa4_paged_kv_supported(head_dim, head_dim_v, window_size, page_size=self.backend.page_size)

        out = flash_attn_varlen_func(
            q=q,
            k=k.view(-1, self.backend.page_size, k.shape[1], k.shape[2]),
            v=v.view(-1, self.backend.page_size, v.shape[1], v.shape[2]),
            cu_seqlens_q=self.cu_seqlens_q,
            seqused_k=self.infer_state.b_seq_len.int(),
            max_seqlen_q=self.infer_state.max_q_seq_len,
            max_seqlen_k=self.infer_state.max_kv_seq_len,
            page_table=self.page_table,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=window_size,
            learnable_sink=sink_weight,
            softcap=0.0,
            return_lse=False,
        )
        return unwrap_fa4_output(out)


@dataclasses.dataclass
class Fa4DecodeAttState(PagedFa3DecodeAttState):
    def _normal_decode_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl,
        alloc_func=torch.empty,
    ):
        if att_control.use_sliding_window:
            window_size = att_control.sliding_window
        else:
            window_size = (-1, -1)

        if att_control.use_att_sink:
            sink_weight = att_control.sink_weight
        else:
            sink_weight = None

        head_dim = q.shape[-1]
        head_dim_v = v.shape[-1]
        softmax_scale = 1.0 / (head_dim ** 0.5)
        _ensure_fa4_paged_kv_supported(head_dim, head_dim_v, window_size, page_size=self.backend.page_size)

        out = flash_attn_varlen_func(
            q=q,
            k=k.view(-1, self.backend.page_size, k.shape[1], k.shape[2]),
            v=v.view(-1, self.backend.page_size, v.shape[1], v.shape[2]),
            cu_seqlens_q=self.cu_seqlens_q,
            seqused_k=self.b_att_seq_len.int(),
            max_seqlen_q=self.decode_max_q_seq_len,
            max_seqlen_k=self.infer_state.max_kv_seq_len,
            page_table=self.page_table,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=window_size,
            learnable_sink=sink_weight,
            softcap=0.0,
            return_lse=False,
        )
        return unwrap_fa4_output(out)
