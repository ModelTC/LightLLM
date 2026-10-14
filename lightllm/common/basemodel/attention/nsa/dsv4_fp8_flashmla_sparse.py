import dataclasses
import inspect
from typing import TYPE_CHECKING

import torch
import flash_mla as flashmla

from lightllm.common.kv_cache_mem_manager.deepseek4_mem_manager import (
    DSV4_C128_PAGE_SIZE,
    DSV4_C4_PAGE_SIZE,
    DSV4_MLA_BYTES_PER_TOKEN,
    DSV4_SWA_PAGE_SIZE,
)

from ..base_att import AttControl, BaseAttBackend, BaseDecodeAttState, BasePrefillAttState

if TYPE_CHECKING:
    from flash_mla.flash_mla_interface import FlashMLASchedMeta
    from lightllm.common.basemodel.infer_struct import InferStateInfo
    from lightllm.common.kv_cache_mem_manager.deepseek4_mem_manager import DeepseekV4MemoryManager

# Hopper sparse decode kernels support these local Q-head counts.
_SUPPORTED_Q_HEADS = (64, 128)
_COMPRESSED_PAGE_SIZES = {4: DSV4_C4_PAGE_SIZE, 128: DSV4_C128_PAGE_SIZE}


def get_dsv4_flashmla_padded_q_heads(q_head_num: int) -> int:
    for supported_head_num in _SUPPORTED_Q_HEADS:
        if q_head_num <= supported_head_num:
            return supported_head_num
    raise ValueError(f"FlashMLA does not support {q_head_num} local Q heads; supported counts: {_SUPPORTED_Q_HEADS}")


def _view_cache(buffer: torch.Tensor, page_size: int) -> torch.Tensor:
    """Expose FlashMLA's [page, token, KV head, byte] axes while retaining the padded page stride."""
    bytes_per_page = page_size * DSV4_MLA_BYTES_PER_TOKEN
    return buffer[:, :bytes_per_page].view(buffer.shape[0], page_size, 1, DSV4_MLA_BYTES_PER_TOKEN)


def _get_compressed_cache(mem_manager: "DeepseekV4MemoryManager", nsa_dict: dict) -> torch.Tensor | None:
    compress_ratio = nsa_dict["compress_ratio"]
    if compress_ratio == 0:
        return None

    page_size = _COMPRESSED_PAGE_SIZES.get(compress_ratio)
    if page_size is None:
        raise ValueError(f"unsupported DeepSeek-V4 compress ratio: {compress_ratio}")

    packed_cache = mem_manager.get_compressed_kv_buffer(nsa_dict["layer_index"])
    return _view_cache(packed_cache, page_size)


class DeepseekV4FlashMlaFp8SparseAttBackend(BaseAttBackend):
    def __init__(self, model):
        super().__init__(model=model)
        self.real_q_head_num = model.config["num_attention_heads"] // model.tp_world_size_
        self.padded_q_head_num = get_dsv4_flashmla_padded_q_heads(self.real_q_head_num)
        self.compress_ratios = tuple(dict.fromkeys(model.config["compress_ratios"]))
        # Use the package's optional out interface; older upstream builds use a copy.
        self.supports_out = "out" in inspect.signature(flashmla.flash_mla_with_kvcache).parameters

    def _flashmla_att(
        self,
        q: torch.Tensor,
        packed_kv: torch.Tensor,
        mem_manager: "DeepseekV4MemoryManager",
        nsa_dict: dict,
        sched_meta: "FlashMLASchedMeta",
        out: torch.Tensor = None,
    ) -> torch.Tensor:
        """Attend to SWA plus optional compressed KV, then trim padded Q heads from the result."""
        compressed_cache = _get_compressed_cache(mem_manager, nsa_dict)
        swa_cache = _view_cache(packed_kv, DSV4_SWA_PAGE_SIZE)
        output_kwargs = {}
        if self.supports_out and out is not None:
            output_kwargs["out"] = out

        padded_out, _ = flashmla.flash_mla_with_kvcache(
            q=q.unsqueeze(1),
            k_cache=swa_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=nsa_dict["head_dim_v"],
            tile_scheduler_metadata=sched_meta,
            num_splits=None,
            softmax_scale=nsa_dict["softmax_scale"],
            causal=False,
            is_fp8_kvcache=True,
            indices=nsa_dict["swa_indices"],
            attn_sink=nsa_dict["attn_sink"],
            topk_length=nsa_dict["swa_lengths"],
            extra_k_cache=compressed_cache,
            extra_indices_in_kvcache=nsa_dict.get("extra_indices"),
            extra_topk_length=nsa_dict.get("extra_lengths"),
            **output_kwargs,
        )
        return padded_out[:, 0, : self.real_q_head_num, :]

    def create_att_prefill_state(self, infer_state: "InferStateInfo") -> "_PrefillAttState":
        return _PrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state: "InferStateInfo") -> "_DecodeAttState":
        return _DecodeAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class _PrefillAttState(BasePrefillAttState):
    sched_meta_by_ratio: dict[int, "FlashMLASchedMeta"] | None = None

    def init_state(self):
        self.sched_meta_by_ratio = {}

    def _get_sched_meta(self, compress_ratio: int) -> "FlashMLASchedMeta":
        if compress_ratio not in self.sched_meta_by_ratio:
            self.sched_meta_by_ratio[compress_ratio] = flashmla.get_mla_metadata()[0]
        return self.sched_meta_by_ratio[compress_ratio]

    def prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
        *,
        out: torch.Tensor = None,
    ) -> torch.Tensor:
        assert att_control.nsa_prefill, "nsa_prefill must be True for NSA prefill attention"
        assert att_control.nsa_prefill_dict is not None, "nsa_prefill_dict is required"
        nsa_dict = att_control.nsa_prefill_dict
        needs_head_padding = self.backend.real_q_head_num != self.backend.padded_q_head_num
        can_write_out = out is not None and self.backend.supports_out and not needs_head_padding

        # FlashMLA writes all padded heads, so only an unpadded output can be passed directly.
        flashmla_out = out.unsqueeze(1) if can_write_out else None
        att_out = self.backend._flashmla_att(
            q,
            k,
            self.infer_state.mem_manager,
            nsa_dict,
            self._get_sched_meta(nsa_dict["compress_ratio"]),
            out=flashmla_out,
        )

        if can_write_out:
            return out
        if out is None and not needs_head_padding:
            return att_out

        # Copy when heads were trimmed or the package cannot write the caller's buffer.
        if out is None:
            out = alloc_func(att_out.shape, dtype=att_out.dtype, device=att_out.device)
        out.copy_(att_out)
        return out


@dataclasses.dataclass
class _DecodeAttState(BaseDecodeAttState):
    sched_meta_by_ratio: dict[int, "FlashMLASchedMeta"] | None = None

    def init_state(self):
        self.reset_sched_meta_for_capture()

    def reset_sched_meta_for_capture(self):
        # Each ratio has different cache geometry. Reset before capture so scheduling runs on every replay.
        self.sched_meta_by_ratio = {ratio: flashmla.get_mla_metadata()[0] for ratio in self.backend.compress_ratios}

    def decode_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        assert att_control.nsa_decode, "nsa_decode must be True for NSA decode attention"
        assert att_control.nsa_decode_dict is not None, "nsa_decode_dict is required"
        nsa_dict = att_control.nsa_decode_dict
        real_out = self.backend._flashmla_att(
            q,
            k,
            self.infer_state.mem_manager,
            nsa_dict,
            self.sched_meta_by_ratio[nsa_dict["compress_ratio"]],
        )
        return real_out.contiguous()


DSV4_NSA_BACKENDS = {"fp8kv_dsa": {"flashmla_sparse": DeepseekV4FlashMlaFp8SparseAttBackend}}
