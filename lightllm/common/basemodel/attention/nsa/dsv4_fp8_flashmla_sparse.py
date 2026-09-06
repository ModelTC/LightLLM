import dataclasses
from typing import TYPE_CHECKING

import torch
from vllm.v1.attention.ops import flashmla

from ..base_att import AttControl, BaseAttBackend, BaseDecodeAttState, BasePrefillAttState

if TYPE_CHECKING:
    from lightllm.common.basemodel.infer_struct import InferStateInfo


# The current FlashMLA MODEL1 binary only instantiates these Q-head counts.
_SUPPORTED_Q_HEADS = (64, 128)


def get_dsv4_flashmla_padded_q_heads(q_head_num: int) -> int:
    for supported_head_num in _SUPPORTED_Q_HEADS:
        if q_head_num <= supported_head_num:
            return supported_head_num
    raise ValueError(f"FlashMLA does not support {q_head_num} local Q heads; supported counts: {_SUPPORTED_Q_HEADS}")


def _view_cache(buffer: torch.Tensor, page_size: int) -> torch.Tensor:
    from lightllm.common.kv_cache_mem_manager.deepseek4_mem_manager import DSV4_MLA_BYTES_PER_TOKEN

    byte_num = page_size * DSV4_MLA_BYTES_PER_TOKEN
    return buffer[:, :byte_num].view(buffer.shape[0], page_size, 1, DSV4_MLA_BYTES_PER_TOKEN)


class DeepseekV4FlashMlaFp8SparseAttBackend(BaseAttBackend):
    def __init__(self, model):
        super().__init__(model=model)
        self.real_q_head_num = model.config["num_attention_heads"] // model.tp_world_size_
        self.padded_q_head_num = get_dsv4_flashmla_padded_q_heads(self.real_q_head_num)
        self.compress_ratios = tuple(dict.fromkeys(model.config["compress_ratios"]))

    def _flashmla_att(
        self,
        q: torch.Tensor,
        packed_kv: torch.Tensor,
        mem_manager,
        nsa_dict: dict,
        sched_meta,
        flashmla_out: torch.Tensor = None,
    ) -> torch.Tensor:
        from lightllm.common.kv_cache_mem_manager.deepseek4_mem_manager import (
            DSV4_C128_PAGE_SIZE,
            DSV4_C4_PAGE_SIZE,
            DSV4_SWA_PAGE_SIZE,
        )

        ratio = nsa_dict["compress_ratio"]
        extra_cache = None
        if ratio == 4:
            extra_page_size = DSV4_C4_PAGE_SIZE
        elif ratio == 128:
            extra_page_size = DSV4_C128_PAGE_SIZE
        elif ratio != 0:
            raise ValueError(f"unsupported DeepSeek-V4 compress ratio: {ratio}")
        if ratio:
            buffer = mem_manager.get_compressed_kv_buffer(nsa_dict["layer_index"])
            extra_cache = _view_cache(buffer, extra_page_size)

        kwargs = dict(
            q=q.unsqueeze(1),
            k_cache=_view_cache(packed_kv, DSV4_SWA_PAGE_SIZE),
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
            extra_k_cache=extra_cache,
            extra_indices_in_kvcache=nsa_dict.get("extra_indices"),
            extra_topk_length=nsa_dict.get("extra_lengths"),
        )
        if flashmla_out is not None:
            kwargs["out"] = flashmla_out
        full_out, _ = flashmla.flash_mla_with_kvcache(**kwargs)
        return full_out[:, 0, : self.real_q_head_num, :]

    def create_att_prefill_state(self, infer_state: "InferStateInfo") -> "_PrefillAttState":
        return _PrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state: "InferStateInfo") -> "_DecodeAttState":
        return _DecodeAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class _PrefillAttState(BasePrefillAttState):
    flashmla_sched_meta: dict = None

    def init_state(self):
        self.flashmla_sched_meta = {}

    def _get_sched_meta(self, compress_ratio: int):
        if compress_ratio not in self.flashmla_sched_meta:
            self.flashmla_sched_meta[compress_ratio] = flashmla.get_mla_metadata()[0]
        return self.flashmla_sched_meta[compress_ratio]

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
        if out is None:
            out = alloc_func(
                (q.shape[0], self.backend.real_q_head_num, nsa_dict["head_dim_v"]),
                dtype=q.dtype,
                device=q.device,
            )
        full_out = self.infer_state.dsv4_workspace.flashmla_prefill_full_out[: q.shape[0]]
        out.copy_(
            self.backend._flashmla_att(
                q,
                k,
                self.infer_state.mem_manager,
                nsa_dict,
                self._get_sched_meta(nsa_dict["compress_ratio"]),
                flashmla_out=full_out,
            )
        )
        return out


@dataclasses.dataclass
class _DecodeAttState(BaseDecodeAttState):
    flashmla_sched_meta: dict = None

    def init_state(self):
        self.reset_sched_meta_for_capture()

    def reset_sched_meta_for_capture(self):
        # FlashMLA lazily binds extra-cache geometry, so ratios cannot share one sched-meta object.
        self.flashmla_sched_meta = {ratio: flashmla.get_mla_metadata()[0] for ratio in self.backend.compress_ratios}

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
            self.flashmla_sched_meta[nsa_dict["compress_ratio"]],
        )
        return real_out.contiguous()


DSV4_NSA_BACKENDS = {"fp8kv_dsa": {"flashmla_sparse": DeepseekV4FlashMlaFp8SparseAttBackend}}
