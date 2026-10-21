import dataclasses
import inspect
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import flash_mla as flashmla

from ..base_att import AttControl, BaseAttBackend, BaseDecodeAttState, BasePrefillAttState

if TYPE_CHECKING:
    from flash_mla.flash_mla_interface import FlashMLASchedMeta
    from lightllm.common.basemodel.infer_struct import InferStateInfo


class DeepseekV4FlashMlaFp8SparseAttBackend(BaseAttBackend):
    def __init__(self, model):
        super().__init__(model=model)
        self.cache_config = model.req_manager.cache_config
        self.tp_q_head_num = model.config["num_attention_heads"] // model.tp_world_size_
        # Sparse FlashMLA uses 64-head tiles; TP can leave fewer local Q heads.
        self.padded_q_head_num = ((self.tp_q_head_num + 63) // 64) * 64
        self.compress_ratios = tuple(dict.fromkeys(model.config["compress_ratios"]))
        self.flashmla_supports_out = "out" in inspect.signature(flashmla.flash_mla_with_kvcache).parameters

    def _flashmla_att(
        self,
        q: torch.Tensor,
        packed_kv: Tuple[torch.Tensor, Optional[torch.Tensor]],
        nsa_dict: dict,
        sched_meta: "FlashMLASchedMeta",
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Attend to SWA plus optional compressed KV and return all padded Q heads."""
        swa_packed, compressed_packed = packed_kv
        config = self.cache_config
        bytes_per_token = config.kv_bytes
        # Expose FlashMLA's [page, token, KV head, byte] axes while retaining padded page strides.
        compressed_cache = None
        if compressed_packed is not None:
            page_size = config.history_token_alignment // nsa_dict["compress_ratio"]
            bytes_per_page = page_size * bytes_per_token
            compressed_cache = compressed_packed[:, :bytes_per_page].view(
                compressed_packed.shape[0], page_size, 1, bytes_per_token
            )
        swa_bytes_per_page = config.swa_token_alignment * bytes_per_token
        swa_cache = swa_packed[:, :swa_bytes_per_page].view(
            swa_packed.shape[0], config.swa_token_alignment, 1, bytes_per_token
        )

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
            **({"out": out.unsqueeze(1)} if self.flashmla_supports_out else {}),
        )
        if not self.flashmla_supports_out:
            out.copy_(padded_out[:, 0])
            return out
        return padded_out[:, 0]

    def create_att_prefill_state(self, infer_state: "InferStateInfo") -> "_PrefillAttState":
        return _PrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state: "InferStateInfo") -> "_DecodeAttState":
        return _DecodeAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class _PrefillAttState(BasePrefillAttState):
    sched_meta_by_ratio: dict[int, "FlashMLASchedMeta"] | None = None

    def init_state(self):
        self.sched_meta_by_ratio = {ratio: flashmla.get_mla_metadata()[0] for ratio in self.backend.compress_ratios}

    def prefill_att(
        self,
        q: torch.Tensor,
        k: Tuple[torch.Tensor, Optional[torch.Tensor]],
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
        *,
        out: torch.Tensor = None,
    ) -> torch.Tensor:
        nsa_dict = att_control.nsa_prefill_dict
        if out is None:
            out = alloc_func((q.shape[0], q.shape[1], nsa_dict["head_dim_v"]), dtype=q.dtype, device=q.device)

        padded_out = self.backend._flashmla_att(
            q,
            k,
            nsa_dict,
            self.sched_meta_by_ratio[nsa_dict["compress_ratio"]],
            out=out,
        )
        return padded_out[:, : self.backend.tp_q_head_num]


@dataclasses.dataclass
class _DecodeAttState(BaseDecodeAttState):
    sched_meta_by_ratio: dict[int, "FlashMLASchedMeta"] | None = None
    capture_sched_meta_by_ratio: dict[int, "FlashMLASchedMeta"] | None = None

    def init_state(self):
        self.sched_meta_by_ratio = {ratio: flashmla.get_mla_metadata()[0] for ratio in self.backend.compress_ratios}
        # Keep capture metadata untouched by warmup so its GPU planning kernels enter the graph.
        self.capture_sched_meta_by_ratio = {
            ratio: flashmla.get_mla_metadata()[0] for ratio in self.backend.compress_ratios
        }

    def decode_att(
        self,
        q: torch.Tensor,
        k: Tuple[torch.Tensor, Optional[torch.Tensor]],
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        nsa_dict = att_control.nsa_decode_dict
        sched_meta_by_ratio = (
            self.capture_sched_meta_by_ratio if torch.cuda.is_current_stream_capturing() else self.sched_meta_by_ratio
        )
        out = alloc_func((q.shape[0], q.shape[1], nsa_dict["head_dim_v"]), dtype=q.dtype, device=q.device)
        padded_out = self.backend._flashmla_att(
            q,
            k,
            nsa_dict,
            sched_meta_by_ratio[nsa_dict["compress_ratio"]],
            out=out,
        )
        return padded_out[:, : self.backend.tp_q_head_num]


DSV4_NSA_BACKENDS = {"fp8kv_dsa": {"flashmla_sparse": DeepseekV4FlashMlaFp8SparseAttBackend}}
