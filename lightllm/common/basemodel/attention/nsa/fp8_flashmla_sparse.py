import dataclasses
import torch
from typing import TYPE_CHECKING, Tuple

from ..base_att import AttControl, BaseAttBackend, BaseDecodeAttState, BasePrefillAttState
from lightllm.utils.dist_utils import get_current_device_id

if TYPE_CHECKING:
    from lightllm.common.basemodel.infer_struct import InferStateInfo


# this flash_mla extra-cache fork only instantiates h_q in {64, 128}; pad TP-split q heads up
# to the nearest supported count (zero heads are discarded from the output slice).
FLASHMLA_SUPPORTED_HEADS = (64, 128)


def _pad_q_heads(q_4d: torch.Tensor, attn_sink: torch.Tensor):
    h_q = q_4d.shape[2]
    if h_q in FLASHMLA_SUPPORTED_HEADS:
        return q_4d, attn_sink, h_q
    target = next((h for h in FLASHMLA_SUPPORTED_HEADS if h >= h_q), None)
    assert target is not None, f"num q heads {h_q} exceeds flash_mla support {FLASHMLA_SUPPORTED_HEADS}"
    q_pad = torch.nn.functional.pad(q_4d, (0, 0, 0, target - h_q))
    sink_pad = torch.nn.functional.pad(attn_sink, (0, target - h_q))
    return q_pad, sink_pad, h_q


class DeepseekV4MissingOperatorError(RuntimeError):
    pass


def _missing_attention_op(feature: str) -> None:
    raise DeepseekV4MissingOperatorError(
        f"DeepSeek-V4 {feature} has no production batch operator. The flashmla_kvcache path "
        f"(packed swa/c4/c128 pools + paged compressor + indexer top-k) is the supported route; "
        f"this legacy/non-flashmla entry point was never wired and is fenced on purpose."
    )


def _view_dsv4_flashmla_cache(layer_buffer: torch.Tensor, page_size: int) -> torch.Tensor:
    from lightllm.common.kv_cache_mem_manager.deepseek4_mem_manager import DSV4_MLA_BYTES_PER_TOKEN

    usable = page_size * DSV4_MLA_BYTES_PER_TOKEN
    return layer_buffer[:, :usable].view(layer_buffer.shape[0], page_size, 1, DSV4_MLA_BYTES_PER_TOKEN)


@dataclasses.dataclass
class _Dsv4Metadata:
    swa_indices: torch.Tensor
    swa_lengths: torch.Tensor
    extra_cache: torch.Tensor = None
    extra_indices: torch.Tensor = None
    extra_lengths: torch.Tensor = None


def _metadata_from_dict(infer_state, nsa_dict: dict) -> "_Dsv4Metadata":
    """Bundle the model-built FINAL index tensors (carried in nsa_dict by DeepseekV4IndexInfer) with
    the layer-keyed fp8 extra-cache byte view. The cache view is data-independent (a fixed per-layer
    buffer slice), so it is built here -- a genuine flash_mla ABI concern -- rather than on the model
    side; only the index/length tensors cross the att_control boundary."""
    from lightllm.common.kv_cache_mem_manager.deepseek4_mem_manager import DSV4_C128_PAGE_SIZE, DSV4_C4_PAGE_SIZE

    ratio = nsa_dict["compress_ratio"]
    extra_cache = None
    if ratio:
        page = DSV4_C4_PAGE_SIZE if ratio == 4 else DSV4_C128_PAGE_SIZE
        extra_buffer = infer_state.mem_manager.get_compressed_kv_buffer(nsa_dict["layer_index"])
        extra_cache = _view_dsv4_flashmla_cache(extra_buffer, page)
    return _Dsv4Metadata(
        swa_indices=nsa_dict["swa_indices"],
        swa_lengths=nsa_dict["swa_lengths"],
        extra_cache=extra_cache,
        extra_indices=nsa_dict.get("extra_indices"),
        extra_lengths=nsa_dict.get("extra_lengths"),
    )


class NsaFlashMlaFp8SparseAttBackend(BaseAttBackend):
    def __init__(self, model):
        super().__init__(model=model)
        device = get_current_device_id()
        self.ragged_mem_buffers = [
            torch.empty(model.graph_max_batch_size * model.max_seq_length, dtype=torch.int32, device=device)
            for _ in range(2)
        ]

    def create_att_prefill_state(self, infer_state: "InferStateInfo") -> "NsaFlashMlaFp8SparsePrefillAttState":
        return NsaFlashMlaFp8SparsePrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state: "InferStateInfo") -> "NsaFlashMlaFp8SparseDecodeAttState":
        return NsaFlashMlaFp8SparseDecodeAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class NsaFlashMlaFp8SparsePrefillAttState(BasePrefillAttState):
    ks: torch.Tensor = None
    ke: torch.Tensor = None
    lengths: torch.Tensor = None
    ragged_mem_index: torch.Tensor = None

    def init_state(self):
        self.backend: NsaFlashMlaFp8SparseAttBackend = self.backend
        self.ragged_mem_index = torch.empty(
            self.infer_state.total_token_num,
            dtype=torch.int32,
            device=get_current_device_id(),
        )
        from lightllm.common.basemodel.triton_kernel.gen_nsa_ks_ke import gen_nsa_ks_ke

        self.ks, self.ke, self.lengths = gen_nsa_ks_ke(
            b_seq_len=self.infer_state.b_seq_len,
            b_q_seq_len=self.infer_state.b_q_seq_len,
            b_req_idx=self.infer_state.b_req_idx,
            req_to_token_index=self.infer_state.req_manager.req_to_token_indexs,
            q_token_num=self.infer_state.total_token_num - self.infer_state.prefix_total_token_num,
            ragged_mem_index=self.ragged_mem_index,
            hold_req_idx=self.infer_state.req_manager.HOLD_REQUEST_ID,
        )
        return

    def prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        assert att_control.nsa_prefill, "nsa_prefill must be True for NSA prefill attention"
        assert att_control.nsa_prefill_dict is not None, "nsa_prefill_dict is required"
        if att_control.nsa_prefill_dict.get("flashmla_kvcache"):
            return self._flashmla_kvcache_prefill_att(
                q=q,
                packed_kv=k,
                nsa_dict=att_control.nsa_prefill_dict,
            )
        return self._nsa_prefill_att(q=q, packed_kv=k, att_control=att_control)

    def _nsa_prefill_att(
        self,
        q: torch.Tensor,
        packed_kv: torch.Tensor,
        att_control: AttControl,
    ) -> torch.Tensor:
        import flash_mla

        nsa_dict = att_control.nsa_prefill_dict
        topk_indices = nsa_dict["topk_indices"]
        softmax_scale = nsa_dict["softmax_scale"]
        kv_lora_rank = nsa_dict["kv_lora_rank"]
        topk_mem_indices = nsa_dict["topk_mem_indices"]
        prefill_cache_kv = nsa_dict["prefill_cache_kv"]
        attn_sink = nsa_dict.get("attn_sink")
        topk_length = nsa_dict.get("topk_length")

        if self.infer_state.prefix_total_token_num > 0:
            # 当前推理生成的token kv部分从 prefill_cache_kv 中获取，历史
            # 部分kv 从 packed_kv 中获取, 并进行反量化，这样可以避免 prefill_cache_kv
            # 部分的数据进行重复的反量化操作，提升整体的性能。
            kv, topk_indices = self.infer_state.mem_manager.get_prefill_kv_cache_and_remap_indices(
                packed_kv=packed_kv,
                topk_indices=topk_mem_indices,
                prefill_mem_index=self.infer_state.mem_index,
                prefill_cache_kv=prefill_cache_kv,
            )
        else:
            kv = prefill_cache_kv

        if topk_indices.ndim == 2:
            topk_indices = topk_indices.unsqueeze(1)

        mla_out, _, _ = flash_mla.flash_mla_sparse_fwd(
            q=q,
            kv=kv,
            indices=topk_indices,
            sm_scale=softmax_scale,
            d_v=kv_lora_rank,
            attn_sink=attn_sink,
            topk_length=topk_length,
        )
        return mla_out

    def _flashmla_kvcache_prefill_att(self, q: torch.Tensor, packed_kv: torch.Tensor, nsa_dict: dict) -> torch.Tensor:
        attn_sink = nsa_dict["attn_sink"].to(torch.float32).contiguous()
        metadata = _metadata_from_dict(self.infer_state, nsa_dict)
        return self._flashmla_kvcache_att(q, packed_kv, metadata, attn_sink, nsa_dict)

    def _flashmla_kvcache_att(
        self,
        q: torch.Tensor,
        packed_kv: torch.Tensor,
        metadata: _Dsv4Metadata,
        attn_sink: torch.Tensor,
        nsa_dict: dict,
    ) -> torch.Tensor:
        import flash_mla
        from lightllm.common.kv_cache_mem_manager.deepseek4_mem_manager import DSV4_SWA_PAGE_SIZE

        q_4d = q.unsqueeze(1).contiguous()
        q_4d, attn_sink, num_real_heads = _pad_q_heads(q_4d, attn_sink)
        k_cache = _view_dsv4_flashmla_cache(packed_kv, DSV4_SWA_PAGE_SIZE)
        sched_meta, _ = flash_mla.get_mla_metadata()
        out, _ = flash_mla.flash_mla_with_kvcache(
            q=q_4d,
            k_cache=k_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=nsa_dict["head_dim_v"],
            tile_scheduler_metadata=sched_meta,
            num_splits=None,
            softmax_scale=nsa_dict["softmax_scale"],
            causal=False,
            is_fp8_kvcache=True,
            indices=metadata.swa_indices,
            attn_sink=attn_sink,
            topk_length=metadata.swa_lengths,
            extra_k_cache=metadata.extra_cache,
            extra_indices_in_kvcache=metadata.extra_indices,
            extra_topk_length=metadata.extra_lengths,
        )
        return out[:, 0, :num_real_heads].contiguous()


@dataclasses.dataclass
class NsaFlashMlaFp8SparseDecodeAttState(BaseDecodeAttState):
    ks: torch.Tensor = None
    ke: torch.Tensor = None
    lengths: torch.Tensor = None
    ragged_mem_index: torch.Tensor = None
    flashmla_sched_meta: object = None

    def init_state(self):
        self.backend: NsaFlashMlaFp8SparseAttBackend = self.backend
        model = self.backend.model
        use_cuda_graph = (
            self.infer_state.batch_size <= model.graph_max_batch_size
            and self.infer_state.max_kv_seq_len <= model.graph_max_len_in_batch
        )

        if use_cuda_graph:
            self.ragged_mem_index = self.backend.ragged_mem_buffers[self.infer_state.microbatch_index]
        else:
            self.ragged_mem_index = torch.empty(
                self.infer_state.total_token_num,
                dtype=torch.int32,
                device=get_current_device_id(),
            )

        from lightllm.common.basemodel.triton_kernel.gen_nsa_ks_ke import gen_nsa_ks_ke

        self.ks, self.ke, self.lengths = gen_nsa_ks_ke(
            b_seq_len=self.infer_state.b_seq_len,
            b_q_seq_len=self.infer_state.b_q_seq_len,
            b_req_idx=self.infer_state.b_req_idx,
            req_to_token_index=self.infer_state.req_manager.req_to_token_indexs,
            q_token_num=self.infer_state.b_seq_len.shape[0],
            ragged_mem_index=self.ragged_mem_index,
            hold_req_idx=self.infer_state.req_manager.HOLD_REQUEST_ID,
        )
        import flash_mla

        # one sched_meta per layer type: the lazy config locks extra-cache geometry (page size,
        # presence) on first invocation, so swa-only/c4/c128 layers must not share one object.
        self.flashmla_sched_meta = {ratio: flash_mla.get_mla_metadata()[0] for ratio in (0, 4, 128)}
        return

    def reset_sched_meta_for_capture(self):
        # cuda-graph capture hook: the warmup pass already locked/stored sched meta on this
        # (shared) state object; reset so the capture pass re-plans INSIDE the graph and every
        # replay re-plans from the live tensors instead of binding warmup leftovers.
        import flash_mla

        self.flashmla_sched_meta = {ratio: flash_mla.get_mla_metadata()[0] for ratio in (0, 4, 128)}
        return

    def decode_att(
        self,
        q: Tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        assert att_control.nsa_decode, "nsa_decode must be True for NSA decode attention"
        assert att_control.nsa_decode_dict is not None, "nsa_decode_dict is required"
        if att_control.nsa_decode_dict.get("flashmla_kvcache"):
            return self._flashmla_kvcache_decode_att(
                q=q,
                packed_kv=k,
                nsa_dict=att_control.nsa_decode_dict,
            )
        return self._nsa_decode_att(q=q, packed_kv=k, att_control=att_control)

    def _nsa_decode_att(
        self,
        q: Tuple[torch.Tensor, torch.Tensor],
        packed_kv: torch.Tensor,
        att_control: AttControl,
    ) -> torch.Tensor:
        import flash_mla

        nsa_dict = att_control.nsa_decode_dict
        topk_mem_indices = nsa_dict["topk_mem_indices"]
        softmax_scale = nsa_dict["softmax_scale"]
        kv_lora_rank = nsa_dict["kv_lora_rank"]
        attn_sink = nsa_dict.get("attn_sink")
        topk_length = nsa_dict.get("topk_length")
        extra_k_cache = nsa_dict.get("extra_k_cache")
        extra_indices = nsa_dict.get("extra_indices_in_kvcache")
        extra_topk_length = nsa_dict.get("extra_topk_length")

        if topk_mem_indices.ndim == 2:
            topk_mem_indices = topk_mem_indices.unsqueeze(1)
        assert topk_mem_indices.shape[1] == 1, "FlashMLA sparse decode path currently expects seq_len_q == 1"

        q_nope, q_rope = q
        q_all = torch.cat([q_nope, q_rope], dim=-1).unsqueeze(1).contiguous()
        kv = torch.as_strided(
            packed_kv,
            size=(packed_kv.shape[0], 1, 1, packed_kv.shape[-1]),
            stride=(packed_kv.stride(0), packed_kv.shape[-1], packed_kv.shape[-1], packed_kv.stride(-1)),
        )

        o_tensor, _ = flash_mla.flash_mla_with_kvcache(
            q=q_all,
            k_cache=kv,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=kv_lora_rank,
            tile_scheduler_metadata=self.flashmla_sched_meta[0],
            softmax_scale=softmax_scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=topk_mem_indices,
            attn_sink=attn_sink,
            topk_length=topk_length,
            extra_k_cache=extra_k_cache,
            extra_indices_in_kvcache=extra_indices,
            extra_topk_length=extra_topk_length,
        )
        return o_tensor[:, 0, :, :]  # [b, 1, h, d] -> [b, h, d]

    def _flashmla_kvcache_decode_att(self, q: torch.Tensor, packed_kv: torch.Tensor, nsa_dict: dict) -> torch.Tensor:
        attn_sink = nsa_dict["attn_sink"].to(torch.float32).contiguous()
        metadata = _metadata_from_dict(self.infer_state, nsa_dict)
        return self._flashmla_kvcache_att(q, packed_kv, metadata, attn_sink, nsa_dict)

    def _flashmla_kvcache_att(
        self,
        q: torch.Tensor,
        packed_kv: torch.Tensor,
        metadata: _Dsv4Metadata,
        attn_sink: torch.Tensor,
        nsa_dict: dict,
    ) -> torch.Tensor:
        import flash_mla
        from lightllm.common.kv_cache_mem_manager.deepseek4_mem_manager import DSV4_SWA_PAGE_SIZE

        q_4d = q.unsqueeze(1).contiguous()
        q_4d, attn_sink, num_real_heads = _pad_q_heads(q_4d, attn_sink)
        k_cache = _view_dsv4_flashmla_cache(packed_kv, DSV4_SWA_PAGE_SIZE)
        out, _ = flash_mla.flash_mla_with_kvcache(
            q=q_4d,
            k_cache=k_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=nsa_dict["head_dim_v"],
            tile_scheduler_metadata=self.flashmla_sched_meta[nsa_dict["compress_ratio"]],
            num_splits=None,
            softmax_scale=nsa_dict["softmax_scale"],
            causal=False,
            is_fp8_kvcache=True,
            indices=metadata.swa_indices,
            attn_sink=attn_sink,
            topk_length=metadata.swa_lengths,
            extra_k_cache=metadata.extra_cache,
            extra_indices_in_kvcache=metadata.extra_indices,
            extra_topk_length=metadata.extra_lengths,
        )
        return out[:, 0, :num_real_heads].contiguous()
