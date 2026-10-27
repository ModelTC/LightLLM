import dataclasses
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import triton
import flash_mla as flashmla


from ..base_att import AttControl, BaseAttBackend, BaseDecodeAttState, BasePrefillAttState

if TYPE_CHECKING:
    from flash_mla.flash_mla_interface import FlashMLASchedMeta
    from lightllm.common.basemodel.infer_struct import InferStateInfo


class DeepseekV4FlashMlaFp8SparseAttBackend(BaseAttBackend):
    """FlashMLA layouts and operator workspace; forward states own sparse metadata."""

    def __init__(self, model):
        super().__init__(model=model)
        self.cache_config = model.req_manager.cache_config
        self.tp_q_head_num = model.config["num_attention_heads"] // model.tp_world_size_
        # Sparse FlashMLA uses 64-head tiles; TP can leave fewer local Q heads.
        self.padded_q_head_num = ((self.tp_q_head_num + 63) // 64) * 64
        self.compress_ratios = tuple(dict.fromkeys(model.config["compress_ratios"]))

    @classmethod
    def get_swa_prefill_workspace(cls, model):
        """由 backend 复用计算临时区；模型在 KV 容量评估前调用，以计入实际显存占用。"""
        if model.args.run_mode == "decode":
            return None, 0
        config = model.req_manager.cache_config
        page_token_num = config.swa_page_token_num
        max_forward_tokens = max(model.batch_max_tokens, model.max_req_num) + 2 * model.tp_world_size_
        capacity = (
            triton.cdiv(
                max_forward_tokens + model.max_req_num * (config.sliding_window + 2 * page_token_num), page_token_num
            )
            * page_token_num
        )
        regions = 1 + int(model.args.enable_prefill_microbatch_overlap)
        # 每个 microbatch 使用独立临时区；最后一页留给 HOLD 请求。
        token_num = regions * capacity + page_token_num
        buffer = cls.get_gpu_workspace_buffer(
            key_name="deepseek_v4_swa_prefill",
            workspace_size=config.layer_num * token_num * config.swa_slot_bytes,
            dtype=torch.uint8,
        )
        return buffer.view(config.layer_num, token_num, config.swa_slot_bytes), capacity

    def get_att_input_params(self, infer_state, layer_index):
        from lightllm.models.deepseek_v4.triton_kernel.cache_io import view_swa_kv

        swa, compressed_kv = infer_state.req_manager.get_att_input_params(layer_index)
        if infer_state.is_prefill:
            swa = view_swa_kv(infer_state.prefill_att_state.swa_buffer[layer_index], self.cache_config)
        return swa, compressed_kv

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
            page_size = config.compressed_kv_token_alignment // nsa_dict["compress_ratio"]
            bytes_per_page = page_size * bytes_per_token
            compressed_cache = compressed_packed[:, :bytes_per_page].view(
                compressed_packed.shape[0], page_size, 1, bytes_per_token
            )
        swa_bytes_per_page = config.swa_page_token_num * bytes_per_token
        swa_cache = swa_packed[:, :swa_bytes_per_page].view(
            swa_packed.shape[0], config.swa_page_token_num, 1, bytes_per_token
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
            out=out.unsqueeze(1),
        )
        return padded_out[:, 0]

    def create_att_prefill_state(self, infer_state: "InferStateInfo") -> "_PrefillAttState":
        return _PrefillAttState(backend=self)

    def create_att_decode_state(self, infer_state: "InferStateInfo") -> "_DecodeAttState":
        return _DecodeAttState(backend=self)


@dataclasses.dataclass
class _PrefillAttState(BasePrefillAttState):
    sched_meta_by_ratio: dict[int, "FlashMLASchedMeta"] | None = None

    def prepare_swa_workspace(self, infer_state):
        from lightllm.models.deepseek_v4.triton_kernel.swa_cache import prepare_swa_prefill_workspace

        requests = infer_state.req_manager
        self.swa_buffer, capacity = self.backend.get_swa_prefill_workspace(self.backend.model)
        self.HOLD_SWA_SLOT = self.swa_buffer.shape[1] - self.backend.cache_config.swa_page_token_num
        self.prefill_token_ranges = torch.zeros(
            (requests.max_request_num + 1, 2), dtype=torch.int32, device=self.swa_buffer.device
        )
        self.swa_prefill_workspace_offsets = torch.full(
            (requests.max_request_num + 1,), -1, dtype=torch.int32, device=self.swa_buffer.device
        )
        prepare_swa_prefill_workspace(
            requests,
            self,
            infer_state.b_req_idx,
            infer_state.b_ready_cache_len,
            infer_state.b_seq_len,
            infer_state.microbatch_index * capacity,
        )

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
