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
        from lightllm.models.deepseek_v4.config import create_attention_config

        config = create_attention_config(
            model_config=model.config,
            model_logical_layer_ids=tuple(layer.layer_num_ for layer in model.layers_infer),
            tp_rank=model.tp_rank_,
            tp_world_size=model.tp_world_size_,
        )
        self.tp_q_head_num = model.config["num_attention_heads"] // model.tp_world_size_
        # Sparse FlashMLA uses 64-head tiles; TP can leave fewer local Q heads.
        self.padded_q_head_num = ((self.tp_q_head_num + 63) // 64) * 64
        self.compress_ratios = tuple(dict.fromkeys(config.compress_ratios))
        self.layer_compress_ratios = dict(zip(config.logical_layer_ids, config.compress_ratios))
        self.swa_config = config.create_swa_config()
        self.compressed_configs = {
            ratio: config.create_compressed_config(compress_ratio=ratio) for ratio in self.compress_ratios if ratio
        }

    @classmethod
    def get_swa_prefill_workspace(cls, model):
        """由 backend 复用计算临时区；模型在 KV 容量评估前调用，以计入实际显存占用。"""
        if model.args.run_mode == "decode":
            return None, 0
        from lightllm.models.deepseek_v4.config import create_attention_config

        config = create_attention_config(
            model_config=model.config,
            model_logical_layer_ids=tuple(layer.layer_num_ for layer in model.layers_infer),
            tp_rank=model.tp_rank_,
            tp_world_size=model.tp_world_size_,
        ).create_swa_config()
        page_token_num = config.page_token_size
        max_forward_tokens = max(model.batch_max_tokens, model.max_req_num) + 2 * model.tp_world_size_
        capacity = (
            triton.cdiv(
                max_forward_tokens + model.max_req_num * (config.window_size + 2 * page_token_num), page_token_num
            )
            * page_token_num
        )
        regions = 1 + int(model.args.enable_prefill_microbatch_overlap)
        # 每个 microbatch 使用独立临时区；最后一页留给 HOLD 请求。
        token_num = regions * capacity + page_token_num
        # 物理行宽由 SWA 原子配置给出，backend 仅决定 workspace 的 token 容量。
        slot_bytes = config.get_packed_kv_bytes(page_token_num) // page_token_num
        layer_count = len(config.logical_layer_ids)
        buffer = cls.get_gpu_workspace_buffer(
            key_name=f"deepseek_v4_swa_prefill_{model.att_state_role}",
            workspace_size=layer_count * token_num * slot_bytes,
            dtype=torch.uint8,
        )
        return buffer.view(layer_count, token_num, slot_bytes), capacity

    def get_att_input_params(self, infer_state, logical_layer_id: int):
        state = infer_state.swa_state
        swa_buffer = state.get_layer_view(logical_layer_id=logical_layer_id)
        config = state.config
        swa = swa_buffer.view(-1, config.get_packed_kv_bytes(config.page_token_size))
        ratio = self.layer_compress_ratios[logical_layer_id]
        compressed_kv = infer_state.get_compressed_kv_view(logical_layer_id=logical_layer_id).kv if ratio else None
        if infer_state.is_prefill:
            swa = infer_state.prefill_att_state.swa_buffer[config.logical_layer_ids.index(logical_layer_id)].view(
                -1, config.get_packed_kv_bytes(config.page_token_size),
            )
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
        # 原子配置提供计算视图；backend 不再计算编码字节数或解释压缩块布局。
        compressed_cache = None
        if compressed_packed is not None:
            config = self.compressed_configs[nsa_dict["compress_ratio"]]
            compressed_cache = config.get_attention_view(packed_kv=compressed_packed)
        swa_cache = self.swa_config.get_attention_view(packed_kv=swa_packed)

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

        swa_state = infer_state.swa_state
        self.swa_buffer, capacity = self.backend.get_swa_prefill_workspace(self.backend.model)
        self.HOLD_SWA_SLOT = self.swa_buffer.shape[1] - swa_state.config.page_token_size
        self.prefill_token_ranges = torch.zeros(
            (swa_state.request_capacity, 2), dtype=torch.int32, device=self.swa_buffer.device
        )
        self.swa_prefill_workspace_offsets = torch.full(
            (swa_state.request_capacity,), -1, dtype=torch.int32, device=self.swa_buffer.device
        )
        prepare_swa_prefill_workspace(
            swa_state=swa_state,
            prefill_state=self,
            b_req_idx=infer_state.b_req_idx,
            b_ready_cache_len=infer_state.b_ready_cache_len,
            b_seq_len=infer_state.b_seq_len,
            workspace_base=infer_state.microbatch_index * capacity,
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


DSV4_NSA_BACKENDS = {
    "fp8kv_dsa": {
        "flashmla_sparse": "nsa.dsv4_fp8_flashmla_sparse:DeepseekV4FlashMlaFp8SparseAttBackend",
    }
}
