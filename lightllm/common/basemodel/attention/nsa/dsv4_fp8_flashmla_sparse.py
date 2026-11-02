import dataclasses
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import triton
import flash_mla as flashmla


from ..base_att import BaseAttBackend, BaseAttState
from ..sparse import BoundCompressedSparse, CompressedSparseInput

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
    def _swa_prefill_workspace_shape(cls, model):
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
                max_forward_tokens + model.max_req_num * (config.window_size + 2 * page_token_num),
                page_token_num,
            )
            * page_token_num
        )
        if model.args.run_mode == "decode":
            # D 节点初始化仍会对 HOLD 请求执行一次 prefill；只保留末尾的 HOLD 页，
            # 不为普通请求预留长 prefill 临时区，实际请求的前缀由 P 节点传入。
            capacity = 0
        regions = 1 + int(model.args.enable_prefill_microbatch_overlap)
        # 每个 microbatch 使用独立临时区；最后一页留给 HOLD 请求。
        token_num = regions * capacity + page_token_num
        # 物理行宽由 SWA 原子配置给出，backend 仅决定 workspace 的 token 容量。
        slot_bytes = config.get_packed_kv_bytes(page_token_num) // page_token_num
        layer_count = len(config.logical_layer_ids)
        return (layer_count, token_num, slot_bytes), capacity

    def runtime_memory_demands(self):
        shape, _ = self._swa_prefill_workspace_shape(self.model)
        return self.workspace_memory_demands(
            f"deepseek_v4_swa_prefill_{self.model.att_state_role}",
            shape[0] * shape[1] * shape[2],
            lanes=1,
            dtype=torch.uint8,
        )

    def reserve_runtime_resources(self):
        self.get_swa_prefill_workspace(self.model)

    @classmethod
    def get_swa_prefill_workspace(cls, model):
        shape, capacity = cls._swa_prefill_workspace_shape(model)
        buffer = cls.get_gpu_workspace_buffer(
            key_name=f"deepseek_v4_swa_prefill_{model.att_state_role}",
            workspace_size=shape[0] * shape[1] * shape[2],
            dtype=torch.uint8,
        )
        return buffer.view(shape), capacity

    def bind_layer_attention(self, layer, resources, weight):
        window = resources["window"]
        config = window.implementation.config
        workspace, _ = self.get_swa_prefill_workspace(self.model)
        row = config.logical_layer_ids.index(layer.layer_num_)
        compressed = resources.get("kv")
        compressed_cache = None
        if compressed is not None:
            compressed_cache = compressed.implementation.config.get_attention_view(compressed.view.kv)
        return BoundCompressedSparse(
            window.resource,
            config.get_attention_view(workspace[row]),
            config.get_attention_view(window.view),
            compressed_cache,
            layer.compress_ratio,
            layer.v_head_dim,
            layer.softmax_scale,
            weight.attn_sink_.weight,
        )

    def _flashmla_att(self, inputs: CompressedSparseInput, binding: BoundCompressedSparse, swa_cache, sched_meta, out):
        padded_out, _ = flashmla.flash_mla_with_kvcache(
            q=inputs.q.unsqueeze(1),
            k_cache=swa_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=binding.value_dim,
            tile_scheduler_metadata=sched_meta,
            num_splits=None,
            softmax_scale=binding.softmax_scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=inputs.selection.swa_indices,
            attn_sink=binding.attention_sink,
            topk_length=inputs.selection.swa_lengths,
            extra_k_cache=binding.compressed_cache,
            extra_indices_in_kvcache=inputs.selection.extra_indices,
            extra_topk_length=inputs.selection.extra_lengths,
            out=out.unsqueeze(1),
        )
        return padded_out[:, 0]

    def create_att_prefill_state(self, infer_state: "InferStateInfo") -> "_PrefillAttState":
        return _PrefillAttState(backend=self)

    def create_att_decode_state(self, infer_state: "InferStateInfo") -> "_DecodeAttState":
        return _DecodeAttState(backend=self)


@dataclasses.dataclass
class _PrefillAttState(BaseAttState):
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

    def run(
        self,
        inputs: CompressedSparseInput,
        binding: BoundCompressedSparse,
        alloc_func=torch.empty,
        *,
        out=None,
    ):
        if out is None:
            out = alloc_func(
                (inputs.q.shape[0], inputs.q.shape[1], binding.value_dim),
                dtype=inputs.q.dtype,
                device=inputs.q.device,
            )
        padded_out = self.backend._flashmla_att(
            inputs, binding, binding.prefill_cache, self.sched_meta_by_ratio[binding.compress_ratio], out
        )
        return padded_out[:, : self.backend.tp_q_head_num]


@dataclasses.dataclass
class _DecodeAttState(BaseAttState):
    sched_meta_by_ratio: dict[int, "FlashMLASchedMeta"] | None = None
    capture_sched_meta_by_ratio: dict[int, "FlashMLASchedMeta"] | None = None

    def init_state(self):
        self.sched_meta_by_ratio = {ratio: flashmla.get_mla_metadata()[0] for ratio in self.backend.compress_ratios}
        # Keep capture metadata untouched by warmup so its GPU planning kernels enter the graph.
        self.capture_sched_meta_by_ratio = {
            ratio: flashmla.get_mla_metadata()[0] for ratio in self.backend.compress_ratios
        }

    def run(self, inputs: CompressedSparseInput, binding: BoundCompressedSparse, alloc_func=torch.empty):
        sched_meta = (
            self.capture_sched_meta_by_ratio if torch.cuda.is_current_stream_capturing() else self.sched_meta_by_ratio
        )
        out = alloc_func(
            (inputs.q.shape[0], inputs.q.shape[1], binding.value_dim),
            dtype=inputs.q.dtype,
            device=inputs.q.device,
        )
        padded_out = self.backend._flashmla_att(
            inputs, binding, binding.decode_cache, sched_meta[binding.compress_ratio], out
        )
        return padded_out[:, : self.backend.tp_q_head_num]


DSV4_NSA_BACKENDS = {
    "fp8kv_dsa": {
        "flashmla_sparse": "nsa.dsv4_fp8_flashmla_sparse:DeepseekV4FlashMlaFp8SparseAttBackend",
    }
}
