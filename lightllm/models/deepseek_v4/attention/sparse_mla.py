import inspect
from dataclasses import dataclass

import torch
import triton

from lightllm.common.attention.api import AttentionFamily, CapabilityResult
from lightllm.common.attention.backends.base import BaseAttentionBackend, PreparedPlan


@dataclass(frozen=True)
class DeepseekV4Params:
    swa: torch.Tensor
    prefill_swa: torch.Tensor
    compressed: torch.Tensor | None
    compress_ratio: int
    head_dim_v: int
    softmax_scale: float
    attn_sink: torch.Tensor


@dataclass(frozen=True)
class DeepseekV4Inputs:
    query: torch.Tensor
    swa_indices: torch.Tensor
    swa_lengths: torch.Tensor
    extra_indices: torch.Tensor | None = None
    extra_lengths: torch.Tensor | None = None


@dataclass
class SwaPrefillWorkspace:
    swa_buffer: torch.Tensor
    HOLD_SWA_SLOT: int
    prefill_token_ranges: torch.Tensor
    swa_prefill_workspace_offsets: torch.Tensor


def prepare_swa_workspace(swa_state, buffer, capacity, batch, slot):
    from lightllm.models.deepseek_v4.triton_kernel.swa_cache import prepare_swa_prefill_workspace

    metadata = SwaPrefillWorkspace(
        buffer,
        buffer.shape[1] - swa_state.config.page_token_size,
        torch.zeros((swa_state.request_capacity, 2), dtype=torch.int32, device=buffer.device),
        torch.full((swa_state.request_capacity,), -1, dtype=torch.int32, device=buffer.device),
    )
    prepare_swa_prefill_workspace(
        swa_state=swa_state,
        prefill_state=metadata,
        b_req_idx=batch.b_req_idx,
        b_ready_cache_len=batch.b_ready_cache_len,
        b_seq_len=batch.b_seq_len,
        workspace_base=slot * capacity,
    )
    return metadata


@dataclass
class DeepseekV4Plan(PreparedPlan):
    sched_meta_by_ratio: dict | None = None
    capture_sched_meta_by_ratio: dict | None = None


class DeepseekV4FlashMlaFp8SparseAttBackend(BaseAttentionBackend):
    supports_attention_sink = True
    supports_noncausal_decode = True
    family = AttentionFamily.SPARSE_MLA
    provider = "flashmla_sparse"
    dependency = "flash_mla"
    min_device_capability = (9, 0)
    activation_dtypes = frozenset({torch.float16, torch.bfloat16})
    cache_formats = frozenset({"fp8kv_dsa"})

    @classmethod
    def check_environment(cls, caps):
        result = super().check_environment(caps)
        if not result.supported:
            return result
        import flash_mla

        parameters = inspect.signature(flash_mla.flash_mla_with_kvcache).parameters
        required = {"out", "attn_sink", "indices", "topk_length", "extra_k_cache", "extra_topk_length"}
        missing = required.difference(parameters)
        if missing:
            return CapabilityResult.reject(
                "DSV4 requires the FlashMLA version pinned in docker/Dockerfile; missing: " + ", ".join(sorted(missing))
            )
        return result

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
        if hasattr(model, "swa_prefill_workspace"):
            return model.swa_prefill_workspace
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
        buffer = torch.empty(layer_count * token_num * slot_bytes, dtype=torch.uint8, device="cuda")
        model.swa_prefill_workspace = buffer.view(layer_count, token_num, slot_bytes), capacity
        return model.swa_prefill_workspace

    def _allocate_prefill_plan(self, capacity, capture, workspace):
        return DeepseekV4Plan(capacity, capture, workspace)

    def _allocate_decode_plan(self, capacity, capture, workspace):
        return DeepseekV4Plan(capacity, capture, workspace)

    def _prepare_prefill(self, plan):
        self._prepare_decode(plan)

    def _prepare_decode(self, plan):
        import flash_mla as flashmla

        if plan.sched_meta_by_ratio is None:
            plan.sched_meta_by_ratio = {ratio: flashmla.get_mla_metadata()[0] for ratio in self.compress_ratios}
            plan.capture_sched_meta_by_ratio = {ratio: flashmla.get_mla_metadata()[0] for ratio in self.compress_ratios}

    def _run(self, layer, plan, inputs, swa, out):
        import flash_mla as flashmla

        params = layer.params
        compressed = None
        if params.compressed is not None:
            compressed = self.compressed_configs[params.compress_ratio].get_attention_view(packed_kv=params.compressed)
        scheduler = (
            plan.capture_sched_meta_by_ratio if torch.cuda.is_current_stream_capturing() else plan.sched_meta_by_ratio
        )
        q = inputs.query
        if out is None:
            out = layer.alloc_tensor((q.shape[0], q.shape[1], params.head_dim_v), dtype=q.dtype, device=q.device)
        padded_out, _ = flashmla.flash_mla_with_kvcache(
            q=q.unsqueeze(1),
            k_cache=self.swa_config.get_attention_view(packed_kv=swa),
            block_table=None,
            cache_seqlens=None,
            head_dim_v=params.head_dim_v,
            tile_scheduler_metadata=scheduler[params.compress_ratio],
            num_splits=None,
            softmax_scale=params.softmax_scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=inputs.swa_indices,
            attn_sink=params.attn_sink,
            topk_length=inputs.swa_lengths,
            extra_k_cache=compressed,
            extra_indices_in_kvcache=inputs.extra_indices,
            extra_topk_length=inputs.extra_lengths,
            out=out.unsqueeze(1),
        )
        return padded_out[:, 0, : self.tp_q_head_num]

    def _forward_prefill(self, layer, plan, inputs: DeepseekV4Inputs, *, out=None):
        return self._run(layer, plan, inputs, layer.params.prefill_swa, out)

    def _forward_decode(self, layer, plan, inputs: DeepseekV4Inputs, *, out=None):
        return self._run(layer, plan, inputs, layer.params.swa, out)


DSV4_NSA_BACKENDS = {
    "fp8kv_dsa": {
        "flashmla_sparse": "lightllm.models.deepseek_v4.attention.sparse_mla:DeepseekV4FlashMlaFp8SparseAttBackend",
    }
}
