import dataclasses
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import triton


from ..base_att import AttControl, BaseAttBackend, BaseDecodeAttState, BasePrefillAttState

if TYPE_CHECKING:
    from flash_mla.flash_mla_interface import FlashMLASchedMeta
    from lightllm.common.basemodel.infer_struct import InferStateInfo


class DeepseekV4SparseAttBackend(BaseAttBackend):
    """Shared sparse-attention lifecycle; subclasses own encoding-specific computation."""

    kv_type = "fp8kv_dsa"

    def __init__(self, model):
        super().__init__(model=model)
        from lightllm.models.deepseek_v4.config import create_attention_config

        config = create_attention_config(
            model_config=model.config,
            model_logical_layer_ids=tuple(layer.layer_num_ for layer in model.layers_infer),
            tp_rank=model.tp_rank_,
            tp_world_size=model.tp_world_size_,
            kv_type=self.kv_type,
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
    def prepare_workspaces(cls, model):
        cls.get_swa_prefill_workspace(model)

    def create_scheduler_metadata(self):
        return {ratio: None for ratio in self.compress_ratios}

    @classmethod
    def get_swa_prefill_workspace(cls, model):
        """由 backend 复用计算临时区；模型在 KV 容量评估前调用，以计入实际显存占用。"""
        from lightllm.models.deepseek_v4.config import create_attention_config

        config = create_attention_config(
            model_config=model.config,
            model_logical_layer_ids=tuple(layer.layer_num_ for layer in model.layers_infer),
            tp_rank=model.tp_rank_,
            tp_world_size=model.tp_world_size_,
            kv_type=cls.kv_type,
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
        buffer = cls.get_gpu_workspace_buffer(
            key_name=f"deepseek_v4_swa_prefill_{model.att_state_role}_{config.kv_format}",
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
                -1,
                config.get_packed_kv_bytes(config.page_token_size),
            )
        return swa, compressed_kv

    def create_att_prefill_state(self, infer_state: "InferStateInfo") -> "_PrefillAttState":
        return _PrefillAttState(backend=self, workspace_index=infer_state.microbatch_index)

    def create_att_decode_state(self, infer_state: "InferStateInfo") -> "_DecodeAttState":
        return _DecodeAttState(backend=self, workspace_index=infer_state.microbatch_index)


@dataclasses.dataclass
class _PrefillAttState(BasePrefillAttState):
    workspace_index: int = 0
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
        self.sched_meta_by_ratio = self.backend.create_scheduler_metadata()

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

        padded_out = self.backend._sparse_attention(
            q,
            k,
            nsa_dict,
            self.sched_meta_by_ratio[nsa_dict["compress_ratio"]],
            out=out,
            workspace_index=self.workspace_index,
        )
        return padded_out[:, : self.backend.tp_q_head_num]


@dataclasses.dataclass
class _DecodeAttState(BaseDecodeAttState):
    workspace_index: int = 0
    sched_meta_by_ratio: dict[int, "FlashMLASchedMeta"] | None = None
    capture_sched_meta_by_ratio: dict[int, "FlashMLASchedMeta"] | None = None

    def init_state(self):
        self.sched_meta_by_ratio = self.backend.create_scheduler_metadata()
        # Keep capture metadata untouched by warmup so its GPU planning kernels enter the graph.
        self.capture_sched_meta_by_ratio = self.backend.create_scheduler_metadata()

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
        padded_out = self.backend._sparse_attention(
            q,
            k,
            nsa_dict,
            sched_meta_by_ratio[nsa_dict["compress_ratio"]],
            out=out,
            workspace_index=self.workspace_index,
        )
        return padded_out[:, : self.backend.tp_q_head_num]
