import dataclasses
import weakref
import torch
from ..base_att import BaseAttBackend, BasePrefillAttState, BaseDecodeAttState, AttControl
from typing import TYPE_CHECKING, Tuple
from lightllm.common.attention.backend import TensorAllocator
from lightllm.common.attention.backends.mla.triton import (
    MLADecodeInputs,
    MLAPrefillInputs,
    TritonMLADecodeBackend,
    TritonMLADecodeBatch,
    TritonMLAPrefillBackend,
    TritonMLAPrefillBatch,
)

if TYPE_CHECKING:
    from lightllm.common.basemodel.basemodel import TpPartBaseModel


class MlaTritonAttBackend(BaseAttBackend):
    def __init__(self, model: "TpPartBaseModel") -> None:
        super().__init__(model=model)
        self.prefill_backend = TritonMLAPrefillBackend()
        self.decode_backend = TritonMLADecodeBackend(
            q_heads=model.config["num_attention_heads"] // model.tp_world_size_,
            latent_dim=model.kv_lora_rank,
            rope_dim=model.qk_rope_head_dim,
            dtype=model.data_type,
        )

    def create_att_prefill_state(self, infer_state) -> "MlaTritonPrefillAttState":
        return MlaTritonPrefillAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))

    def create_att_decode_state(self, infer_state) -> "MlaTritonDecodeAttState":
        return MlaTritonDecodeAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))


@dataclasses.dataclass
class MlaTritonPrefillAttState(BasePrefillAttState):
    def init_state(self):
        pass

    def prefill_att(
        self,
        q: torch.Tensor,
        k: Tuple[torch.Tensor, torch.Tensor],
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        assert (
            att_control.use_alibi is False
            and att_control.use_sliding_window is False
            and att_control.use_att_sink is False
        )
        if q.shape[2] == 256:
            # glm model
            return self._contiguous_kv_prefill_att(q=q, k=k, v=v, att_control=att_control, alloc_func=alloc_func)

        assert att_control.mla_prefill
        state = self.infer_state
        inputs = MLAPrefillInputs(
            q=q, k_nope=k[0], k_rope=k[1], v=v, softmax_scale=att_control.mla_prefill_dict["softmax_scale"]
        )
        batch = TritonMLAPrefillBatch(
            b_q_start_loc=state.b_q_start_loc,
            b_kv_start_loc=state.b1_cu_kv_seq_len,
            b_seq_len=state.b_seq_len,
            b_ready_cache_len=state.b_ready_cache_len,
            max_q_seq_len=state.max_q_seq_len,
        )
        out = alloc_func((q.shape[0], q.shape[1], v.shape[-1]), dtype=q.dtype, device=q.device)
        return self.backend.prefill_backend.forward(batch=batch, inputs=inputs, out=out, alloc_func=alloc_func)

    def _contiguous_kv_prefill_att(
        self,
        q: torch.Tensor,
        k: Tuple[torch.Tensor, torch.Tensor],
        v: torch.Tensor,
        att_control: AttControl,
        alloc_func=torch.empty,
    ):
        from lightllm.common.attention.kernels.gqa.bf16.prefill.context_flashattention_nopad import (
            context_attention_fwd_contiguous_kv,
        )

        k_nope, k_rope = k
        q_head_num = q.shape[1]

        k_merged = torch.cat([k_nope.expand(-1, q_head_num, -1), k_rope.expand(-1, q_head_num, -1)], dim=-1)

        o_tensor = alloc_func(q.shape, dtype=q.dtype, device=q.device)

        context_attention_fwd_contiguous_kv(
            q,
            k_merged,
            v,
            o_tensor,
            self.infer_state.b_q_start_loc,
            self.infer_state.b1_cu_kv_seq_len,
            self.infer_state.b_seq_len,
            self.infer_state.max_q_seq_len,
            self.infer_state.b_ready_cache_len,
        )
        return o_tensor


@dataclasses.dataclass
class MlaTritonDecodeAttState(BaseDecodeAttState):
    batch: TritonMLADecodeBatch | None = None

    def init_state(self) -> None:
        state = self.infer_state
        if state.is_cuda_graph_replay:
            # Replay executes the captured state's plan, refreshed in copy_for_decode_cuda_graph.
            self.batch = None
            return
        if state.is_cuda_graph_capture:
            # Replays reuse this plan's kernel configuration, so it must not depend on capture-time lengths.
            tuning_seq_len = self.backend.model.graph_max_len_in_batch
        else:
            tuning_seq_len = state.total_token_num // state.batch_size
        self.batch = self.backend.decode_backend.plan(
            req_to_token_indexs=state.req_manager.req_to_token_indexs,
            b_req_idx=state.b_req_idx,
            b_seq_len=state.b_seq_len,
            tuning_seq_len=tuning_seq_len,
        )

    def copy_for_decode_cuda_graph(self, new_state: "MlaTritonDecodeAttState") -> None:
        super().copy_for_decode_cuda_graph(new_state)
        # The owning InferStateInfo has already copied the replay lengths into this batch's b_seq_len.
        self.backend.decode_backend.refresh(self.batch)

    def decode_att(
        self,
        q: Tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        assert (
            att_control.use_sliding_window is False
            and att_control.use_att_sink is False
            and att_control.use_alibi is False
        )
        assert v is None

        assert att_control.mla_decode
        q_nope, q_rope = q
        inputs = MLADecodeInputs(
            q_nope=q_nope, q_rope=q_rope, kv_cache=k, softmax_scale=att_control.mla_decode_dict["softmax_scale"]
        )
        out = alloc_func(q_nope.shape, dtype=q_nope.dtype, device=q_nope.device)
        return self.backend.decode_backend.forward(batch=self.batch, inputs=inputs, out=out, alloc_func=alloc_func)
