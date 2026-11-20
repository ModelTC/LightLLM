import dataclasses
import weakref
import torch
from ..base_att import (
    BaseAttBackend,
    BasePrefillAttState,
    BaseDecodeAttState,
    AttControl,
    build_attention_batch,
)
from typing import TYPE_CHECKING, Tuple
from lightllm.common.attention.backend import TensorAllocator
from lightllm.common.attention.backends.mla import (
    MLADecodeLayerInputs,
    MLADecodeLayerConfig,
    MLAPrefillLayerInputs,
    MLAPrefillLayerConfig,
)
from lightllm.common.attention.backends.mla.triton import (
    TritonMLADecodeBackend,
    TritonMLADecodeStepArgs,
    TritonMLAPrefillBackend,
    TritonMLAPrefillStepArgs,
)

if TYPE_CHECKING:
    from lightllm.common.basemodel.basemodel import TpPartBaseModel


class MlaTritonAttBackend(BaseAttBackend):
    def __init__(self, model: "TpPartBaseModel") -> None:
        super().__init__(model=model)
        self.prefill_step_builder = TritonMLAPrefillBackend.get_step_builder_cls()()
        # BaseAttBackend reuses one instance per model but reruns __init__ on every construction.
        # Keep the decode step builder: captured CUDA graphs read the schedule buffers it owns.
        if getattr(self, "decode_step_builder", None) is None:
            # One decode step builder serves all MLA layers of the model, which share these dimensions;
            # TritonMLADecodeImpl rejects step args built for a different configuration.
            self.decode_step_builder = TritonMLADecodeBackend.get_step_builder_cls()(
                num_heads=model.config["num_attention_heads"] // model.tp_world_size_,
                kv_lora_rank=model.kv_lora_rank,
                qk_rope_head_dim=model.qk_rope_head_dim,
                dtype=model.data_type,
            )

    def create_att_prefill_state(self, infer_state) -> "MlaTritonPrefillAttState":
        return MlaTritonPrefillAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))

    def create_att_decode_state(self, infer_state) -> "MlaTritonDecodeAttState":
        return MlaTritonDecodeAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))


@dataclasses.dataclass
class MlaTritonPrefillAttState(BasePrefillAttState):
    step_args: TritonMLAPrefillStepArgs | None = None

    def init_state(self) -> None:
        batch = build_attention_batch(self.infer_state, self.backend)
        self.step_args = self.backend.prefill_step_builder.build(batch)

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
        k_nope, k_rope = k
        layer_config = MLAPrefillLayerConfig(
            num_heads=q.shape[1],
            qk_nope_head_dim=k_nope.shape[-1],
            qk_rope_head_dim=k_rope.shape[-1],
            v_head_dim=v.shape[-1],
            softmax_scale=att_control.mla_prefill_dict["softmax_scale"],
            dtype=q.dtype,
        )
        impl = self.backend.get_attention_impl(TritonMLAPrefillBackend, layer_config)
        out = alloc_func((q.shape[0], q.shape[1], v.shape[-1]), dtype=q.dtype, device=q.device)
        return impl.forward(
            self.step_args,
            MLAPrefillLayerInputs(q=q, k_nope=k_nope, k_rope=k_rope, v=v),
            out=out,
            alloc_func=alloc_func,
        )

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
    step_args: TritonMLADecodeStepArgs | None = None

    def init_state(self) -> None:
        # Replay steps also build: the step builder rewrites the captured step's split schedule in place.
        batch = build_attention_batch(self.infer_state, self.backend)
        self.step_args = self.backend.decode_step_builder.build(batch)

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
        layer_config = MLADecodeLayerConfig(
            num_heads=q_nope.shape[1],
            kv_lora_rank=q_nope.shape[-1],
            qk_rope_head_dim=q_rope.shape[-1],
            softmax_scale=att_control.mla_decode_dict["softmax_scale"],
            dtype=q_nope.dtype,
        )
        impl = self.backend.get_attention_impl(TritonMLADecodeBackend, layer_config)
        out = alloc_func(q_nope.shape, dtype=q_nope.dtype, device=q_nope.device)
        return impl.forward(
            self.step_args,
            MLADecodeLayerInputs(q_nope=q_nope, q_rope=q_rope, kv_cache=k),
            out=out,
            alloc_func=alloc_func,
        )
