import dataclasses
import weakref
from typing import Optional

import torch
from ..base_att import BaseAttBackend, BasePrefillAttState, BaseDecodeAttState, AttControl, build_attention_batch
from lightllm.utils.dist_utils import get_dp_world_size, get_current_device_id
from lightllm.common.attention.backends.gqa import GQALayerConfig, GQALayerInputs
from lightllm.common.attention.backends.gqa.flashinfer import (
    FlashInferGQABuilderConfig,
    FlashInferGQADecodeBackend,
    FlashInferGQADecodeStepArgs,
    FlashInferGQAPrefillBackend,
    FlashInferGQAPrefillStepArgs,
)
from .env_utils import set_flashinfer_envs


class FlashInferAttBackend(BaseAttBackend):
    workspace_buffer_key = "flashinfer_fp"
    workspace_buffer_size = 512 * 1024 * 1024

    def __init__(self, model):
        set_flashinfer_envs()
        super().__init__(model=model)
        workspace_mib = getattr(model.args, "flashinfer_workspace_size", 512)
        if workspace_mib <= 0:
            raise ValueError("--flashinfer_workspace_size must be positive (MiB)")
        self.workspace_buffer_size = workspace_mib * 1024 * 1024
        self._init_infer_page_size()
        tp_world_size = get_dp_world_size()
        self.tp_q_head_num = model.config["num_attention_heads"] // tp_world_size
        self.tp_kv_head_num = max(model.config["num_key_value_heads"] // tp_world_size, 1)
        head_dim = model.config["hidden_size"] // model.config["num_attention_heads"]
        self.head_dim = model.config.get("head_dim", head_dim)
        self.q_data_type = model.data_type
        self.kv_data_type = self._kv_data_type()
        config = FlashInferGQABuilderConfig(
            num_heads=self.tp_q_head_num,
            num_kv_heads=self.tp_kv_head_num,
            head_dim=self.head_dim,
            page_size=self.infer_page_size,
            q_dtype=self.q_data_type,
            kv_dtype=self.kv_data_type,
        )
        self.prefill_step_builder = FlashInferGQAPrefillBackend.get_step_builder_cls()(
            config, workspace=self._workspace, max_seq_len=model.max_seq_length
        )
        # BaseAttBackend reuses one instance per model but reruns __init__ on every construction.
        # Keep the decode step builder: captured CUDA graphs use the wrappers and buffers it owns.
        if getattr(self, "decode_step_builder", None) is None:
            self.decode_step_builder = FlashInferGQADecodeBackend.get_step_builder_cls()(
                config,
                workspace=self._workspace,
                max_seq_len=model.max_seq_length,
                max_batch_size=model.graph_max_batch_size,
                device=get_current_device_id(),
            )

    def _init_infer_page_size(self):
        self.infer_page_size = self.model.args.page_size
        assert self.model.args.page_size % self.infer_page_size == 0, (
            f"model page_size {self.model.args.page_size} "
            f"must be divisible by infer_page_size {self.infer_page_size}"
        )

    def _kv_data_type(self) -> torch.dtype:
        return self.model.data_type

    def _workspace(self) -> torch.Tensor:
        # Allocated on first use, so prefill-only and decode-only nodes create it only when needed.
        return self.get_gpu_workspace_buffer(
            key_name=self.workspace_buffer_key, workspace_size=self.workspace_buffer_size
        )

    def create_att_prefill_state(self, infer_state) -> "FlashInferPrefillAttState":
        return FlashInferPrefillAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))

    def create_att_decode_state(self, infer_state) -> "FlashInferDecodeAttState":
        return FlashInferDecodeAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))


def _gqa_layer_config(q: torch.Tensor, k: torch.Tensor, att_control: AttControl, causal: bool) -> GQALayerConfig:
    # Pass every attention property the model asks for, so the impl rejects what FlashInfer cannot run.
    assert att_control.use_alibi is False, "FlashInfer GQA does not support alibi"
    return GQALayerConfig(
        num_heads=q.shape[1],
        num_kv_heads=k.shape[1],
        head_dim=q.shape[2],
        dtype=q.dtype,
        sliding_window=att_control.sliding_window if att_control.use_sliding_window else (-1, -1),
        causal=causal,
        has_sinks=att_control.use_att_sink,
    )


@dataclasses.dataclass
class FlashInferPrefillAttState(BasePrefillAttState):
    step_args: Optional[FlashInferGQAPrefillStepArgs] = None
    # The planned wrapper; the FP8 subclass still runs it directly.
    prefill_wrapper: object = None
    causal: bool = True

    def init_state(self):
        self.causal = self.backend.uses_causal_attention()
        self.step_args = self.backend.prefill_step_builder.build(build_attention_batch(self.infer_state, self.backend))
        self.prefill_wrapper = self.step_args.wrapper

    def prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        impl = self.backend.get_attention_impl(
            FlashInferGQAPrefillBackend, _gqa_layer_config(q, k, att_control, self.causal)
        )
        return impl.forward(
            self.step_args,
            GQALayerInputs(q=q, k_cache=k, v_cache=v),
            out=alloc_func(q.shape, dtype=q.dtype, device=q.device),
            alloc_func=alloc_func,
        )


@dataclasses.dataclass
class FlashInferDecodeAttState(BaseDecodeAttState):
    step_args: Optional[FlashInferGQADecodeStepArgs] = None
    # The planned wrapper; the FP8 subclass still runs it directly.
    decode_wrapper: object = None
    causal: bool = True

    def init_state(self):
        self.causal = self.backend.uses_causal_attention()
        # Replay steps also build: the step builder re-plans the wrapper the graph captured.
        self.step_args = self.backend.decode_step_builder.build(build_attention_batch(self.infer_state, self.backend))
        self.decode_wrapper = self.step_args.wrapper

    def decode_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ):
        impl = self.backend.get_attention_impl(
            FlashInferGQADecodeBackend, _gqa_layer_config(q, k, att_control, self.causal)
        )
        return impl.forward(
            self.step_args,
            GQALayerInputs(q=q, k_cache=k, v_cache=v),
            out=alloc_func(q.shape, dtype=q.dtype, device=q.device),
            alloc_func=alloc_func,
        )
