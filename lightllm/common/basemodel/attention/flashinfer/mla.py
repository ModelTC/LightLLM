import dataclasses
import weakref
from typing import Optional, Tuple

import torch

from lightllm.common.attention.backends.mla.flashinfer import (
    FlashInferMLABuilderConfig,
    FlashInferMLADecodeBackend,
    FlashInferMLADecodeStepArgs,
    FlashInferMLAPrefillBackend,
    FlashInferMLAPrefillStepArgs,
)
from lightllm.utils.dist_utils import get_current_device_id, get_dp_world_size
from ..base_att import BaseAttBackend, BasePrefillAttState, BaseDecodeAttState, AttControl, build_attention_batch
from ..mla_layer import mla_decode_layer, mla_prefill_layer
from .env_utils import set_flashinfer_envs


class MlaFlashInferAttBackend(BaseAttBackend):
    workspace_buffer_key = "flashinfer_mla"
    workspace_buffer_size = 256 * 1024 * 1024

    def __init__(self, model):
        set_flashinfer_envs()
        super().__init__(model=model)
        self._init_infer_page_size()
        softmax_scale = (model.qk_nope_head_dim + model.qk_rope_head_dim) ** (-0.5)
        if model.config["rope_scaling"] is not None:
            from lightllm.models.llama.yarn_rotary_utils import get_deepseek_mscale

            rope_scaling = model.config["rope_scaling"]
            mscale_all_dim = rope_scaling.get("mscale_all_dim", 0)
            if mscale_all_dim:
                mscale = get_deepseek_mscale(rope_scaling["factor"], mscale_all_dim)
                softmax_scale = softmax_scale * mscale * mscale
        # FlashInfer plans with the scale before the first layer; the impls check that every layer
        # passes the same value.
        config = FlashInferMLABuilderConfig(
            num_heads=model.config["num_attention_heads"] // get_dp_world_size(),
            qk_nope_head_dim=model.qk_nope_head_dim,
            qk_rope_head_dim=model.qk_rope_head_dim,
            v_head_dim=model.v_head_dim,
            kv_lora_rank=model.kv_lora_rank,
            page_size=self.infer_page_size,
            dtype=model.data_type,
            softmax_scale=softmax_scale,
        )
        self.prefill_step_builder = FlashInferMLAPrefillBackend.get_step_builder_cls()(
            config, workspace=self._workspace
        )
        # BaseAttBackend reuses one instance per model but reruns __init__ on every construction.
        # Keep the decode step builder: captured CUDA graphs use the wrappers and buffers it owns.
        if getattr(self, "decode_step_builder", None) is None:
            self.decode_step_builder = FlashInferMLADecodeBackend.get_step_builder_cls()(
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

    def _workspace(self) -> torch.Tensor:
        # Allocated on first use, so prefill-only and decode-only nodes create it only when needed.
        return self.get_gpu_workspace_buffer(
            key_name=self.workspace_buffer_key, workspace_size=self.workspace_buffer_size
        )

    def create_att_prefill_state(self, infer_state) -> "MlaFlashInferPrefillAttState":
        return MlaFlashInferPrefillAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))

    def create_att_decode_state(self, infer_state) -> "MlaFlashInferDecodeAttState":
        return MlaFlashInferDecodeAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))


@dataclasses.dataclass
class MlaFlashInferPrefillAttState(BasePrefillAttState):
    step_args: Optional[FlashInferMLAPrefillStepArgs] = None
    causal: bool = True

    def init_state(self):
        self.causal = self.backend.uses_causal_attention()
        self.step_args = self.backend.prefill_step_builder.build(build_attention_batch(self.infer_state, self.backend))

    def prefill_att(
        self,
        q: torch.Tensor,
        k: Tuple[torch.Tensor, torch.Tensor],
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        layer_inputs, layer_config = mla_prefill_layer(q, k, v, att_control, self.causal)
        impl = self.backend.get_attention_impl(FlashInferMLAPrefillBackend, layer_config)
        out = alloc_func((q.shape[0], q.shape[1], v.shape[-1]), dtype=q.dtype, device=q.device)
        return impl.forward(self.step_args, layer_inputs, out=out, alloc_func=alloc_func)


@dataclasses.dataclass
class MlaFlashInferDecodeAttState(BaseDecodeAttState):
    step_args: Optional[FlashInferMLADecodeStepArgs] = None
    causal: bool = True

    def init_state(self):
        self.causal = self.backend.uses_causal_attention()
        # Replay steps also build: the step builder refills the plan of the wrapper the graph captured.
        self.step_args = self.backend.decode_step_builder.build(build_attention_batch(self.infer_state, self.backend))

    def decode_att(
        self,
        q: Tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ):
        layer_inputs, layer_config = mla_decode_layer(q, k, v, att_control, self.causal)
        impl = self.backend.get_attention_impl(FlashInferMLADecodeBackend, layer_config)
        q_nope = layer_inputs.q_nope
        out = alloc_func(q_nope.shape, dtype=q_nope.dtype, device=q_nope.device)
        return impl.forward(self.step_args, layer_inputs, out=out, alloc_func=alloc_func)
