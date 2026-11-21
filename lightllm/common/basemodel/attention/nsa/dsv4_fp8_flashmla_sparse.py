"""DeepSeek V4 sparse attention: FlashMLA over the FP8 sliding-window cache plus the compressed caches.

The model resolves each layer's packed caches (DeepseekV4InferStateInfo.get_att_kv) and indices; this
adapter turns the packed caches into FlashMLA views with the model's cache layouts and runs the impl.
"""

import dataclasses
from typing import Optional, Tuple

import torch

from lightllm.common.attention.backends.sparse_mla import SWASparseMLALayerConfig, SWASparseMLALayerInputs
from lightllm.common.attention.backends.sparse_mla.flashmla_swa import (
    FlashMLASWADecodeBackend,
    FlashMLASWAPrefillBackend,
    FlashMLASWAStepArgs,
)
from ..base_att import AttControl, BaseAttBackend, BaseDecodeAttState, BasePrefillAttState


class DeepseekV4FlashMlaFp8SparseAttBackend(BaseAttBackend):
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
        self.swa_config = config.create_swa_config()
        self.compressed_configs = {
            ratio: config.create_compressed_config(compress_ratio=ratio) for ratio in self.compress_ratios if ratio
        }
        self.prefill_step_builder = FlashMLASWAPrefillBackend.get_step_builder_cls()(
            compress_ratios=self.compress_ratios, for_cuda_graph_capture=False
        )
        self.decode_step_builder = FlashMLASWADecodeBackend.get_step_builder_cls()(
            compress_ratios=self.compress_ratios, for_cuda_graph_capture=True
        )

    def layer(self, q, packed_kv, nsa_dict) -> Tuple[SWASparseMLALayerInputs, SWASparseMLALayerConfig]:
        """Build the layer config and inputs from the model's packed caches and nsa dict."""
        swa_packed, compressed_packed = packed_kv
        ratio = nsa_dict["compress_ratio"]
        compressed_cache = None
        if compressed_packed is not None:
            compressed_cache = self.compressed_configs[ratio].get_attention_view(packed_kv=compressed_packed)
        attn_sink = nsa_dict.get("attn_sink")
        layer_config = SWASparseMLALayerConfig(
            num_heads=q.shape[1],
            head_dim_v=nsa_dict["head_dim_v"],
            softmax_scale=nsa_dict["softmax_scale"],
            dtype=q.dtype,
            compress_ratio=ratio,
            has_attn_sink=attn_sink is not None,
        )
        layer_inputs = SWASparseMLALayerInputs(
            q=q,
            swa_cache=self.swa_config.get_attention_view(packed_kv=swa_packed),
            swa_indices=nsa_dict["swa_indices"],
            swa_lengths=nsa_dict["swa_lengths"],
            compressed_cache=compressed_cache,
            extra_indices=nsa_dict.get("extra_indices"),
            extra_lengths=nsa_dict.get("extra_lengths"),
            attn_sink=attn_sink,
        )
        return layer_inputs, layer_config

    # The step args do not depend on the step's requests, so the states keep no InferStateInfo.
    def create_att_prefill_state(self, infer_state) -> "_PrefillAttState":
        return _PrefillAttState(backend=self)

    def create_att_decode_state(self, infer_state) -> "_DecodeAttState":
        return _DecodeAttState(backend=self)


def _run(backend, backend_cls, step_args, q, packed_kv, nsa_dict, out):
    layer_inputs, layer_config = backend.layer(q, packed_kv, nsa_dict)
    impl = backend.get_attention_impl(backend_cls, layer_config)
    padded_out = impl.forward(step_args, layer_inputs, out=out)
    return padded_out[:, : backend.tp_q_head_num]


@dataclasses.dataclass
class _PrefillAttState(BasePrefillAttState):
    step_args: Optional[FlashMLASWAStepArgs] = None

    def init_state(self):
        self.step_args = self.backend.prefill_step_builder.build()

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
        return _run(self.backend, FlashMLASWAPrefillBackend, self.step_args, q, k, nsa_dict, out)


@dataclasses.dataclass
class _DecodeAttState(BaseDecodeAttState):
    step_args: Optional[FlashMLASWAStepArgs] = None

    def init_state(self):
        self.step_args = self.backend.decode_step_builder.build()

    def decode_att(
        self,
        q: torch.Tensor,
        k: Tuple[torch.Tensor, Optional[torch.Tensor]],
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        nsa_dict = att_control.nsa_decode_dict
        out = alloc_func((q.shape[0], q.shape[1], nsa_dict["head_dim_v"]), dtype=q.dtype, device=q.device)
        return _run(self.backend, FlashMLASWADecodeBackend, self.step_args, q, k, nsa_dict, out)


DSV4_NSA_BACKENDS = {
    "fp8kv_dsa": {
        "flashmla_sparse": "nsa.dsv4_fp8_flashmla_sparse:DeepseekV4FlashMlaFp8SparseAttBackend",
    }
}
