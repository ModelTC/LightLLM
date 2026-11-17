"""Single-GPU weights for Loop AWQ exports with independent input scales.

Unlike folded AWQ, this format stores one FP32 input_scale per source linear.
Projections with different scales must remain separate GEMMs.
"""

import torch
import triton

from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager
from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight
from lightllm.common.basemodel.layer_weights.meta_weights.base_weight import BaseWeightTpl
from lightllm.models.sensenova_loop.triton_kernel.awq import int4_embedding_kernel, scale_input_kernel


class InputScaledAWQWeight(ROWMMWeight):
    def __init__(self, in_dim, out_dim, weight_name, data_type, quant_method):
        super().__init__(
            in_dim=in_dim,
            out_dims=[out_dim],
            weight_names=weight_name,
            data_type=data_type,
            quant_method=quant_method,
        )
        if self.tp_world_size_ != 1 or quant_method.method_name != "awq_marlin":
            raise ValueError("Loop runtime-scale AWQ currently requires TP=1 and AWQ-Marlin")
        self.input_scale_name = weight_name.removesuffix(".weight") + ".input_scale"
        self.input_scale = torch.empty(in_dim, dtype=torch.float32, device=self.device_id_)
        self.input_scale_loaded = False

    def load_hf_weights(self, weights):
        super().load_hf_weights(weights)
        if self.input_scale_name in weights:
            scale = weights[self.input_scale_name]
            if scale.shape != self.input_scale.shape or not torch.all(torch.isfinite(scale) & (scale > 0)):
                raise ValueError(f"Invalid input scale: {self.input_scale_name}")
            self.input_scale.copy_(scale)
            self.input_scale_loaded = True

    def verify_load(self):
        return super().verify_load() and self.input_scale_loaded

    def mm(self, input_tensor, out=None, use_custom_tensor_mananger=True):
        input_tensor = input_tensor.contiguous()
        alloc = g_cache_manager.empty if use_custom_tensor_mananger else torch.empty
        scaled = alloc(input_tensor.shape, dtype=input_tensor.dtype, device=input_tensor.device)
        scale_input_kernel[(triton.cdiv(input_tensor.numel(), 1024),)](
            input_tensor, self.input_scale, scaled, input_tensor.numel(), self.in_dim, 1024
        )
        result = super().mm(scaled, use_custom_tensor_mananger=use_custom_tensor_mananger)
        if out is not None:
            out.copy_(result)
            return out
        return result


class InputScaledAWQBundle(BaseWeightTpl):
    """Concatenate independently scaled projections in the native kernel's order."""

    def __init__(self, in_dim, out_dims, weight_names, data_type, quant_factory):
        super().__init__(data_type=data_type)
        self.out_dims = out_dims
        self.projections = [
            InputScaledAWQWeight(in_dim, dim, name, data_type, quant_factory())
            for dim, name in zip(out_dims, weight_names)
        ]
        self.quant_method = self.projections[0].quant_method

    def _create_weight(self):
        pass

    def load_hf_weights(self, weights):
        for projection in self.projections:
            projection.load_hf_weights(weights)

    def verify_load(self):
        return all(projection.verify_load() for projection in self.projections)

    def mm(self, input_tensor, out=None, use_custom_tensor_mananger=True):
        if out is None:
            alloc = g_cache_manager.empty if use_custom_tensor_mananger else torch.empty
            out = alloc(
                (input_tensor.shape[0], sum(self.out_dims)), dtype=input_tensor.dtype, device=input_tensor.device
            )
        offset = 0
        for projection, dim in zip(self.projections, self.out_dims):
            projection.mm(
                input_tensor, out=out[:, offset : offset + dim], use_custom_tensor_mananger=use_custom_tensor_mananger
            )
            offset += dim
        return out


class LoopINT4EmbeddingWeight(BaseWeightTpl):
    """Row-major INT4 lookup; never materialize the full vocabulary in BF16."""

    is_packed_embedding = True

    def __init__(self, dim, vocab_size, group_size, data_type):
        super().__init__(data_type=data_type)
        if self.tp_world_size_ != 1:
            raise ValueError("Loop packed embeddings currently require TP=1")
        if dim % group_size or dim % 8:
            raise ValueError("Embedding dimensions must be divisible by group_size and eight")
        self.dim, self.vocab_size, self.group_size = dim, vocab_size, group_size
        self.tp_vocab_start_id, self.tp_vocab_end_id = 0, vocab_size
        self._create_weight()

    def _create_weight(self):
        self.qweight = torch.empty((self.vocab_size, self.dim // 8), dtype=torch.int32, device=self.device_id_)
        shape = (self.vocab_size, self.dim // self.group_size)
        self.scales = torch.empty(shape, dtype=self.data_type_, device=self.device_id_)
        self.qzeros = torch.empty(shape, dtype=torch.uint8, device=self.device_id_)
        self.loaded = set()

    def load_hf_weights(self, weights):
        for suffix in ("qweight", "scales", "qzeros"):
            name = f"model.embed_tokens.{suffix}"
            if name in weights:
                target = getattr(self, suffix)
                if weights[name].shape != target.shape or weights[name].dtype != target.dtype:
                    raise ValueError(f"Invalid packed embedding tensor: {name}")
                target.copy_(weights[name])
                self.loaded.add(suffix)

    def verify_load(self):
        return self.loaded == {"qweight", "scales", "qzeros"}

    def __call__(self, input_ids, out=None, alloc_func=torch.empty):
        if out is None:
            out = alloc_func((input_ids.numel(), self.dim), dtype=self.data_type_, device=input_ids.device)
        int4_embedding_kernel[(input_ids.numel(), triton.cdiv(self.dim, 256))](
            input_ids.contiguous(),
            self.qweight,
            self.scales,
            self.qzeros,
            out,
            self.dim,
            self.vocab_size,
            self.group_size,
            256,
        )
        return out


class LoopAWQLMHeadWeight(InputScaledAWQWeight):
    def __init__(self, dim, vocab_size, data_type, quant_method):
        super().__init__(dim, vocab_size, "lm_head.weight", data_type, quant_method)
        self.vocab_size = vocab_size
        self.tp_vocab_start_id, self.tp_vocab_end_id = 0, vocab_size

    def __call__(self, input, out=None, alloc_func=torch.empty):
        result = self.mm(input.t().contiguous()).t()
        if out is not None:
            out.copy_(result)
            return out
        return result
