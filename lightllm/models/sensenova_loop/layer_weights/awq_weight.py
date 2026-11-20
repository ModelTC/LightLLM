"""Packed INT4 embedding and AWQ lm_head weights for SenseNova Loop."""

import torch
import triton

from lightllm.common.basemodel.layer_weights.meta_weights import EmbeddingWeight, ROWMMWeight
from lightllm.models.sensenova_loop.triton_kernel.awq import int4_embedding_kernel


class LoopINT4EmbeddingWeight(EmbeddingWeight):
    """Row-major INT4 embedding lookup with vocab-parallel sharding."""

    is_packed_embedding = True

    def __init__(self, dim, vocab_size, group_size, data_type):
        if dim % group_size or dim % 8:
            raise ValueError("Embedding dimensions must be divisible by group_size and eight")
        self.group_size = group_size
        super().__init__(dim, vocab_size, "model.embed_tokens.weight", data_type)

    def _create_weight(self):
        self.local_vocab_size = self.tp_vocab_end_id - self.tp_vocab_start_id
        self.qweight = torch.empty((self.local_vocab_size, self.dim // 8), dtype=torch.int32, device=self.device_id_)
        shape = (self.local_vocab_size, self.dim // self.group_size)
        # Keep scales in FP32 so F16 checkpoint scales are not rounded before lookup.
        self.scales = torch.empty(shape, dtype=torch.float32, device=self.device_id_)
        # The kernel consumes unpacked zero points; the checkpoint may store them packed.
        self.qzeros = torch.empty(shape, dtype=torch.uint8, device=self.device_id_)
        self.loaded = set()

    def load_hf_weights(self, weights):
        prefix = self.weight_name.removesuffix(".weight")
        for suffix in ("qweight", "scales", "qzeros"):
            name = f"{prefix}.{suffix}"
            if name not in weights:
                continue
            source, target = weights[name], getattr(self, suffix)
            packed_zeros = suffix == "qzeros" and source.dtype == torch.int32
            columns = triton.cdiv(target.shape[1], 8) if packed_zeros else target.shape[1]
            if source.shape != (self.vocab_size, columns):
                raise ValueError(f"Invalid packed embedding tensor shape: {name}")
            source = source[self.tp_vocab_start_id : self.tp_vocab_end_id]
            if packed_zeros:
                source = source.to(self.device_id_)
                shifts = torch.arange(8, device=source.device) * 4
                source = ((source.unsqueeze(-1) >> shifts) & 0xF).flatten(1)[:, : target.shape[1]]
            target.copy_(source)
            self.loaded.add(suffix)

    def verify_load(self):
        return self.loaded == {"qweight", "scales", "qzeros"}

    def _triton_forward(self, input_ids, out=None, alloc_func=torch.empty):
        input_ids = input_ids.reshape(-1)
        if out is None:
            out = alloc_func((input_ids.numel(), self.dim), dtype=self.data_type_, device=input_ids.device)
        int4_embedding_kernel[(input_ids.numel(), triton.cdiv(self.dim, 256))](
            input_ids.contiguous(),
            self.qweight,
            self.scales,
            self.qzeros,
            out,
            self.dim,
            self.tp_vocab_start_id,
            self.tp_vocab_end_id,
            self.local_vocab_size,
            self.group_size,
            256,
        )
        return out


class LoopAWQLMHeadWeight(ROWMMWeight):
    """Standard HF AWQ/Marlin lm_head with vocab-parallel output."""

    def __init__(self, dim, vocab_size, data_type, quant_method):
        super().__init__(
            in_dim=dim,
            out_dims=[vocab_size],
            weight_names="lm_head.weight",
            data_type=data_type,
            quant_method=quant_method,
        )
        self.vocab_size = vocab_size
        self.tp_vocab_start_id = self.tp_rank_ * self.out_dims[0]
        self.tp_vocab_end_id = self.tp_vocab_start_id + self.out_dims[0]

    def __call__(self, input, out=None, alloc_func=torch.empty):
        result = self.mm(input.t().contiguous(), use_custom_tensor_mananger=out is None).t()
        if out is not None:
            out.copy_(result)
            return out
        return result
