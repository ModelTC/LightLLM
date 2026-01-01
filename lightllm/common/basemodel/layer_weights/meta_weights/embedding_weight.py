import torch
import numpy as np
from typing import Dict, Optional
from .base_weight import BaseWeightTpl
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.common.basemodel.triton_kernel.embedding import embedding as embedding_kernel


class EmbeddingWeight(BaseWeightTpl):
    def __init__(self, weight_name, data_type, vocab_size: int):
        super().__init__()
        self.weight_name: str = weight_name
        self.data_type_ = data_type
        self.weight: torch.Tensor = None
        self.vocab_size = vocab_size
        split_indexes = np.linspace(0, self.vocab_size, self.tp_world_size_ + 1, dtype=np.int64)
        self.tp_vocab_start_id = split_indexes[self.tp_rank_]
        self.tp_vocab_end_id = split_indexes[self.tp_rank_ + 1]

    def load_hf_weights(self, weights: Dict[str, torch.Tensor]):
        if self.weight_name in weights and self.weight is None:
            t_weight = weights[self.weight_name]
            assert len(t_weight) == self.vocab_size
            self.weight = (
                t_weight[self.tp_vocab_start_id : self.tp_vocab_end_id, :]
                .to(self.data_type_)
                .cuda(get_current_device_id())
            )

    def verify_load(self):
        load_ok = True
        load_ok = load_ok and self.weight is not None

        return load_ok

    def embedding(self, input_ids: torch.Tensor, out: Optional[torch.Tensor] = None, alloc_func=torch.empty):
        if out is None:
            out = alloc_func(
                (input_ids.shape[0], self.weight.shape[1]), dtype=self.weight.dtype, device=self.weight.device
            )

        embedding_kernel(
            input_ids=input_ids,
            weight=self.weight,
            vob_start_id=self.tp_vocab_start_id,
            vob_end_id=self.tp_vocab_end_id,
            out=out,
        )

        return out

    def lm_head(self, input: torch.Tensor, out: Optional[torch.Tensor] = None, alloc_func=torch.empty):
        assert input.ndim == 2
        if out is None:
            out = alloc_func(
                (self.weight.shape[0], input.shape[1]),
                dtype=input.dtype,
                device=input.device,
            )

        torch.mm(self.weight, input, out=out)
        return out


class LMHeadWeight(EmbeddingWeight):
    def __init__(self, weight_name, data_type, vocab_size):
        super().__init__(weight_name, data_type, vocab_size)
