import torch
import numpy as np
from typing import Dict, Optional
from .base_weight import BaseWeightTpl
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.common.basemodel.triton_kernel.embedding import embedding as embedding_kernel
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class EmbeddingWeight(BaseWeightTpl):
    def __init__(self, weight_name, data_type):
        super().__init__()
        self.weight_name: str = weight_name
        self.data_type_ = data_type
        self.weight: torch.Tensor = None

    def load_hf_weights(self, weights: Dict[str, torch.Tensor]):
        if self.weight_name not in weights or self.weight is not None:
            return

        t_weight = weights[self.weight_name]
        # init some params
        self.vocab_size = len(t_weight)
        split_indexes = np.linspace(0, self.vocab_size, self.tp_world_size_ + 1, dtype=np.int64)
        self.tp_vocab_start_id = int(split_indexes[self.tp_rank_])
        self.tp_vocab_end_id = int(split_indexes[self.tp_rank_ + 1])

        logger.info(f"loaded weight vocab_size: {self.vocab_size}")

        self.weight = (
            t_weight[self.tp_vocab_start_id : self.tp_vocab_end_id, :].to(self.data_type_).cuda(get_current_device_id())
        )

    def verify_load(self):
        return self.weight is not None

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
    def __init__(self, weight_name, data_type):
        super().__init__(weight_name, data_type)


class NoTpPosEmbeddingWeight(BaseWeightTpl):
    def __init__(self, weight_name, data_type):
        super().__init__()
        self.weight_name: str = weight_name
        self.data_type_ = data_type
        self.weight: torch.Tensor = None
        self.tp_world_size_ = 1
        self.tp_rank_ = 0

    def load_hf_weights(self, weights: Dict[str, torch.Tensor]):
        if self.weight_name not in weights or self.weight is not None:
            return

        t_weight = weights[self.weight_name]
        self.weight = t_weight.to(self.data_type_).cuda(get_current_device_id())
        self.end_position_id: int = t_weight.shape[0]
        logger.info(f"loaded weight end_position_id: {self.end_position_id}")

    def verify_load(self):
        return self.weight is not None

    def embedding(self, input_ids: torch.Tensor, out: Optional[torch.Tensor] = None, alloc_func=torch.empty):
        if out is None:
            out = alloc_func(
                (input_ids.shape[0], self.weight.shape[1]), dtype=self.weight.dtype, device=self.weight.device
            )

        embedding_kernel(
            input_ids=input_ids,
            weight=self.weight,
            vob_start_id=0,
            vob_end_id=self.end_position_id,
            out=out,
        )

        return out
