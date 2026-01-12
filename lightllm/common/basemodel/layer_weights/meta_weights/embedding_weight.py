import torch
import numpy as np
from typing import Dict, Optional
from .base_weight import BaseWeightTpl
from .platform_op import PlatformAwareOp
from lightllm.common.basemodel.triton_kernel.embedding import embedding as embedding_kernel
from lightllm.utils.dist_utils import get_dp_world_size, get_current_rank_in_dp
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class EmbeddingWeight(BaseWeightTpl, PlatformAwareOp):
    def __init__(self, dim: int, vocab_size: int, weight_name: str, data_type: torch.dtype):
        BaseWeightTpl.__init__(self, data_type=data_type)
        self.dim = dim
        self.vocab_size = vocab_size
        self.tp_world_size_ = get_dp_world_size()
        self.tp_rank_ = get_current_rank_in_dp()
        # 计算 split_indexes
        split_indexes = np.linspace(0, self.vocab_size, self.tp_world_size_ + 1, dtype=np.int64)
        self.tp_vocab_start_id = int(split_indexes[self.tp_rank_])
        self.tp_vocab_end_id = int(split_indexes[self.tp_rank_ + 1])
        self.weight_name: str = weight_name
        self.data_type_ = data_type
        self._create_weight()
        PlatformAwareOp.__init__(self)

    def _create_weight(self):
        tp_vocab_size = self.tp_vocab_end_id - self.tp_vocab_start_id
        self.weight: torch.Tensor = torch.empty(tp_vocab_size, self.dim, dtype=self.data_type_, device=self.device_id_)

    def load_hf_weights(self, weights: Dict[str, torch.Tensor]):
        if self.weight_name not in weights:
            return
        t_weight = weights[self.weight_name]
        # init some params
        loaded_vocab_size = len(t_weight)
        assert (
            loaded_vocab_size == self.vocab_size
        ), f"loaded weight vocab_size: {loaded_vocab_size} != expected vocab_size: {self.vocab_size}"
        logger.info(f"loaded weight vocab_size: {self.vocab_size}")
        self.weight.copy_(t_weight[self.tp_vocab_start_id : self.tp_vocab_end_id, :].to(self.data_type_))

    def _native_forward(
        self, input_ids: torch.Tensor, out: Optional[torch.Tensor] = None, _alloc_func=torch.empty
    ) -> torch.Tensor:
        # Adjust input_ids for tp split
        adjusted_ids = input_ids - self.tp_vocab_start_id
        # Clamp to valid range for this partition
        adjusted_ids = torch.clamp(adjusted_ids, 0, self.weight.shape[0] - 1)
        # Use PyTorch native embedding
        result = torch.nn.functional.embedding(adjusted_ids, self.weight)
        if out is not None:
            out.copy_(result)
            return out
        return result

    def _cuda_forward(
        self, input_ids: torch.Tensor, out: Optional[torch.Tensor] = None, alloc_func=torch.empty
    ) -> torch.Tensor:
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

    def __call__(
        self, input_ids: torch.Tensor, out: Optional[torch.Tensor] = None, alloc_func=torch.empty
    ) -> torch.Tensor:
        return self._forward(input_ids=input_ids, out=out, alloc_func=alloc_func)


class LMHeadWeight(BaseWeightTpl, PlatformAwareOp):
    def __init__(
        self,
        dim: int,
        vocab_size: int,
        weight_name: str,
        data_type: torch.dtype,
        shared_weight: Optional[EmbeddingWeight] = None,
    ):
        BaseWeightTpl.__init__(self, data_type=data_type)
        self.dim = dim
        self.vocab_size = vocab_size
        self.tp_world_size_ = get_dp_world_size()
        self.tp_rank_ = get_current_rank_in_dp()
        # 计算 split_indexes
        split_indexes = np.linspace(0, self.vocab_size, self.tp_world_size_ + 1, dtype=np.int64)
        self.tp_vocab_start_id = int(split_indexes[self.tp_rank_])
        self.tp_vocab_end_id = int(split_indexes[self.tp_rank_ + 1])
        self.weight_name: str = weight_name
        self.data_type_ = data_type
        self._shared_weight = shared_weight
        if shared_weight is None:
            self._create_weight()
        PlatformAwareOp.__init__(self)

    @property
    def weight(self) -> torch.Tensor:
        if self._shared_weight is not None:
            return self._shared_weight.weight
        return self._weight

    def _create_weight(self):
        tp_vocab_size = self.tp_vocab_end_id - self.tp_vocab_start_id
        self._weight: torch.Tensor = torch.empty(tp_vocab_size, self.dim, dtype=self.data_type_, device=self.device_id_)

    def load_hf_weights(self, weights: Dict[str, torch.Tensor]):
        # When using shared weight, no need to load - EmbeddingWeight already loaded it
        if self._shared_weight is not None:
            return
        if self.weight_name not in weights:
            return
        t_weight = weights[self.weight_name]
        loaded_vocab_size = len(t_weight)
        assert (
            loaded_vocab_size == self.vocab_size
        ), f"loaded weight vocab_size: {loaded_vocab_size} != expected vocab_size: {self.vocab_size}"
        logger.info(f"loaded weight vocab_size: {self.vocab_size}")
        self._weight.copy_(t_weight[self.tp_vocab_start_id : self.tp_vocab_end_id, :].to(self.data_type_))

    def _native_forward(
        self, input: torch.Tensor, out: Optional[torch.Tensor] = None, _alloc_func=torch.empty
    ) -> torch.Tensor:
        assert input.ndim == 2
        result = torch.mm(self.weight, input)
        if out is not None:
            out.copy_(result)
            return out
        return result

    def _cuda_forward(
        self, input: torch.Tensor, out: Optional[torch.Tensor] = None, alloc_func=torch.empty
    ) -> torch.Tensor:
        assert input.ndim == 2
        if out is None:
            out = alloc_func(
                (self.weight.shape[0], input.shape[1]),
                dtype=input.dtype,
                device=input.device,
            )
        torch.mm(self.weight, input, out=out)
        return out

    def __call__(self, input: torch.Tensor, out: Optional[torch.Tensor] = None, alloc_func=torch.empty) -> torch.Tensor:
        return self._forward(input=input, out=out, alloc_func=alloc_func)


class NoTpPosEmbeddingWeight(BaseWeightTpl, PlatformAwareOp):
    def __init__(self, dim: int, max_position_embeddings: int, weight_name: str, data_type: torch.dtype):
        BaseWeightTpl.__init__(self, data_type=data_type)
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.weight_name: str = weight_name
        self.data_type_ = data_type
        self.tp_world_size_ = 1
        self.tp_rank_ = 0
        self._create_weight()
        PlatformAwareOp.__init__(self)

    def _create_weight(self):
        self.weight: torch.Tensor = torch.empty(
            self.max_position_embeddings, self.dim, dtype=self.data_type_, device=self.device_id_
        )

    def load_hf_weights(self, weights: Dict[str, torch.Tensor]):
        if self.weight_name not in weights:
            return
        t_weight = weights[self.weight_name]
        loaded_max_position_embeddings = t_weight.shape[0]
        assert (
            loaded_max_position_embeddings == self.max_position_embeddings
        ), f"max_position_embeddings: {loaded_max_position_embeddings} != expected: {self.max_position_embeddings}"
        logger.info(f"loaded weight max_position_embeddings: {self.max_position_embeddings}")
        self.weight.copy_(t_weight.to(self.data_type_))

    def _native_forward(
        self, input_ids: torch.Tensor, out: Optional[torch.Tensor] = None, _alloc_func=torch.empty
    ) -> torch.Tensor:
        # Use PyTorch native embedding
        result = torch.nn.functional.embedding(input_ids, self.weight)
        if out is not None:
            out.copy_(result)
            return out
        return result

    def _cuda_forward(
        self, input_ids: torch.Tensor, out: Optional[torch.Tensor] = None, alloc_func=torch.empty
    ) -> torch.Tensor:
        if out is None:
            out = alloc_func(
                (input_ids.shape[0], self.weight.shape[1]), dtype=self.weight.dtype, device=self.weight.device
            )
        embedding_kernel(
            input_ids=input_ids,
            weight=self.weight,
            vob_start_id=0,
            vob_end_id=self.max_position_embeddings,
            out=out,
        )
        return out

    def __call__(
        self, input_ids: torch.Tensor, out: Optional[torch.Tensor] = None, alloc_func=torch.empty
    ) -> torch.Tensor:
        return self._forward(input_ids=input_ids, out=out, alloc_func=alloc_func)
