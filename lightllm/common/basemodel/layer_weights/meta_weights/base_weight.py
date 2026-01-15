import torch
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
from lightllm.utils.dist_utils import get_dp_world_size, get_current_rank_in_dp, get_current_device_id
from lightllm.utils.weight_checker import WeightChecker


class BaseWeight(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def load_hf_weights(self, weights):
        pass

    @abstractmethod
    def verify_load(self) -> bool:
        pass


class BaseWeightTpl(BaseWeight):
    def __init__(self, tp_rank: int = None, tp_world_size: int = None, data_type: torch.dtype = None):
        self.tp_world_size_ = tp_world_size if tp_world_size is not None else get_dp_world_size()
        self.tp_rank_ = tp_rank if tp_rank is not None else get_current_rank_in_dp()
        self.device_id_ = get_current_device_id()
        self.data_type_ = data_type
        self._weight_checker: Optional[WeightChecker] = None
        self._enable_checksum = False

    def load_hf_weights(self, weights):
        raise NotImplementedError("load_hf_weights must implement this method")

    def verify_load(self) -> bool:
        raise NotImplementedError("verify_load must implement this method")

    def enable_checksum_verification(self):
        """Enable checksum verification for this weight."""
        self._enable_checksum = True
        if self._weight_checker is None:
            self._weight_checker = WeightChecker(tp_rank=self.tp_rank_, tp_world_size=self.tp_world_size_)

    def get_weight_checker(self) -> Optional[WeightChecker]:
        """Get the weight checker instance."""
        return self._weight_checker

    def compute_checksums(self, force_bfloat16: bool = True) -> Dict[str, str]:
        """
        Compute checksums for all weights in this layer.
        Subclasses should override this to compute checksums for their specific weights.

        Args:
            force_bfloat16: Whether to cast to bfloat16 before hashing

        Returns:
            Dictionary mapping parameter names to checksums
        """
        return {}

    def verify_checksums(self, expected_checksums: Dict[str, str], force_bfloat16: bool = True) -> bool:
        """
        Verify checksums against expected values.

        Args:
            expected_checksums: Dictionary mapping parameter names to expected checksums
            force_bfloat16: Whether to cast to bfloat16 before hashing

        Returns:
            True if all checksums match, False otherwise
        """
        if not self._enable_checksum or self._weight_checker is None:
            return True

        computed_checksums = self.compute_checksums(force_bfloat16)
        all_passed = True

        for param_name, expected_checksum in expected_checksums.items():
            if param_name in computed_checksums:
                if computed_checksums[param_name] != expected_checksum:
                    all_passed = False

        return all_passed

    def _get_head_tp_split_params(self, weight: torch.Tensor) -> Tuple[int, int]:
        """
        Docstring for _get_head_tp_split_params,
        一个常用的tp 划分head获取head_index 范围的功能函数, 一些继承类可能会使用。
        :param self: Description
        :param weight: Description
        :type weight: torch.Tensor
        :return: Description
        :rtype: Tuple[int, int]
        """
        assert weight.ndim == 2

        all_head_num = weight.shape[0]
        tp_head_num = all_head_num // self.tp_world_size_

        if tp_head_num > 0:
            start_head_index = self.tp_rank_ * tp_head_num
            end_head_index = (self.tp_rank_ + 1) * tp_head_num
        else:
            # 当 tp_world_size 大于 all_head_num 时的特殊处理
            scale_size = self.tp_world_size_ // all_head_num
            assert self.tp_world_size_ % all_head_num == 0
            start_head_index = self.tp_rank_ // scale_size
            end_head_index = start_head_index + 1

        return start_head_index, end_head_index
