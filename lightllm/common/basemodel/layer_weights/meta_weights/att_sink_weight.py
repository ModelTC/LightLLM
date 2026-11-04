import torch
from typing import Dict, Tuple
from .base_weight import BaseWeightTpl


class TpAttSinkWeight(BaseWeightTpl):
    def __init__(self, all_q_head_num: int, weight_name: str, data_type, head_padding_multiple: int = 1):
        super().__init__()
        self.all_q_head_num = all_q_head_num
        self.weight_name = weight_name
        self.data_type_ = data_type
        self._start_head_index, self._end_head_index = self._get_head_tp_split_params(all_head_num=self.all_q_head_num)
        self.tp_q_head_num = self._end_head_index - self._start_head_index
        # Round the local head count up to this multiple. Use 1 for no padding, as in GPT-OSS.
        # DSV4 FlashMLA pads Q heads to a multiple of 64. Sink weights must have the same length.
        self.padded_q_head_num = (
            (self.tp_q_head_num + head_padding_multiple - 1) // head_padding_multiple * head_padding_multiple
        )
        self._create_weight()

    def _create_weight(self):
        self.weight = torch.empty((self.padded_q_head_num,), dtype=self.data_type_, device=self.device_id_)
        self.weight.load_ok = False

    def load_hf_weights(self, weights: Dict[str, torch.Tensor]):
        if self.weight_name not in weights:
            return

        t_weight = weights[self.weight_name]
        self.weight[: self.tp_q_head_num].copy_(t_weight[self._start_head_index : self._end_head_index])
        # Set weights for dummy heads to zero at load time. Inference can use this tensor directly.
        self.weight[self.tp_q_head_num :].zero_()
        self.weight.load_ok = True

    def verify_load(self):
        return self.weight.load_ok

    def _get_head_tp_split_params(self, all_head_num: int) -> Tuple[int, int]:
        """
        Docstring for _get_head_tp_split_params,
        一个常用的tp 划分head获取head_index 范围的功能函数, 一些继承类可能会使用。
        :param self: Description
        :param weight: Description
        :type weight: torch.Tensor
        :return: Description
        :rtype: Tuple[int, int]
        """
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
