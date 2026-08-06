from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.mm_weight import MMWeightTpl
from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.mm_slicer import get_row_slice_mixin
from lightllm.utils.dist_utils import get_current_rank_in_dp, get_dp_world_size


class QKVGatedROWNMMWeight(MMWeightTpl):
    def __init__(
        self,
        in_dim,
        q_head_num,
        kv_head_num,
        head_dim,
        weight_names,
        data_type,
        bias_names=None,
        quant_method=None,
        tp_rank=None,
        tp_world_size=None,
    ):
        self.tp_rank_ = tp_rank if tp_rank is not None else get_current_rank_in_dp()
        self.tp_world_size_ = tp_world_size if tp_world_size is not None else get_dp_world_size()
        self.q_repeat_times = 1
        self.kv_repeat_times = self._get_kv_repeat_times(kv_head_num)
        assert (
            q_head_num % self.tp_world_size_ == 0
        ), f"q_head_num must be divisible by tp_world_size_, found {q_head_num} % {self.tp_world_size_}"
        q_hidden_size = (q_head_num // self.tp_world_size_) * head_dim
        kv_hidden_size = self._get_tp_padded_head_num(kv_head_num, self.kv_repeat_times) * head_dim
        super().__init__(
            in_dim=in_dim,
            out_dims=[q_hidden_size, kv_hidden_size, kv_hidden_size, q_hidden_size],
            weight_names=weight_names,
            bias_names=bias_names,
            data_type=data_type,
            quant_method=quant_method,
            tp_rank=self.tp_rank_,
            tp_world_size=self.tp_world_size_,
        )
        self.q_param_slicer = get_row_slice_mixin(
            self.quant_method.method_name,
            tp_rank=self.tp_rank_,
            tp_world_size=self.tp_world_size_,
            repeat_times=self.q_repeat_times,
        )
        self.kv_param_slicer = get_row_slice_mixin(
            self.quant_method.method_name,
            tp_rank=self.tp_rank_,
            tp_world_size=self.tp_world_size_,
            repeat_times=self.kv_repeat_times,
        )

    def _get_param_slicer(self, sub_child_index):
        if sub_child_index == 0 or sub_child_index == 3:
            return self.q_param_slicer
        return self.kv_param_slicer

    def load_hf_weights(self, weights):
        super().load_hf_weights(weights)
        if self.bias_names is not None:
            for sub_child_index, bias_name in enumerate(self.bias_names):
                if bias_name is None:
                    self.bias_list[sub_child_index].zero_()
                    self.bias_list[sub_child_index].load_ok = True

    def _get_kv_repeat_times(self, kv_head_num):
        assert kv_head_num % self.tp_world_size_ == 0 or self.tp_world_size_ % kv_head_num == 0, (
            f"kv_head_num must be divisible by tp_world_size_ or vice versa, "
            f"found {kv_head_num} % {self.tp_world_size_}"
        )
        if kv_head_num % self.tp_world_size_ == 0:
            return 1
        return self.tp_world_size_ // kv_head_num

    def _get_tp_padded_head_num(self, head_num, repeat_times):
        return repeat_times * head_num // self.tp_world_size_
