import torch
import dataclasses


@dataclasses.dataclass
class LinearAttCacheConfig:
    tp_world_size: int
    # full att 的参数
    full_att_dtype: torch.dtype
    full_att_num_kv_heads: int
    full_att_head_dim: int

    # linear att 的参数
    num_linear_k_heads: int
    num_linear_v_heads: int
    head_linear_k_dim: int
    head_linear_v_dim: int
    conv_kernel_size: int
    linear_layer_num: int
    conv_state_dtype: torch.dtype
    ssm_state_dtype: torch.dtype
    full_attention_interval: int
    all_layer_num: int  # 包括 linear att 和 full att 的层加起来的层数

    def get_conv_dim(self):
        return self.head_linear_k_dim * self.num_linear_k_heads * 2 + self.head_linear_v_dim * self.num_linear_v_heads

    def get_conv_state_shape(self):
        return (self.get_conv_dim(), self.conv_kernel_size - 1)

    def get_ssm_state_shape(self):
        return (self.num_linear_v_heads, self.head_linear_k_dim, self.head_linear_v_dim)

    def get_conv_state_bytes(self):
        return self.get_conv_dim() * (self.conv_kernel_size - 1) * self.conv_state_dtype.itemsize

    def get_ssm_state_bytes(self):
        return self.num_linear_v_heads * self.head_linear_k_dim * self.head_linear_v_dim * self.ssm_state_dtype.itemsize

    def get_full_att_bytes(self):
        return 2 * self.full_att_num_kv_heads * self.full_att_head_dim * self.full_att_dtype.itemsize
