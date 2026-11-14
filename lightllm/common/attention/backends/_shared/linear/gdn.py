"""GDN tensor-parallel dimensions and input views shared by native and replay kernels."""

import torch


class GDNMetadata:
    def _init_linear_layer_metadata(self):
        spec = self.spec
        self.head_k_dim = spec.head_size
        self.head_v_dim = spec.value_head_size
        self.tp_num_k_heads = spec.cache.key_heads
        self.tp_num_v_heads = spec.config.value_heads
        self.tp_key_dim = self.head_k_dim * self.tp_num_k_heads
        self.tp_value_dim = self.head_v_dim * self.tp_num_v_heads
        self.tp_qkvz_dim = self.tp_key_dim * 2 + self.tp_value_dim * 2
        self.tp_ba_dim = self.tp_num_v_heads * 2
        self.conv_kernel_dim = spec.config.conv_kernel_size
        self.activation = spec.config.activation
        if self.tp_num_v_heads % self.tp_num_k_heads:
            raise ValueError("value_heads must be divisible by key_heads")
        self.num_v_heads_per_k_head = self.tp_num_v_heads // self.tp_num_k_heads
        self.ssm_state_dtype = spec.cache.state_dtype

    def _split_qkvzba(self, mixed_qkvzba):
        qkv_dim = self.tp_key_dim * 2 + self.tp_value_dim
        z_end = qkv_dim + self.tp_value_dim
        b_end = z_end + self.tp_num_v_heads
        mixed_qkv = mixed_qkvzba[:, :qkv_dim]
        z = mixed_qkvzba[:, qkv_dim:z_end].view(-1, self.tp_num_v_heads, self.head_v_dim)
        b = mixed_qkvzba[:, z_end:b_end]
        a = mixed_qkvzba[:, b_end:]
        return (mixed_qkv, z, b, a)

    def _rearrange_mixed_qkv(self, mixed_qkv, decode=False):
        if decode:
            query, key, value = torch.split(mixed_qkv, [self.tp_key_dim, self.tp_key_dim, self.tp_value_dim], dim=-1)
            batch_size = mixed_qkv.shape[0]
            query = query.view(batch_size, 1, self.tp_num_k_heads, self.head_k_dim)
            key = key.view(batch_size, 1, self.tp_num_k_heads, self.head_k_dim)
            value = value.view(batch_size, 1, self.tp_num_v_heads, self.head_v_dim)
            return (query, key, value)
        else:
            query, key, value = torch.split(mixed_qkv, [self.tp_key_dim, self.tp_key_dim, self.tp_value_dim], dim=-1)
            seq_len = query.shape[0]
            query = query.view(1, seq_len, self.tp_num_k_heads, self.head_k_dim)
            key = key.view(1, seq_len, self.tp_num_k_heads, self.head_k_dim)
            value = value.view(1, seq_len, self.tp_num_v_heads, self.head_v_dim)
            return (query, key, value)
