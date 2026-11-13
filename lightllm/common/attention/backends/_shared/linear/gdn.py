"""GDN tensor-parallel dimensions and input views shared by native and replay kernels."""

import torch


class GDNMetadata:
    def _init_linear_layer_metadata(self, network_config, tp_world_size):
        self.num_v_heads = network_config["linear_num_value_heads"]
        self.num_k_heads = network_config["linear_num_key_heads"]
        self.head_k_dim = network_config["linear_key_head_dim"]
        self.head_v_dim = network_config["linear_value_head_dim"]
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_dim = network_config["linear_conv_kernel_dim"]
        self.activation = network_config["hidden_act"]
        self.tp_qkvz_dim = (self.key_dim * 2 + self.value_dim * 2) // tp_world_size
        self.tp_ba_dim = self.num_v_heads * 2 // tp_world_size
        self.tp_num_k_heads = self.num_k_heads // tp_world_size
        self.tp_num_v_heads = self.num_v_heads // tp_world_size
        self.tp_key_dim = self.key_dim // tp_world_size
        self.tp_value_dim = self.value_dim // tp_world_size
        if self.num_v_heads % self.num_k_heads:
            raise ValueError("num_v_heads must be divisible by num_k_heads")
        self.num_v_heads_per_k_head = self.num_v_heads // self.num_k_heads
        self.ssm_state_dtype = self.spec.cache.state_dtype
        return

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
