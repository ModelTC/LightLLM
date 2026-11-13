"""V4 sparse attention on packed NVFP4 shared-KV pages."""

import torch

from .dsv4 import DeepseekV4SparseAttBackend
from ...triton_kernel.mla_att.packed_sparse import STAGED_KV_BYTES, QUERY_TILE_SIZE, packed_sparse_attention


class DeepseekV4Nvfp4SparseAttBackend(DeepseekV4SparseAttBackend):
    kv_type = "nvfp4kv"

    def __init__(self, model):
        super().__init__(model)
        if torch.cuda.get_device_capability()[0] != 9:
            raise ValueError("V4 NVFP4 packed attention requires a Hopper GPU with FA3")
        self._workspaces = self.allocate_attention_workspaces(model)

    @classmethod
    def prepare_workspaces(cls, model):
        super().prepare_workspaces(model)
        cls.allocate_attention_workspaces(model)

    @classmethod
    def allocate_attention_workspaces(cls, model):
        dim = model.config["head_dim"]
        assert model.data_type == torch.bfloat16, "V4 NVFP4 attention requires BF16 computation"
        capacity = STAGED_KV_BYTES // (dim * torch.bfloat16.itemsize)
        regions = 1 + int(model.args.enable_prefill_microbatch_overlap or model.args.enable_decode_microbatch_overlap)
        prefix = f"dsv4_nvfp4_{model.att_state_role}"
        kv = cls.get_gpu_workspace_buffer(prefix + "_kv", regions * capacity * dim, torch.bfloat16).view(
            regions, capacity, 1, dim
        )
        selected = cls.get_gpu_workspace_buffer(prefix + "_selected", regions * capacity, torch.int32).view(
            regions, capacity
        )
        lengths = cls.get_gpu_workspace_buffer(prefix + "_lengths", regions * QUERY_TILE_SIZE, torch.int32).view(
            regions, QUERY_TILE_SIZE
        )
        pages = cls.get_gpu_workspace_buffer(prefix + "_pages", capacity, torch.int32)
        pages.copy_(torch.arange(capacity, device=pages.device, dtype=torch.int32))
        cu_q = cls.get_gpu_workspace_buffer(prefix + "_cu_q", QUERY_TILE_SIZE + 1, torch.int32)
        cu_q.copy_(torch.arange(QUERY_TILE_SIZE + 1, device=cu_q.device, dtype=torch.int32))
        return [
            dict(kv=kv[i], selected=selected[i], lengths=lengths[i], pages=pages, cu_q=cu_q) for i in range(regions)
        ]

    def _sparse_attention(self, q, packed_kv, nsa_dict, sched_meta, out, workspace_index=0):
        ratio = nsa_dict["compress_ratio"]
        compressed = self.compressed_configs[ratio] if ratio else None
        return packed_sparse_attention(
            q,
            packed_kv[0],
            packed_kv[1],
            nsa_dict["swa_indices"],
            nsa_dict["swa_lengths"],
            nsa_dict.get("extra_indices"),
            nsa_dict.get("extra_lengths"),
            self.swa_config.kv_layout,
            compressed.kv_layout if compressed is not None else None,
            self.swa_config.page_token_size,
            compressed.block_token_size // ratio if compressed is not None else 1,
            nsa_dict["softmax_scale"],
            nsa_dict["attn_sink"],
            self._workspaces[workspace_index],
            out,
        )
