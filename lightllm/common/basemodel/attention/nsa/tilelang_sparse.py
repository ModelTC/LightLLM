"""TileLang sparse prefill attention for NSA/DSA models.

The kernel is provided by SGLang's kernel package. Decode intentionally keeps
using the FlashMLA implementation: TileLang is selected only for the much
larger prefill workload where it is advantageous on Hopper.
"""

import dataclasses

import torch

from ..base_att import AttControl
from .flashmla_sparse import (
    NsaFlashMlaSparseAttBackend,
    NsaFlashMlaSparsePrefillAttState,
)


def pad_sparse_indices(indices: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """Mask-pad the sparse index table to the TileLang block width."""

    if indices.ndim == 2:
        indices = indices.unsqueeze(1)
    if indices.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D sparse index tensor, got shape {tuple(indices.shape)}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    padding = (-indices.shape[-1]) % block_size
    if padding:
        indices = torch.cat(
            (indices, indices.new_full((*indices.shape[:-1], padding), -1)),
            dim=-1,
        )
    return indices


class NsaTilelangSparseAttBackend(NsaFlashMlaSparseAttBackend):
    """Use TileLang for prefill and FlashMLA for decode."""

    def create_att_prefill_state(self, infer_state):
        return NsaTilelangSparsePrefillAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class NsaTilelangSparsePrefillAttState(NsaFlashMlaSparsePrefillAttState):
    def _nsa_prefill_att(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        att_control: AttControl,
    ) -> torch.Tensor:
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd

        nsa_dict = att_control.nsa_prefill_dict
        # GLM packs its persistent MLA KV and FP8 indexer key in one allocation,
        # so the MLA slice has a padded token stride. TileLang requires packed
        # KV. For an uncached prefill, the freshly projected batch KV is the
        # same ragged ordering addressed by topk_indices; materialize that
        # compact view instead of copying the whole persistent cache. Prefix
        # cache requests retain the FlashMLA path, which supports the padded
        # persistent layout and global memory indices.
        if self.infer_state.max_cache_len != 0:
            return super()._nsa_prefill_att(q=q, kv=kv, att_control=att_control)

        compact_kv = nsa_dict["prefill_cache_kv"].contiguous()
        topk_indices = pad_sparse_indices(nsa_dict["topk_indices"])
        if topk_indices.dtype != torch.int32:
            topk_indices = topk_indices.to(torch.int32)

        output = tilelang_sparse_fwd(
            q=q,
            kv=compact_kv,
            indices=topk_indices,
            sm_scale=nsa_dict["softmax_scale"],
            d_v=nsa_dict["kv_lora_rank"],
        )
        # TileLang's generated kernel retains the synthetic batch dimension
        # inserted by its Python wrapper. LightLLM uses token-major tensors.
        if output.ndim == 4:
            if output.shape[0] != 1:
                raise RuntimeError(f"Unexpected TileLang sparse output shape {tuple(output.shape)}")
            output = output.squeeze(0)
        return output
