from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from .base import AttentionCompute, KVWrite
from lightllm.common.attention_state.types import FieldName

if TYPE_CHECKING:
    from lightllm.common.attention_state.full_att.packed_mla.formats import PackedMLALayerView
    from lightllm.common.attention_state.full_att.compressed_att.formats import CompressedAttentionLayerView


class MLAKVWriter(AttentionCompute, KVWrite):
    """普通 MLA latent 的计算视图和索引写入，不引入 GQA 的 K/V 拆分。"""

    def layer_view(self, storage, logical_layer_id: int) -> torch.Tensor:
        """按逻辑层号返回压缩 latent 视图，不拆分为 GQA 的 K/V head。"""
        return storage.layer_view(FieldName.KV, logical_layer_id)

    def write(self, storage, logical_layer_id: int, token_indices: torch.Tensor, kv: torch.Tensor) -> None:
        """通过 CUDA 算子将形状匹配的 MLA latent 数据写入指定物理 token 位置。"""
        assert token_indices.is_cuda, "token indices must reside on CUDA"
        if not len(token_indices):
            return
        assert storage.buffers[FieldName.KV].is_cuda, "KV writes require CUDA storage"
        from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv

        destindex_copy_kv(K=kv, DestLoc=token_indices, Out=storage.layer_view(FieldName.KV, logical_layer_id))


class PackedMLAKVWriter(AttentionCompute, KVWrite):
    """从统一规格借用 KV/indexer；write 只更新 KV，indexer 由其计算算子独立写入。"""

    def layer_view(self, storage, logical_layer_id: int) -> PackedMLALayerView:
        """分别返回 KV 与 packed indexer key/scale 的视图，保持原行步长。"""
        from lightllm.common.attention_state.full_att.packed_mla.formats import PackedMLALayerView

        return PackedMLALayerView(
            kv=storage.layer_view(FieldName.KV_DATA, logical_layer_id),
            indexer_k=storage.layer_view(FieldName.INDEXER, logical_layer_id),
        )

    def write(self, storage, logical_layer_id: int, token_indices: torch.Tensor, kv: torch.Tensor) -> None:
        """写入 MLA KV 及其量化参数，保留同一 token 行中的 indexer 区域。"""
        assert token_indices.is_cuda, "token indices must reside on CUDA"
        if not len(token_indices):
            return
        assert storage.buffers[FieldName.KV].is_cuda, "KV writes require CUDA storage"
        if self.config.kv_encoding == "fp8_group128":
            from lightllm.common.basemodel.triton_kernel.kv_copy.destindex_copy_kv_flashmla_fp8 import (
                destindex_copy_kv_flashmla_fp8,
            )

            nope_dim = self.config.latent_dim - self.config.rope_dim
            destindex_copy_kv_flashmla_fp8(
                KV_nope=kv[..., :nope_dim],
                KV_rope=kv[..., nope_dim:],
                DestLoc=token_indices,
                O_nope=storage.layer_view(FieldName.KV_NOPE, logical_layer_id).view(torch.float8_e4m3fn),
                O_scale=storage.layer_view(FieldName.KV_SCALE, logical_layer_id),
                O_rope=storage.layer_view(FieldName.KV_ROPE, logical_layer_id),
            )
        else:
            from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv

            destindex_copy_kv(
                K=kv,
                DestLoc=token_indices,
                Out=storage.layer_view(FieldName.KV_DATA, logical_layer_id),
            )


class FP8E8M0KVWriter(AttentionCompute, KVWrite):
    """固定 group64/E8M0 编码的 KV 写入；保留 indexer 和物理行 padding。"""

    def layer_view(self, storage, logical_layer_id: int) -> PackedMLALayerView:
        """返回固定 584 字节 KV 和 indexer key/scale 视图，保留物理行 stride。"""
        from lightllm.common.attention_state.full_att.packed_mla.formats import PackedMLALayerView

        return PackedMLALayerView(
            kv=storage.layer_view(FieldName.KV_DATA, logical_layer_id),
            indexer_k=storage.layer_view(FieldName.INDEXER, logical_layer_id),
        )

    def write(self, storage, logical_layer_id: int, token_indices: torch.Tensor, kv: torch.Tensor) -> None:
        """使用固定 group64/E8M0 算子量化并写入 KV，保留 indexer 和行尾 padding。"""
        from lightllm.common.attention_compute.fp8_e8m0_write import write_fp8_e8m0_packed_kv

        assert token_indices.is_cuda, "token indices must reside on CUDA"
        if token_indices.numel() == 0:
            return
        write_fp8_e8m0_packed_kv(
            kv=kv, token_indices=token_indices, packed_kv=self.layer_view(storage, logical_layer_id=logical_layer_id).kv
        )


class CompressedAttentionCompute(AttentionCompute):
    """提供融合压缩器使用的块视图，不提供独立 KVWrite 能力。"""

    def layer_view(self, storage, logical_layer_id: int) -> CompressedAttentionLayerView:
        """从同一物理压缩块取 KV 和 indexer 视图，不重排数据或创建副本。"""
        from lightllm.common.attention_state.full_att.compressed_att.formats import CompressedAttentionLayerView

        return CompressedAttentionLayerView(
            kv=storage.layer_view(FieldName.KV_DATA, logical_layer_id).squeeze(1),
            indexer_k=(
                storage.layer_view(FieldName.INDEXER, logical_layer_id).squeeze(1) if self.config.indexer_dim else None
            ),
        )
