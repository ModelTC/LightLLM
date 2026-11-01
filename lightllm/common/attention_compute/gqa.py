from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Mapping
import torch
from .base import AttentionCompute, KVWrite
from lightllm.common.attention_state.types import FieldName

if TYPE_CHECKING:
    from lightllm.common.attention_state.full_att.token_storage import TokenStorage


class GQAKVWriter(AttentionCompute, KVWrite):
    """GQA 的计算视图与独立写入能力；逐 token 存储由 TokenStorage 拥有。"""

    def layer_view(self, storage: TokenStorage, logical_layer_id: int):
        return storage.layer_view(FieldName.K, logical_layer_id), storage.layer_view(FieldName.V, logical_layer_id)


class PlainKVWriter(GQAKVWriter):
    """将输入的 K|V 写到独立字段，保持输入精度。"""

    def write(
        self, storage: TokenStorage, logical_layer_id: int, token_indices: torch.Tensor, kv: torch.Tensor
    ) -> None:
        """通过 CUDA 算子将形状匹配的 K/V 数据写入指定物理 token 位置。"""
        assert token_indices.is_cuda, "token indices must reside on CUDA"
        if not len(token_indices):
            return
        assert storage.buffers[FieldName.K].is_cuda, "KV writes require CUDA storage"
        from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv

        # TODO：将同一层 K/V 的两次索引写入融合到一个 kernel，直接写入独立 K/V 存储。
        destindex_copy_kv(
            K=kv[:, : self.config.head_num],
            DestLoc=token_indices,
            Out=storage.layer_view(FieldName.K, logical_layer_id),
        )
        destindex_copy_kv(
            K=kv[:, self.config.head_num :],
            DestLoc=token_indices,
            Out=storage.layer_view(FieldName.V, logical_layer_id),
        )


class GroupQuantizedKVWriter(GQAKVWriter):
    """写入编码 KV 和逐组动态 scale，二者必须对应同一组 token 索引。"""

    def layer_view(self, storage: TokenStorage, logical_layer_id: int):
        return (
            (
                storage.layer_view(FieldName.K, logical_layer_id),
                storage.layer_view(FieldName.K_SCALE, logical_layer_id),
            ),
            (
                storage.layer_view(FieldName.V, logical_layer_id),
                storage.layer_view(FieldName.V_SCALE, logical_layer_id),
            ),
        )

    def write(
        self, storage: TokenStorage, logical_layer_id: int, token_indices: torch.Tensor, kv: torch.Tensor
    ) -> None:
        """量化输入浮点 K/V，并将编码数据和分组 scale 写入相同 token 位置。"""
        assert token_indices.is_cuda, "token indices must reside on CUDA"
        storage_layer_index = storage.layer_index(logical_layer_id)
        if not len(token_indices):
            return
        assert storage.buffers[FieldName.K].is_cuda, "KV writes require CUDA storage"
        # TODO：融合 K/V 两次量化写入，同时输出各自编码和动态 scale，保持分组及舍入语义。
        # K/V 分别写入独立存储，各自携带同一 token 的动态 scale。
        for offset, destination, scales_buffer in (
            (0, storage.buffers[FieldName.K], storage.buffers[FieldName.K_SCALE]),
            (self.config.head_num, storage.buffers[FieldName.V], storage.buffers[FieldName.V_SCALE]),
        ):
            values = kv[:, offset : offset + self.config.head_num]
            if self.config.bits == 4:
                from lightllm.common.basemodel.triton_kernel.kv_copy.ppl_int4kv_copy_kv import destindex_copy_int4kv

                destindex_copy_int4kv(
                    KV=values,
                    DestLoc=token_indices,
                    KV_buffer=destination[storage_layer_index],
                    KV_scale_buffer=scales_buffer[storage_layer_index],
                    quant_group_size=self.config.group_size,
                )
            else:
                from lightllm.common.basemodel.triton_kernel.kv_copy.ppl_int8kv_copy_kv import (
                    destindex_copy_quantize_kv,
                )

                destindex_copy_quantize_kv(
                    K=values,
                    DestLoc=token_indices,
                    Out=destination[storage_layer_index],
                    Out_scale=scales_buffer[storage_layer_index],
                    quant_group_dim=self.config.group_size,
                )


class StaticFP8KVWriter(GQAKVWriter):
    """持有固定标定参数并执行 FP8 编码；标定不随逐 token CPU/PD 页面传输。"""

    def __init__(self, config):
        super().__init__(config)
        self.scales = None
        self.q_scales = None
        self.cpu_scales = None
        self._prepared = False

    def bind_calibration(
        self,
        calibration: Mapping[str, object] | None,
        storage_layer_start: int,
    ) -> None:
        """在 prepare/存储分配前绑定一次标定；storage_layer_start 按物理层计数。"""
        assert not self._prepared and self.scales is None, "calibration must be bound once before storage allocation"
        config = self.config
        storage_layer_count = len(config.logical_layer_groups)
        if calibration is None:
            assert config.scale_mode == "per_tensor", "per-head FP8 requires a KV calibration file"
            scales = torch.ones(size=(storage_layer_count, 2), dtype=torch.float32)
            q_scales = None
        else:
            # 阶段一：检查文件编码和物理行覆盖范围。
            calibrated_layer_count = calibration["num_layers"]
            storage_layer_end = storage_layer_start + storage_layer_count
            assert (
                0 <= storage_layer_start < storage_layer_end <= calibrated_layer_count
            ), "calibration does not cover this State's physical layers"
            assert (
                calibration.get("quant_type", config.scale_mode) == config.scale_mode
            ), "calibration quantization mode does not match the State"
            limits = torch.finfo(torch.float8_e4m3fn)
            assert (
                calibration.get("qmin", limits.min) == limits.min and calibration.get("qmax", limits.max) == limits.max
            ), "calibration range does not match float8_e4m3fn"

            def scale_tensor(values: Mapping[str, object], shape: tuple[int, ...]) -> torch.Tensor:
                tensor = torch.tensor(data=values["scales"], dtype=torch.float32)
                assert tuple(values["scales_shape"]) == tensor.shape == shape, "invalid calibration scale shape"
                assert bool(
                    (torch.isfinite(tensor) & (tensor > 0)).all()
                ), "calibration scales must be finite and positive"
                return tensor[storage_layer_start:storage_layer_end]

            def shard(tensor: torch.Tensor, calibrated_heads: int) -> torch.Tensor:
                heads = config.head_num
                assert (
                    calibrated_heads > 0 and (heads * config.tp_world_size) % calibrated_heads == 0
                ), "calibration heads do not cover the model's KV heads"
                expanded = tensor.repeat_interleave(repeats=heads * config.tp_world_size // calibrated_heads, dim=-1)
                return expanded[:, config.tp_rank * heads : (config.tp_rank + 1) * heads].contiguous()

            # 阶段二：先按物理层取行，再分别切分 K、V、Q 的 head；标定 Tensor 暂留 CPU。
            if config.scale_mode == "per_head":
                calibrated_heads = calibration["num_head"]
                all_scales = scale_tensor(calibration, (calibrated_layer_count, 2 * calibrated_heads))
                k, v = all_scales.split(split_size=calibrated_heads, dim=-1)
                scales = torch.cat(tensors=(shard(k, calibrated_heads), shard(v, calibrated_heads)), dim=-1)
                q_config = calibration.get("q_calibration")
                q_scales = None
                if q_config is not None:
                    q_scales = shard(
                        scale_tensor(q_config, (calibrated_layer_count, q_config["num_head"])),
                        q_config["num_head"],
                    )
            else:
                scales = scale_tensor(calibration, (calibrated_layer_count, 2))
                q_scales = None

        # 阶段三：绑定标定数值；后续 allocate_storage 将其迁移到实际设备。
        self.scales = scales
        self.q_scales = q_scales

    def prepare(self, device: torch.device) -> None:
        """将已绑定的标定移到设备；缓存复用时标定一致性仍由调用方保证。"""
        assert self.scales is not None, "static FP8 calibration must be bound before allocation"
        if device.type != "meta":
            self.scales = self.scales.to(device=device)
            if self.q_scales is not None:
                self.q_scales = self.q_scales.to(device=device)
        self.cpu_scales = self.scales.detach().cpu().numpy()
        self._prepared = True

    def write(
        self, storage: TokenStorage, logical_layer_id: int, token_indices: torch.Tensor, kv: torch.Tensor
    ) -> None:
        """按固定 per-tensor 或 per-head scale 量化输入 K/V，写入 FP8 编码。"""
        assert token_indices.is_cuda, "token indices must reside on CUDA"
        storage_layer_index = storage.layer_index(logical_layer_id)
        if not len(token_indices):
            return
        assert storage.buffers[FieldName.K].is_cuda, "KV writes require CUDA storage"
        from lightllm.common.basemodel.triton_kernel.destindex_copy_kv_fp8 import destindex_copy_kv_fp8

        destindex_copy_kv_fp8(
            KV=kv,
            DestLoc=token_indices,
            scales=self.scales[storage_layer_index],
            Out=storage.layer_view(FieldName.K, logical_layer_id).view(torch.float8_e4m3fn),
            is_per_tensor_quant=self.config.scale_mode == "per_tensor",
            Out_v=storage.layer_view(FieldName.V, logical_layer_id).view(torch.float8_e4m3fn),
        )
