import json
import os
import tempfile

import torch
import torch.distributed as dist

from lightllm.utils.config_utils import get_model_architectures
from lightllm.utils.dist_utils import get_dp_size, get_dp_world_size, get_global_rank
from lightllm.utils.envs_utils import get_added_mtp_kv_layer_num, get_env_start_args, get_model_init_status
from lightllm.utils.log_utils import init_logger

from .mem_manager import MemoryManager
from .operator import NormalMemOperator

logger = init_logger(__name__)


def get_kv_quant_calibration_warmup_count() -> int:
    return int(os.getenv("KV_QUANT_CALIBRATION_WARMUP_COUNT", "0"))


def get_kv_quant_calibration_inference_count() -> int:
    return int(os.getenv("KV_QUANT_CALIBRATION_INFERENCE_COUNT", "4000"))


class ExportCalibrationMemOperator(NormalMemOperator):
    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        super().copy_kv_to_mem_manager(layer_index, mem_index, kv)
        self.mem_manager.update_calibration_data(kv, layer_index)


class ExportCalibrationMemoryManager(MemoryManager):
    """Keep the normal KV cache while collecting FP8 calibration statistics."""

    operator_class = ExportCalibrationMemOperator

    def __init__(
        self,
        size,
        dtype,
        head_num,
        head_dim,
        layer_num,
        always_copy=False,
        mem_fraction=0.9,
    ):
        super().__init__(size, dtype, head_num, head_dim, layer_num, always_copy, mem_fraction)

        self.qmax = torch.finfo(torch.float8_e4m3fn).max
        self.qmin = torch.finfo(torch.float8_e4m3fn).min
        self.total_head_num = head_num * get_dp_world_size()
        # A DSpark step can write draft layers more than once while it writes
        # target layers once.  Count each packed KV layer independently so a
        # busy draft layer cannot shorten another layer's sample window.
        self.calibration_counts = [0] * layer_num
        self._calibration_finalized = False
        self.scales = None

        scales_shape = [layer_num, 2 * head_num] if self._is_per_head_quant() else [layer_num, 2]
        self.abs_max = torch.zeros(scales_shape, dtype=torch.float32, device="cuda")

    @staticmethod
    def _is_per_head_quant() -> bool:
        """Only an explicitly selected FA3 prefill backend accepts per-head scales."""
        return "fa3" in get_env_start_args().llm_prefill_att_backend

    def update_calibration_data(self, kv: torch.Tensor, layer_index: int):
        inference_count = get_kv_quant_calibration_inference_count()
        warmup_count = get_kv_quant_calibration_warmup_count()
        if inference_count <= 0:
            raise ValueError("KV_QUANT_CALIBRATION_INFERENCE_COUNT must be positive")
        if warmup_count < 0:
            raise ValueError("KV_QUANT_CALIBRATION_WARMUP_COUNT must be non-negative")
        if not get_model_init_status() or self._calibration_finalized or kv.numel() == 0:
            return
        if not 0 <= layer_index < self.layer_num:
            raise IndexError(f"calibration layer index {layer_index} is outside [0, {self.layer_num})")

        count = self.calibration_counts[layer_index]
        collection_limit = warmup_count + inference_count
        if count >= collection_limit:
            return

        if count == 0 and layer_index == 0:
            logger.info("kv cache calibration mode will collect kv cache data for quantization calibration")

        if count >= warmup_count:
            if self._is_per_head_quant():
                kv_max = kv.abs().amax(dim=(0, 2)).to(torch.float32)
            else:
                k_max = kv[:, : self.head_num, :].abs().amax().to(torch.float32)
                v_max = kv[:, self.head_num :, :].abs().amax().to(torch.float32)
                kv_max = torch.stack((k_max, v_max))
            self.abs_max[layer_index] = torch.maximum(self.abs_max[layer_index], kv_max)

        self.calibration_counts[layer_index] += 1

        if all(count >= collection_limit for count in self.calibration_counts):
            self._calibration_finalized = True
            self._finalize_calibration_data()

    def _finalize_calibration_data(self):
        final_abs_max = self.abs_max
        if dist.is_initialized() and dist.get_world_size() > 1:
            if self._is_per_head_quant():
                world_size = dist.get_world_size()
                expected_world_size = get_dp_size() * get_dp_world_size()
                if world_size != expected_world_size:
                    raise ValueError(
                        f"global world size {world_size} does not match "
                        f"dp_size * dp_world_size {expected_world_size}"
                    )

                gathered_abs_max = [torch.zeros_like(self.abs_max) for _ in range(world_size)]
                dist.all_gather(gathered_abs_max, self.abs_max, group=None, async_op=False)
                gathered_abs_max = torch.stack(gathered_abs_max)

                # Default global ranks are laid out as [DP replica, TP rank].
                # Collapse data-parallel replicas before placing TP-local heads.
                final_abs_max = gathered_abs_max.view(get_dp_size(), get_dp_world_size(), *self.abs_max.shape).amax(
                    dim=0
                )
                k_max, v_max = torch.chunk(final_abs_max, 2, dim=-1)
                k_max = k_max.permute(1, 0, 2).flatten(1)
                v_max = v_max.permute(1, 0, 2).flatten(1)
                final_abs_max = torch.cat((k_max, v_max), dim=-1)
            else:
                dist.all_reduce(final_abs_max, op=dist.ReduceOp.MAX, group=None, async_op=False)

        if not torch.isfinite(final_abs_max).all():
            raise ValueError("FP8 KV calibration collected non-finite K/V values")

        self.abs_max = final_abs_max
        self.scales = final_abs_max / self.qmax
        self.scales = torch.where(self.scales > 0, self.scales, torch.ones_like(self.scales))

        if get_global_rank() == 0:
            self._export_calibration_data()

    def _export_calibration_data(self):
        model_arch = get_model_architectures(get_env_start_args().model_dir)
        draft_layer_num = get_added_mtp_kv_layer_num()
        target_layer_num = self.layer_num - draft_layer_num
        if target_layer_num <= 0:
            raise ValueError(
                f"packed KV layers ({self.layer_num}) must include target layers; draft layers={draft_layer_num}"
            )
        output_path = (
            "./kv_cache_calib_per_head_with_draft.json"
            if self._is_per_head_quant() and draft_layer_num > 0
            else "./kv_cache_calib.json"
        )
        cfg = {
            "version": "1.0",
            "architectures": model_arch,
            "quant_type": "per_head" if self._is_per_head_quant() else "per_tensor",
            "qmin": self.qmin,
            "qmax": self.qmax,
            "num_layers": self.layer_num,
            "num_target_layers": target_layer_num,
            "num_draft_layers": draft_layer_num,
            "num_head": self.total_head_num,
            "scales_shape": list(self.abs_max.shape),
            "scales": self.scales.cpu().numpy().tolist(),
        }
        output_dir = os.path.dirname(os.path.abspath(output_path))
        with tempfile.NamedTemporaryFile(mode="w", dir=output_dir, prefix=".kv_cache_calib.", delete=False) as f:
            json.dump(cfg, f, indent=4)
            temp_path = f.name
        os.replace(temp_path, output_path)
        logger.info(
            f"Export kv cache calibration data to {output_path}, "
            f"architectures: {model_arch}, "
            f"qmin: {self.qmin}, qmax: {self.qmax}, "
            f"total heads: {self.total_head_num}, "
            f"target layers: {target_layer_num}, draft layers: {draft_layer_num}, "
            f"scales_shape: {list(self.abs_max.shape)}"
        )
