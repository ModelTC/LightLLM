import json
import os

import torch
import torch.distributed as dist

from lightllm.utils.config_utils import get_model_architectures
from lightllm.utils.dist_utils import get_dp_size, get_dp_world_size, get_global_rank
from lightllm.utils.envs_utils import get_env_start_args, get_model_init_status
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
        self.count = 0
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
        if not get_model_init_status() or self.count >= warmup_count + inference_count:
            return

        if self.count == 0 and layer_index == 0:
            logger.info("kv cache calibration mode will collect kv cache data for quantization calibration")

        if self.count >= warmup_count:
            if self._is_per_head_quant():
                kv_max = kv.abs().amax(dim=(0, 2)).to(torch.float32)
            else:
                k_max = kv[:, : self.head_num, :].abs().amax().to(torch.float32)
                v_max = kv[:, self.head_num :, :].abs().amax().to(torch.float32)
                kv_max = torch.stack((k_max, v_max))
            self.abs_max[layer_index] = torch.maximum(self.abs_max[layer_index], kv_max)

            if self.count == warmup_count + inference_count - 1 and layer_index == self.layer_num - 1:
                self._finalize_calibration_data()

        if layer_index == self.layer_num - 1:
            self.count += 1

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

        self.abs_max = final_abs_max
        self.scales = final_abs_max / self.qmax
        self.scales = torch.where(self.scales > 0, self.scales, torch.ones_like(self.scales))

        if get_global_rank() == 0:
            self._export_calibration_data()

    def _export_calibration_data(self):
        model_arch = get_model_architectures(get_env_start_args().model_dir)
        cfg = {
            "version": "1.0",
            "architectures": model_arch,
            "quant_type": "per_head" if self._is_per_head_quant() else "per_tensor",
            "qmin": self.qmin,
            "qmax": self.qmax,
            "num_layers": self.layer_num,
            "num_head": self.total_head_num,
            "scales_shape": list(self.abs_max.shape),
            "scales": self.scales.cpu().numpy().tolist(),
        }
        with open("./kv_cache_calib.json", "w") as f:
            json.dump(cfg, f, indent=4)
        logger.info(
            f"Export kv cache calibration data to kv_cache_calib.json, "
            f"architectures: {model_arch}, "
            f"qmin: {self.qmin}, qmax: {self.qmax}, "
            f"total heads: {self.total_head_num}, "
            f"scales_shape: {list(self.abs_max.shape)}"
        )
