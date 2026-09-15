"""In-process FP8 KV calibration collector.

The collector intentionally owns only rank-local statistics.  The calibration tool
starts/stops collection explicitly and merges rank snapshots on CPU, so normal
inference never enters calibration collectives or writes calibration artifacts.
"""

from __future__ import annotations

from typing import Any

import torch

from lightllm.utils.dist_utils import get_global_rank
from lightllm.utils.config_utils import get_model_architectures
from lightllm.utils.envs_utils import get_added_mtp_kv_layer_num, get_env_start_args, get_model_init_status

from .mem_manager import MemoryManager
from .operator import NormalMemOperator


class ExportCalibrationMemOperator(NormalMemOperator):
    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        super().copy_kv_to_mem_manager(layer_index, mem_index, kv)
        self.mem_manager.update_calibration_data(kv, layer_index)


class ExportCalibrationMemoryManager(MemoryManager):
    """Normal KV storage plus an explicitly controlled rank-local max collector."""

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
        self._init_calibration_state()

    def _init_calibration_state(self) -> None:
        """Initialize collection after any normal/hybrid MemoryManager init."""
        self.qmax = torch.finfo(torch.float8_e4m3fn).max
        self.qmin = torch.finfo(torch.float8_e4m3fn).min
        shape = [self.layer_num, 2 * self.head_num] if self._is_per_head_quant() else [self.layer_num, 2]
        self.abs_max = torch.zeros(shape, dtype=torch.float32, device="cuda")
        self.calibration_counts = [0] * self.layer_num
        self.observed_token_rows = [0] * self.layer_num
        self._calibration_active = False

    @staticmethod
    def _is_per_head_quant() -> bool:
        return "fa3" in get_env_start_args().llm_prefill_att_backend

    def begin_calibration(self) -> dict[str, Any]:
        """Reset this rank's state and begin collecting future KV writes."""
        # Do not expose active collection until prior stream work and the reset
        # have completed on this rank's actual CUDA device.
        if self.abs_max.is_cuda:
            torch.cuda.synchronize(self.abs_max.device)
        self._calibration_active = False
        self.abs_max.zero_()
        self.calibration_counts = [0] * self.layer_num
        self.observed_token_rows = [0] * self.layer_num
        if self.abs_max.is_cuda:
            torch.cuda.synchronize(self.abs_max.device)
        self._calibration_active = True
        return self.calibration_status()

    def calibration_status(self) -> dict[str, Any]:
        return {
            "rank": int(get_global_rank()),
            "active": bool(self._calibration_active),
            "layer_num": int(self.layer_num),
            "head_num": int(self.head_num),
            "per_head": bool(self._is_per_head_quant()),
            "counts": list(self.calibration_counts),
            "observed_token_rows": list(self.observed_token_rows),
            "shape": list(self.abs_max.shape),
            "architecture": get_model_architectures(get_env_start_args().model_dir),
            "num_target_layers": int(self.layer_num - get_added_mtp_kv_layer_num()),
            "num_draft_layers": int(get_added_mtp_kv_layer_num()),
        }

    def snapshot_calibration(self) -> dict[str, Any]:
        """Stop collection and return only CPU-owned rank-local data.

        No distributed communication is performed here.  The caller is responsible
        for collecting all rank snapshots and merging them after requests drain.
        """
        self._calibration_active = False
        if self.abs_max.is_cuda:
            torch.cuda.synchronize(self.abs_max.device)
        data = self.calibration_status()
        data.update(
            {
                "qmin": float(self.qmin),
                "qmax": float(self.qmax),
                "abs_max": self.abs_max.detach().cpu().tolist(),
            }
        )
        return data

    def update_calibration_data(self, kv: torch.Tensor, layer_index: int):
        if not self._calibration_active or not get_model_init_status() or kv.numel() == 0:
            return
        if not 0 <= layer_index < self.layer_num:
            raise IndexError(f"calibration layer index {layer_index} is outside [0, {self.layer_num})")
        if self._is_per_head_quant():
            kv_max = kv.abs().amax(dim=(0, 2)).to(torch.float32)
        else:
            k_max = kv[:, : self.head_num, :].abs().amax().to(torch.float32)
            v_max = kv[:, self.head_num :, :].abs().amax().to(torch.float32)
            kv_max = torch.stack((k_max, v_max))
        self.abs_max[layer_index] = torch.maximum(self.abs_max[layer_index], kv_max)
        self.calibration_counts[layer_index] += 1
        self.observed_token_rows[layer_index] += int(kv.shape[0])
