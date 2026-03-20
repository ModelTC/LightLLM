import os
import json
import torch
import torch.distributed as dist
from lightllm.utils.config_utils import get_model_architectures
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_env_start_args

logger = init_logger(__name__)

from .mem_manager import MemoryManager


class OfflineFP8QuantMemManager(MemoryManager):
    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=False, mem_fraction=0.9):
        # 这里用uint8存储量化后的kv，方便兼容各种torch算子。fp8量化目前采用离线方案，kv_buffer不存储scale
        super().__init__(size, torch.uint8, head_num, head_dim, layer_num, always_copy, mem_fraction)

        self.qmax = torch.finfo(torch.float8_e4m3fn).max
        self.qmin = torch.finfo(torch.float8_e4m3fn).min
        self.total_head_num = head_num * dist.get_world_size() if dist.is_initialized() else head_num
        self.scales = None
        self.scales_list = None

        enable_per_head = self._is_per_head_quant()

        if get_env_start_args().kv_quant_calibration_config_path is not None:
            logger.info(
                f"kv_quant_calibration_config_path {get_env_start_args().kv_quant_calibration_config_path} is set, "
                "will load kv quant calibration config"
            )
            cfg = self._load_and_check_config()

            self.scales_list = cfg["scales"]
            self.scales = torch.tensor(self.scales_list, dtype=torch.float32, device="cuda").view(cfg["scales_shape"])
            if not enable_per_head:
                self.scales = torch.repeat_interleave(self.scales, head_num, dim=-1)
            elif cfg["num_head"] > self.total_head_num:
                factor = cfg["num_head"] // self.total_head_num
                self.scales = self.scales[..., ::factor].contiguous()
            elif cfg["num_head"] < self.total_head_num:
                factor = self.total_head_num // cfg["num_head"]
                self.scales = torch.repeat_interleave(self.scales, factor, dim=-1).contiguous()
            if enable_per_head and dist.is_initialized() and dist.get_world_size() > 1:
                v_offset = self.total_head_num
                start_head = dist.get_rank() * head_num
                end_head = start_head + head_num
                k_scales = self.scales[:, start_head:end_head].contiguous()
                v_scales = self.scales[:, v_offset + start_head : v_offset + end_head].contiguous()
                current_scales = torch.cat((k_scales, v_scales), dim=-1)

                self.scales_list = current_scales.tolist()
                self.scales = current_scales
        else:
            logger.warning("scales is None, no kv_quant_calibration_config_path be set, will use 1.0 as scales")

    @staticmethod
    def _is_per_head_quant():
        """Only fa3 backend supports per-head FP8 KV quantization.
        FlashInfer only accepts scalar (per-tensor) k_scale/v_scale."""
        args = get_env_start_args()
        return "fa3" in args.llm_prefill_att_backend

    def _load_and_check_config(self):
        enable_per_head = self._is_per_head_quant()

        if os.path.exists(get_env_start_args().kv_quant_calibration_config_path):
            with open(get_env_start_args().kv_quant_calibration_config_path, "r") as f:
                cfg = json.load(f)

            if cfg["qmin"] != self.qmin:
                raise ValueError(f"qmin {cfg['qmin']} in config not match torch.float8_e4m3fn.min {self.qmin}")
            if cfg["qmax"] != self.qmax:
                raise ValueError(f"qmax {cfg['qmax']} in config not match torch.float8_e4m3fn.max {self.qmax}")
            model_arch = get_model_architectures(get_env_start_args().model_dir)
            if cfg["architectures"] != model_arch:
                raise ValueError(
                    f"architectures {cfg['architectures']} in config " f"not match current model_arch {model_arch}"
                )
            if cfg["num_layers"] != self.layer_num:
                raise ValueError(
                    f"num_layers {cfg['num_layers']} in config " f"not match current layer_num {self.layer_num}"
                )
            if cfg["num_head"] % self.total_head_num != 0 and self.total_head_num % cfg["num_head"] != 0:
                raise ValueError(
                    f"num_head {cfg['num_head']} in config " f"not match current model head num {self.total_head_num}"
                )
            if enable_per_head:
                if cfg["quant_type"] != "per_head":
                    raise ValueError(f"quant type {cfg['quant_type']} in config not match per-head backend")
            else:
                if cfg["quant_type"] != "per_tensor":
                    raise ValueError(f"quant type {cfg['quant_type']} in config not match per-tensor backend")

            return cfg
        else:
            raise FileNotFoundError(
                f"kv_quant_calibration_config {get_env_start_args().kv_quant_calibration_config_path} not found"
            )
