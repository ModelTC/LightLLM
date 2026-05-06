import os
import json
import torch
import torch.distributed as dist
from typing import Tuple, Any
from lightllm.utils.config_utils import get_model_architectures
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.dist_utils import get_dp_world_size, get_current_rank_in_dp
from .mem_manager import MemoryManager
from .operator import FP8StaticPerHeadQuantMemOperator

logger = init_logger(__name__)


class FP8StaticPerHeadQuantMemManager(MemoryManager):

    operator_class = FP8StaticPerHeadQuantMemOperator

    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=False, mem_fraction=0.9):
        # 这里用uint8存储量化后的kv，方便兼容各种torch算子。fp8量化目前采用离线方案，kv_buffer不存储scale
        super().__init__(size, torch.uint8, head_num, head_dim, layer_num, always_copy, mem_fraction)

        self.qmax = torch.finfo(torch.float8_e4m3fn).max
        self.qmin = torch.finfo(torch.float8_e4m3fn).min
        self.scales = None

        if get_env_start_args().kv_quant_calibration_config_path is not None:
            logger.info(
                f"kv_quant_calibration_config_path {get_env_start_args().kv_quant_calibration_config_path} is set, "
                "will load kv quant calibration config"
            )
            cfg = self._load_and_check_config()
            all_head_num = cfg["num_head"]
            all_scales = torch.tensor(cfg["scales"], dtype=torch.float32, device="cuda").view(cfg["scales_shape"])

            factor = (get_dp_world_size() * head_num) // all_head_num
            assert (get_dp_world_size() * head_num) % all_head_num == 0
            all_scales = torch.repeat_interleave(input=all_scales, repeats=factor, dim=-1)
            rank_in_dp = get_current_rank_in_dp()

            v_offset = all_scales.shape[1] // 2
            start_head = rank_in_dp * head_num
            end_head = start_head + head_num
            k_scales = all_scales[:, start_head:end_head].contiguous()
            v_scales = all_scales[:, v_offset + start_head : v_offset + end_head].contiguous()
            self.scales = torch.cat((k_scales, v_scales), dim=-1)
        else:
            self.scales = torch.ones((self.kv_buffer.shape[0], 2 * head_num), dtype=torch.float32, device="cuda")
        return

    def _load_and_check_config(self):
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
            assert (
                cfg["quant_type"] == "per_head"
            ), f"quant type {cfg['quant_type']} in config not match per-head backend"
            return cfg
        else:
            raise FileNotFoundError(
                f"kv_quant_calibration_config {get_env_start_args().kv_quant_calibration_config_path} not found"
            )

    def get_att_input_params(self, layer_index: int) -> Tuple[Any, Any]:
        k = self.kv_buffer[layer_index][:, : self.head_num, :]
        v = self.kv_buffer[layer_index][:, self.head_num :, :]
        return k, v
