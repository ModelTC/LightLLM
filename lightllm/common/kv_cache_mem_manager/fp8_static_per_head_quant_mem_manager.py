import os
import json
import torch
import torch.distributed as dist
from typing import Tuple, Any
from lightllm.utils.config_utils import get_model_architectures
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_added_mtp_kv_layer_num, get_env_start_args
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

        cfg, all_scales = self._load_and_check_config()
        q_cfg = cfg["q_calibration"]

        all_head_num = cfg["num_head"]
        # A joint target+draft config uses its leading target scale rows on a target-only server.
        all_scales = all_scales[: self.layer_num].to(device="cuda")

        factor = (get_dp_world_size() * head_num) // all_head_num
        all_scales = torch.repeat_interleave(input=all_scales, repeats=factor, dim=-1)
        rank_in_dp = get_current_rank_in_dp()

        v_offset = all_scales.shape[1] // 2
        start_head = rank_in_dp * head_num
        end_head = start_head + head_num
        k_scales = all_scales[:, start_head:end_head].contiguous()
        v_scales = all_scales[:, v_offset + start_head : v_offset + end_head].contiguous()
        self.scales = torch.cat((k_scales, v_scales), dim=-1)

        all_q_scales = torch.tensor(q_cfg["scales"], dtype=torch.float32, device="cuda").view(q_cfg["scales_shape"])[
            : self.layer_num
        ]
        q_factor = (get_dp_world_size() * head_num) // q_cfg["num_head"]
        all_q_scales = torch.repeat_interleave(input=all_q_scales, repeats=q_factor, dim=-1)
        self.q_scales = all_q_scales[:, start_head:end_head].contiguous()
        return

    def _load_and_check_config(self):
        config_path = get_env_start_args().kv_quant_calibration_config_path
        if config_path is None:
            raise ValueError("fp8kv_sph requires kv_quant_calibration_config_path with q_calibration")

        logger.info(f"kv_quant_calibration_config_path {config_path} is set, will load kv quant calibration config")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"kv_quant_calibration_config {config_path} not found")

        with open(config_path, "r") as f:
            cfg = json.load(f)

        num_layers, num_heads = cfg["num_layers"], cfg["num_head"]
        scales = torch.tensor(cfg["scales"], dtype=torch.float32)
        expected_shape = [num_layers, 2 * num_heads]
        if cfg["scales_shape"] != expected_shape or list(scales.shape) != expected_shape:
            raise ValueError(f"calibration scales and scales_shape must match {expected_shape}")

        runtime_draft_layers = get_added_mtp_kv_layer_num()
        runtime_target_layers = self.layer_num - runtime_draft_layers
        if "num_target_layers" in cfg or "num_draft_layers" in cfg:
            target_layers, draft_layers = cfg["num_target_layers"], cfg["num_draft_layers"]
            if (
                draft_layers < 0
                or target_layers + draft_layers != num_layers
                or target_layers != runtime_target_layers
                or (runtime_draft_layers > 0 and draft_layers != runtime_draft_layers)
            ):
                raise ValueError(
                    f"invalid calibration layer layout: target={target_layers}, draft={draft_layers}, "
                    f"runtime target={runtime_target_layers}, draft={runtime_draft_layers}"
                )
        elif num_layers != self.layer_num:
            raise ValueError(f"legacy calibration num_layers {num_layers} do not match runtime {self.layer_num}")

        q_cfg = cfg.get("q_calibration")
        if not q_cfg:
            raise ValueError("fp8kv_sph calibration config requires q_calibration")
        assert (get_dp_world_size() * self.head_num) % num_heads == 0
        assert (get_dp_world_size() * self.head_num) % q_cfg["num_head"] == 0
        return cfg, scales

    def get_att_input_params(self, layer_index: int) -> Tuple[Any, Any]:
        k = self.kv_buffer[layer_index][:, : self.head_num, :]
        v = self.kv_buffer[layer_index][:, self.head_num :, :]
        return k, v
