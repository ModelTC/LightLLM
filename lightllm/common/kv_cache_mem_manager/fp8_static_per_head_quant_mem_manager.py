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

        if get_env_start_args().kv_quant_calibration_config_path is not None:
            logger.info(
                f"kv_quant_calibration_config_path {get_env_start_args().kv_quant_calibration_config_path} is set, "
                "will load kv quant calibration config"
            )
            cfg = self._load_and_check_config()
            all_head_num = cfg["num_head"]
            all_scales = torch.tensor(cfg["scales"], dtype=torch.float32, device="cuda").view(cfg["scales_shape"])
            # A joint target+draft config is deliberately usable by a
            # target-only server. Its metadata has already verified that the
            # leading rows are exactly this target's packed KV layers.
            all_scales = all_scales[: self.layer_num]

            factor = (get_dp_world_size() * head_num) // all_head_num
            if (get_dp_world_size() * head_num) % all_head_num != 0:
                raise ValueError(
                    f"global KV heads {get_dp_world_size() * head_num} are not divisible by "
                    f"calibration config num_head {all_head_num}"
                )
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
            if cfg["quant_type"] != "per_head":
                raise ValueError(f"quant type {cfg['quant_type']} in config not match per-head backend")

            self._validate_config_layout_and_scales(cfg)
            return cfg
        else:
            raise FileNotFoundError(
                f"kv_quant_calibration_config {get_env_start_args().kv_quant_calibration_config_path} not found"
            )

    def _validate_config_layout_and_scales(self, cfg):
        """Validate a complete config before moving calibration scales to CUDA."""
        def require_integer(name):
            value = cfg[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"calibration config {name} must be an integer, got {value!r}")
            return value

        try:
            config_layer_num = require_integer("num_layers")
            config_head_num = require_integer("num_head")
            scales_shape = list(cfg["scales_shape"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid FP8 KV calibration config layer/head/shape metadata") from exc

        if any(isinstance(value, bool) or not isinstance(value, int) for value in scales_shape):
            raise ValueError(f"calibration config scales_shape must contain integers, got {scales_shape!r}")

        if config_layer_num <= 0 or config_head_num <= 0:
            raise ValueError(
                f"calibration config requires positive num_layers and num_head, got "
                f"{config_layer_num} and {config_head_num}"
            )
        expected_shape = [config_layer_num, 2 * config_head_num]
        if scales_shape != expected_shape:
            raise ValueError(
                f"scales_shape {scales_shape} in config does not match expected {expected_shape}"
            )

        try:
            scales = torch.tensor(cfg["scales"], dtype=torch.float32)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("calibration config scales must be a numeric two-dimensional array") from exc
        if list(scales.shape) != expected_shape:
            raise ValueError(
                f"scales tensor shape {list(scales.shape)} in config does not match {expected_shape}"
            )
        if not torch.isfinite(scales).all():
            raise ValueError("calibration config scales must all be finite")
        if not (scales > 0).all():
            raise ValueError("calibration config scales must all be positive")

        runtime_draft_layers = get_added_mtp_kv_layer_num()
        runtime_target_layers = self.layer_num - runtime_draft_layers
        if runtime_target_layers <= 0:
            raise ValueError(
                f"runtime packed KV layers={self.layer_num} are inconsistent with "
                f"draft layers={runtime_draft_layers}"
            )

        has_target_metadata = "num_target_layers" in cfg
        has_draft_metadata = "num_draft_layers" in cfg
        if has_target_metadata != has_draft_metadata:
            raise ValueError("joint calibration config must declare both num_target_layers and num_draft_layers")
        if has_target_metadata:
            try:
                config_target_layers = require_integer("num_target_layers")
                config_draft_layers = require_integer("num_draft_layers")
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("joint calibration config target/draft layer metadata must be integers") from exc
            if config_target_layers <= 0 or config_draft_layers < 0:
                raise ValueError(
                    f"invalid joint calibration layer metadata: target={config_target_layers}, "
                    f"draft={config_draft_layers}"
                )
            if config_target_layers + config_draft_layers != config_layer_num:
                raise ValueError(
                    f"joint calibration metadata target+draft={config_target_layers + config_draft_layers} "
                    f"does not equal num_layers={config_layer_num}"
                )
            if config_target_layers != runtime_target_layers:
                raise ValueError(
                    f"joint calibration target layers={config_target_layers} do not match runtime "
                    f"target layers={runtime_target_layers}"
                )
            if runtime_draft_layers > 0 and config_draft_layers != runtime_draft_layers:
                raise ValueError(
                    f"joint calibration draft layers={config_draft_layers} do not match runtime "
                    f"draft layers={runtime_draft_layers}"
                )
            return

        # Legacy files describe one contiguous KV layout and remain compatible
        # whenever their total packed-layer count exactly matches the runtime.
        if config_layer_num != self.layer_num:
            raise ValueError(
                f"legacy calibration num_layers={config_layer_num} does not match runtime "
                f"layer_num={self.layer_num} (target={runtime_target_layers}, draft={runtime_draft_layers}); "
                "joint configs require target/draft metadata when the total differs"
            )

    def get_att_input_params(self, layer_index: int) -> Tuple[Any, Any]:
        k = self.kv_buffer[layer_index][:, : self.head_num, :]
        v = self.kv_buffer[layer_index][:, self.head_num :, :]
        return k, v
