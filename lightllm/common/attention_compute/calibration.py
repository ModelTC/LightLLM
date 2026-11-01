from __future__ import annotations
from typing import TYPE_CHECKING
from lightllm.common.attention_state.utils import MODEL_ROLES
from lightllm.utils.log_utils import init_logger

if TYPE_CHECKING:
    from easydict import EasyDict
    from lightllm.common.attention_state import AttentionStateManager
    from lightllm.server.core.objs.start_args_type import StartArgs

logger = init_logger(__name__)


def load_full_att_calibration(manager: AttentionStateManager, args: StartArgs | EasyDict) -> None:
    """全部 main/draft State 就绪后统一读取静态标定，并交给具体 State 绑定。

    文件沿用物理行顺序：main、draft0..draft35；同角色有多个 State 时按最小逻辑层号
    排序，State 内保留 logical_layer_groups 顺序。非静态 FP8 的 full State 同样占用
    对应物理行区间，共享逻辑层只占一行；CPU 布局创建不需要调用本流程。
    """
    import json
    from lightllm.common.attention_state.full_att.gqa.static_fp8_state import StaticFP8FullAttentionState

    full_states = manager.get_full_attention_states()
    calibration_states = [state for state in full_states if isinstance(state, StaticFP8FullAttentionState)]
    if not calibration_states:
        return

    # 阶段一：只读取一次文件；无文件时由 per_tensor State 绑定单位 scale。
    calibration = None
    path = args.kv_quant_calibration_config_path
    if path is not None:
        with open(path) as file:
            calibration = json.load(file)
        if any(state.config.scale_mode == "per_tensor" for state in calibration_states):
            from lightllm.utils.config_utils import get_model_architectures

            assert calibration["architectures"] == get_model_architectures(
                args.model_dir
            ), "calibration architecture does not match the model"

    # 阶段二：显式使用模型角色顺序，避免注册先后及 draft10 的字符串排序影响标定行。
    storage_layer_start = 0
    for role in MODEL_ROLES:
        states = sorted(
            manager.get_full_attention_states(role),
            key=lambda state: (
                min(layer for group in state.config.logical_layer_groups for layer in group),
                state.state_name,
            ),
        )
        for state in states:
            if isinstance(state, StaticFP8FullAttentionState):
                state.bind_calibration(
                    calibration=calibration,
                    storage_layer_start=storage_layer_start,
                )
            storage_layer_start += len(state.config.logical_layer_groups)
    logger.info(
        "Full-attention calibration bound: states=%d, source=%s", len(calibration_states), path or "unit scales"
    )
