"""CPU cache 服务布局构建；仅依赖 State 配置，不加载静态标定。"""

from __future__ import annotations

from typing import TYPE_CHECKING
from lightllm.common.att_resources import AttResourceManager
from lightllm.utils.envs_utils import get_env_start_args

if TYPE_CHECKING:
    from collections.abc import Mapping
    from lightllm.common.att_resources.types import StateTransferLayout


def build_cpu_cache_layouts() -> Mapping[str, StateTransferLayout]:
    """使用启动流程相同的 meta State 定义，由管理器统一生成 CPU 服务页布局。"""
    from transformers.configuration_utils import PretrainedConfig
    from lightllm.models import get_draft_model_class
    from lightllm.common.att_resources.utils import MAX_DRAFT_MODELS
    from lightllm.common.basemodel.basemodel_utils import get_resource_config
    from lightllm.utils.llm_utils import get_llm_model_class

    # 阶段一：读取 main 和启用 attention 的 draft 配置，不构造实际模型。
    args = get_env_start_args()
    main_config, _ = PretrainedConfig.get_config_dict(args.model_dir)
    models = [("main", get_llm_model_class(), main_config)]
    if args.mtp_mode is not None and args.mtp_mode not in ("vanilla_no_att", "eagle_no_att"):
        count = args.mtp_step if args.mtp_mode == "vanilla_with_att" else 1
        assert args.mtp_draft_model_dir is not None and len(args.mtp_draft_model_dir) >= count
        assert 0 < count <= MAX_DRAFT_MODELS, "at most 36 draft models are supported"
        for index, directory in enumerate(args.mtp_draft_model_dir[:count]):
            config, _ = PretrainedConfig.get_config_dict(directory)
            models.append((f"draft{index}", get_draft_model_class(config, args.mtp_mode), config))

    # 阶段二：复用模型定义入口创建 meta State，并累计逻辑层号；CPU 页不包含静态标定参数。
    manager = AttResourceManager(request_capacity=0)  # 只描述 meta 布局，不分配请求或缓存存储。
    logical_layer_start = 0
    for role, model_class, config in models:
        count = get_resource_config(config, model_class=model_class)["num_hidden_layers"]
        logical_layer_ids = tuple(range(logical_layer_start, logical_layer_start + count))
        full_states = model_class._create_full_att_states(
            model_config=config, args=args, model_logical_layer_ids=logical_layer_ids,
        )
        bounded_states = model_class._create_bounded_att_states(
            model_config=config,
            args=args,
            model_logical_layer_ids=logical_layer_ids,
            request_capacity=1,
            device="meta",
        )
        # tail 不占 CPU 页；无需创建 tail，也不需要实际模型或临时 CUDA 运行态。
        for state in (*full_states, *bounded_states):
            manager.set_att_state(role, state)
        logical_layer_start += count

    # 阶段三：确认混合模型的页边界一致，由管理器聚合 CPU 联合页布局。
    manager.finish_state_creation()
    if manager.get_bounded_attention_states():
        assert (
            args.cpu_cache_token_page_size == args.linear_att_hash_page_size * args.linear_att_page_block_num
        ), "hybrid CPU pages must match the large state-page token boundary"
    return manager.get_cpu_transfer_layouts()
