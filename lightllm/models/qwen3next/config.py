from __future__ import annotations

from typing import TYPE_CHECKING
import torch
import dataclasses

from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.torch_dtype_utils import get_torch_dtype

if TYPE_CHECKING:
    from easydict import EasyDict
    from lightllm.server.core.objs.start_args_type import StartArgs


@dataclasses.dataclass
class Qwen3NextResourceConfig:
    # 单个模型的 TP 组大小，通常为 args.tp // args.dp，不包含其他 DP 组。
    tp_world_size: int

    # TP 切分前的全局 KV head 数，K/V 各有这么多个，不相加；GLM 的 MLA latent 记为 1。
    full_att_all_num_kv_heads: int
    # 未打包的 KV 特征维度：GQA 为单个 K/V head 的宽度，GLM MLA 为 latent 宽度。
    # 量化、indexer 字段及行 padding 由具体 full-attention State 确定，不计入本字段。
    full_att_head_dim: int

    # TP 切分前的线性 attention K head 数；Q 使用相同的 head 数和维度。
    global_linear_k_heads: int
    # TP 切分前的 V head 数；GDN 可与 K 不同，GLM KDA 与 K 相同。
    global_linear_v_heads: int
    # 每个 Q/K head 的特征维度，对应 SSM 的 key 轴，不随 TP 切分。
    head_linear_k_dim: int
    # 每个 V head 的特征维度，对应 SSM 的 value 轴；KDA 中与 key 维度相等。
    head_linear_v_dim: int
    # 原始卷积核宽度，单位为 token；基准 Conv 历史保存 kernel_size - 1 个位置。
    # MTP 的额外候选位置由 State.mtp_step 决定，不计入本字段。
    conv_kernel_size: int
    # Conv 运行态和大小页快照的元素 dtype，不代表算子的内部累加精度。
    conv_state_dtype: torch.dtype
    # SSM 运行态和大小页快照的元素 dtype，可独立于 Conv 精度配置。
    ssm_state_dtype: torch.dtype
    # 混合层排列周期：每隔这么多层放一个 full-attention 层，零起始层号为 interval - 1。
    full_attention_interval: int
    # 当前模型总层数，包含线性与 full-attention 两类层，不合并其他模型的层。
    all_layer_num: int
    # 显式执行层类型，支持 Loop 等非周期排列；为空时沿用周期配置。
    layer_types: tuple[str, ...] = ()

    def get_full_att_layer_indices(self) -> tuple[int, ...]:
        """返回当前模型内部的 full-attention 层序号，不包含其他模型的层。

        调用方通过 model_logical_layer_ids[index] 转换为实际逻辑层 ID；
        物理存储层映射由 State 管理，draft 使用自身配置独立创建 State。
        """
        if self.layer_types:
            return tuple(index for index, kind in enumerate(self.layer_types) if kind != "linear_attention")
        return tuple(range(self.full_attention_interval - 1, self.all_layer_num, self.full_attention_interval))

    @classmethod
    def load_from_args(cls) -> "Qwen3NextResourceConfig":
        """从启动参数加载模型配置，供 attention 后端校验使用。"""
        args = get_env_start_args()
        model_path = args.model_dir
        from transformers.configuration_utils import PretrainedConfig

        model_cfg, _ = PretrainedConfig.get_config_dict(model_path)
        return cls.from_model_config(model_cfg, args)

    @classmethod
    def from_model_config(
        cls,
        model_cfg: dict[str, object],
        args: StartArgs | EasyDict,
    ) -> "Qwen3NextResourceConfig":
        """解析当前模型的全局几何参数，TP 分片与实际存储由 State 确定。"""
        llm_config = model_cfg.get("text_config", model_cfg)
        n_layer = llm_config["num_hidden_layers"]

        tp_world_size = args.tp // args.dp
        return cls(
            tp_world_size=tp_world_size,
            full_att_all_num_kv_heads=llm_config["num_key_value_heads"],
            full_att_head_dim=llm_config["head_dim"],
            global_linear_k_heads=llm_config["linear_num_key_heads"],
            global_linear_v_heads=llm_config["linear_num_value_heads"],
            head_linear_k_dim=llm_config["linear_key_head_dim"],
            head_linear_v_dim=llm_config["linear_value_head_dim"],
            conv_kernel_size=llm_config["linear_conv_kernel_dim"],
            conv_state_dtype=get_torch_dtype(args.data_type),
            ssm_state_dtype=get_torch_dtype(args.linear_att_ssm_data_type),
            full_attention_interval=llm_config["full_attention_interval"],
            all_layer_num=n_layer,
            layer_types=tuple(llm_config.get("layer_types", ())),
        )
