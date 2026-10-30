from __future__ import annotations

from typing import TYPE_CHECKING
import torch
import dataclasses

from lightllm.utils.envs_utils import get_added_mtp_kv_layer_num, get_env_start_args
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
    # full-attention 布局使用的 dtype；普通 KV 为浮点精度，静态 FP8 为编码容器 uint8。
    # GLM 还用此浮点精度保存原始 indexer tail，具体量化字段由各 State 定义。
    full_att_dtype: torch.dtype
    # 当前 TP rank 实际保存的 KV head 数；少于 TP 数时按组复制，MLA 为每个 rank 保存 1。
    full_att_num_kv_heads: int
    # 普通 GQA 中为单个 K/V head 的维度；GLM packed MLA 中为包含 indexer/padding 的行元素数。
    full_att_head_dim: int

    # TP 切分前的线性 attention K head 数；Q 使用相同的 head 数和维度。
    global_linear_k_heads: int
    # TP 切分前的 V head 数；GDN 可与 K 不同，GLM KDA 与 K 相同。
    global_linear_v_heads: int
    # 当前 TP rank 的线性 K/Q head 数；Conv 的 Q/K 通道宽度由此与 head_linear_k_dim 相乘。
    num_linear_k_heads: int
    # 当前 TP rank 的线性 V head 数，也是每层 SSM 矩阵的 head 轴长度。
    num_linear_v_heads: int
    # 每个 Q/K head 的特征维度，对应 SSM 的 key 轴，不随 TP 切分。
    head_linear_k_dim: int
    # 每个 V head 的特征维度，对应 SSM 的 value 轴；KDA 中与 key 维度相等。
    head_linear_v_dim: int
    # 原始卷积核宽度，单位为 token；基准 Conv 历史保存 kernel_size - 1 个位置。
    # MTP 的额外候选位置由 State.mtp_step 决定，不计入本字段。
    conv_kernel_size: int
    # 主模型中线性 attention 层的数量，不含 full-attention 层或追加的 draft 层。
    linear_layer_num: int
    # Conv 运行态和大小页快照的元素 dtype，不代表算子的内部累加精度。
    conv_state_dtype: torch.dtype
    # SSM 运行态和大小页快照的元素 dtype，可独立于 Conv 精度配置。
    ssm_state_dtype: torch.dtype
    # 混合层排列周期：每隔这么多层放一个 full-attention 层，零起始层号为 interval - 1。
    full_attention_interval: int
    # 主模型总层数，包含线性与 full-attention 两类层，不包含追加的 draft 层。
    all_layer_num: int
    # 需要追加到 full-attention 层布局中的 draft KV 层数；不包含 draft 的布局取 0。
    # 用于合并布局的层号/物理行换算，不代表 draft 请求数或 MTP 候选 token 数。
    draft_full_att_kv_layer_num: int = 0

    def get_conv_dim(self):
        # 第一项对应q的参数，第二项对应k的参数，第三项对应v的参数
        # 由于 k_dim = q_dim, k_heads = q_heads, 所以第一项和第二项的计算
        # 形式相同，但是实际内在含义是不同的。
        return (
            self.head_linear_k_dim * self.num_linear_k_heads
            + self.head_linear_k_dim * self.num_linear_k_heads
            + self.head_linear_v_dim * self.num_linear_v_heads
        )

    def get_main_model_full_att_layer_num(self):
        full_att_layer_num = self.all_layer_num - self.linear_layer_num
        assert full_att_layer_num == self.all_layer_num // self.full_attention_interval
        return full_att_layer_num

    def get_full_att_kv_layer_num_with_draft_model(self):
        return self.get_main_model_full_att_layer_num() + self.draft_full_att_kv_layer_num

    def get_full_att_logical_layer_ids(self):
        """Logical model layers stored in each physical full-attention row."""
        return tuple(range(self.full_attention_interval - 1, self.all_layer_num, self.full_attention_interval)) + tuple(
            range(self.all_layer_num, self.all_layer_num + self.draft_full_att_kv_layer_num)
        )

    def get_full_att_kv_layer_index(self, layer_index: int) -> int:
        """Map a global target/draft layer index to the packed full-attention cache."""

        layer_index = int(layer_index)
        if layer_index >= self.all_layer_num:
            kv_layer_index = layer_index - self.linear_layer_num
        else:
            kv_layer_index = layer_index // self.full_attention_interval
        assert 0 <= kv_layer_index < self.get_full_att_kv_layer_num_with_draft_model(), (
            f"layer {layer_index} maps outside the packed full-attention KV cache: "
            f"slot={kv_layer_index}, slots={self.get_full_att_kv_layer_num_with_draft_model()}"
        )
        return kv_layer_index

    def get_conv_state_shape(self):
        # Base committed sliding-window state, without speculative MTP tail.
        return (self.get_conv_dim(), self.conv_kernel_size - 1)

    def get_mtp_conv_state_shape(self, mtp_step: int):
        # Working state with room for S speculative tokens before acceptance.
        return (self.get_conv_dim(), (self.conv_kernel_size - 1) + mtp_step)

    def get_ssm_state_shape(self):
        return (self.num_linear_v_heads, self.head_linear_k_dim, self.head_linear_v_dim)

    def get_conv_state_bytes_per_layer(self):
        return self.get_conv_dim() * (self.conv_kernel_size - 1) * self.conv_state_dtype.itemsize

    def get_ssm_state_bytes_per_layer(self):
        return self.num_linear_v_heads * self.head_linear_k_dim * self.head_linear_v_dim * self.ssm_state_dtype.itemsize

    @classmethod
    def load_from_args(cls) -> "Qwen3NextResourceConfig":
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
        *,
        include_draft: bool = True,
    ):
        llm_config = model_cfg.get("text_config", model_cfg)
        n_layer = llm_config["num_hidden_layers"]

        tp_world_size = args.tp // args.dp
        full_att_dtype = (
            torch.uint8 if args.llm_kv_type in {"fp8kv_sph", "fp8kv_spt"} else get_torch_dtype(args.data_type)
        )
        return cls(
            tp_world_size=tp_world_size,
            full_att_all_num_kv_heads=llm_config["num_key_value_heads"],
            full_att_dtype=full_att_dtype,
            full_att_num_kv_heads=max(1, llm_config["num_key_value_heads"] // tp_world_size),
            full_att_head_dim=llm_config["head_dim"],
            global_linear_k_heads=llm_config["linear_num_key_heads"],
            global_linear_v_heads=llm_config["linear_num_value_heads"],
            num_linear_k_heads=max(1, llm_config["linear_num_key_heads"] // tp_world_size),
            num_linear_v_heads=max(1, llm_config["linear_num_value_heads"] // tp_world_size),
            head_linear_k_dim=llm_config["linear_key_head_dim"],
            head_linear_v_dim=llm_config["linear_value_head_dim"],
            conv_kernel_size=llm_config["linear_conv_kernel_dim"],
            linear_layer_num=n_layer - (n_layer // llm_config["full_attention_interval"]),
            conv_state_dtype=get_torch_dtype(args.data_type),
            ssm_state_dtype=get_torch_dtype(args.linear_att_ssm_data_type),
            full_attention_interval=llm_config["full_attention_interval"],
            all_layer_num=n_layer,
            draft_full_att_kv_layer_num=get_added_mtp_kv_layer_num() if include_draft else 0,
        )
