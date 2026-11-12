"""模型层与 attention 实现之间的契约。

初始化时，LayerSpec / CacheLayout 用于选择后端和划分可共享 plan 的层组，
LayerAttention 保存本层参数和共享 backend，通过 forward(plan, inputs) 执行。
运行时，AttentionBatch 描述整个批次，Inputs 提供当前层的 tensor 输入；
backend 专用 plan 保存派生 metadata 和 provider wrapper，由 InferStateInfo 持有。
KV / Conv / SSM 的分配、回收和迁移仍由 AttentionStateManager 管理。
无持久 KV 的视觉/音频编码器使用 VarlenAttentionBackend，直接传入 Q/K/V 和序列边界。

初始化集成见 common.basemodel.attention，运行时入口见 InferStateInfo.prepare_attention。
"""

from .backend import AttentionBackend, LayerAttention, UnsupportedAttention, VarlenAttentionBackend
from .batch import AttentionBatch, PlanCapacity
from .execution import BackendResources, ExecutionRequirements
from .graph import GraphBinding
from .inputs import VarlenAttentionInputs
from .spec import AttentionFamily, AttentionPhase, CacheLayout, LayerSpec
