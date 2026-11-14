from dataclasses import dataclass
from typing import NamedTuple

import torch

from .spec import AttentionPhase


@dataclass(frozen=True, kw_only=True)
class ServingConstraints:
    """启动时确定的执行约束，与 LayerSpec 的计算语义分开保存。"""

    graph_phases: frozenset[AttentionPhase] = frozenset()
    microbatch_count: int = 1
    max_batch_size: int = 1
    max_seq_len: int = 1
    expert_parallel: bool = False
    transfer_paths: frozenset[str] = frozenset()


class WorkspaceRequirement(NamedTuple):
    name: str
    elements: int
    dtype: torch.dtype


@dataclass(frozen=True)
class WorkspaceBudget:
    """捕获前的资源上界；workspace 项为 (名称, 元素数, dtype)。

    同一执行槽位按名称取最大容量，并发槽位分别分配。主机长度快照只在需要
    它的实现中创建；持久存储的初始化由 State 执行，不能在 forward 中反复清零。
    """

    workspace: tuple[WorkspaceRequirement, ...] = ()
