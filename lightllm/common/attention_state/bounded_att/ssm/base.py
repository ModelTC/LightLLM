from __future__ import annotations

import torch

from lightllm.common.attention_state.commit import StateAcceptance
from lightllm.common.ssm_types import SSMRunConfig

from lightllm.common.basemodel.triton_kernel.linear_att.ssm_acceptance import commit_ssm_prefix
from .export import export_indexed_state


def update_accepted_indices(
    acceptance: StateAcceptance, indices: torch.Tensor, *, replay_cursors: torch.Tensor | None = None
) -> None:
    """把接受长度写入 MTP 槽位索引，并按需推进 Replay cursor。"""

    commit_ssm_prefix(indices, acceptance.req_indices, acceptance.accepted_lengths, replay_cursors)


class SSMExecutor:
    """一种 SSM 模式的 decode、接受和快照适配器。

    算法 State 持有资源和逻辑层到物理层的映射；executor 只处理 decode 递推、接受前缀
    以及 accepted-state 导出。prefill 仍由 attention backend 执行。一次 verify 的
    布局由 ``SSMForward`` 保存，并在 commit 阶段通过 ``StateCommit`` 传回同一个 executor。
    """

    # State 负责资源生命周期和层映射，Executor 只负责本模式的 decode、接受和快照导出。
    # 这样 Native、Replay 共享同一条上层调用链，差异收敛在执行器内部。

    slots_per_request = 1
    _run_config: SSMRunConfig | None = None

    @property
    def run_config(self) -> SSMRunConfig | None:
        """返回最近一次 forward 使用的不可变 kernel 配置。"""

        return self._run_config

    @run_config.setter
    def run_config(self, config: SSMRunConfig | dict[str, int] | None) -> None:
        """保存不可变的 kernel 配置，兼容字典形式的 autotune 结果。"""
        self._run_config = config if config is None or isinstance(config, SSMRunConfig) else SSMRunConfig(**config)

    def clear_history(self, req: int | torch.Tensor) -> None:
        """清空请求的临时 history；默认模式没有额外 history。"""

        return None

    def prepare_decode(self, reqs: torch.Tensor, cu_seqlens: torch.Tensor | None = None) -> None:
        """在 decode 前准备本轮基准；Native 默认不需要准备。"""

        return None

    def accept_single_token(self, reqs: torch.Tensor) -> None:
        """提交非 speculative decode 产生的单 token 更新。"""

        return None

    def merge_accepted_updates(self, reqs: torch.Tensor) -> None:
        """把已接受的临时更新合并回 canonical state。"""

        return None

    def decode_indices(self, reqs: torch.Tensor, speculative: bool) -> torch.Tensor:
        """返回 decode 使用的物理槽位；Native MTP 会展开请求槽位。"""

        return reqs

    def get_accepted_state(self, req_idx: int, accepted_index: int | None = None) -> torch.Tensor:
        """返回一个请求的 accepted-state 视图或物化快照。"""

        return self.state[:, req_idx]

    def export_accepted_state(
        self,
        req_indices: list[int],
        targets: list[torch.Tensor],
        accepted_indices: torch.Tensor | None = None,
        *,
        flush: bool = False,
    ) -> None:
        """把 accepted state 写入调用方提供的目标页，不负责分配目标。"""

        export_indexed_state(self.state, req_indices, targets)

    def create_commit(self, run_config: SSMRunConfig | None) -> object | None:
        """把本次 forward 的配置转换为接受阶段可携带的提交 payload。"""

        return None

    def commit(self, acceptance: StateAcceptance, indices: torch.Tensor, state_name: str) -> None:
        """校验并应用接受结果；供直接调用 State 接口时使用。"""

        validator = getattr(self, "validate_commit", None)
        commit_data = validator(acceptance, state_name) if validator is not None else None
        self.commit_validated(acceptance, indices, state_name, commit_data=commit_data)

    def commit_validated(
        self,
        acceptance: StateAcceptance,
        indices: torch.Tensor,
        state_name: str,
        *,
        commit_data: object | None = None,
    ) -> None:
        """应用已由 State/Manager 校验过的接受结果，不重复检查公共契约。"""

        update_accepted_indices(acceptance, indices)

    def forward(
        self,
        layer: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        a_log: torch.Tensor,
        bias: torch.Tensor,
        reqs: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        run_config: SSMRunConfig | None = None,
        *,
        state_indices: torch.Tensor | None = None,
        accepted_token_counts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, SSMRunConfig | None]:
        """执行一层 decode，并返回输出及本次实际采用的 run config。"""

        raise NotImplementedError
