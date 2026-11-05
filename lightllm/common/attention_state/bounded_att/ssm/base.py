from lightllm.common.ssm_types import SSMRunConfig

from lightllm.common.basemodel.triton_kernel.linear_att.ssm_acceptance import commit_ssm_prefix
from .export import export_indexed_state


def update_accepted_indices(acceptance, indices, *, replay_cursors=None):
    commit_ssm_prefix(indices, acceptance.req_indices, acceptance.accepted_lengths, replay_cursors)


class SSMExecutor:
    """一种 SSM 模式的 decode、接受和快照适配器。

    ``ConvSSMState`` 持有资源和逻辑层到物理层的映射；executor 只处理
    decode 递推、接受前缀以及 accepted-state 导出。prefill 仍由 attention
    backend 执行。一次 verify 的布局由 ``SSMForward`` 保存，并在 commit
    阶段通过 ``StateCommit`` 传回同一个 executor。
    """

    # State 负责资源生命周期和层映射，Executor 只负责本模式的 decode、接受和快照导出。
    # 这样 Native、Compact、Replay 共享同一条上层调用链，差异收敛在执行器内部。

    slots_per_request = 1
    _run_config: SSMRunConfig | None = None

    @property
    def run_config(self) -> SSMRunConfig | None:
        """返回最近一次 forward 使用的不可变 kernel 配置。"""

        return self._run_config

    @run_config.setter
    def run_config(self, config: SSMRunConfig | dict | None):
        self._run_config = config if config is None or isinstance(config, SSMRunConfig) else SSMRunConfig(**config)

    def clear_history(self, req):
        """清空请求的临时 history；默认模式没有额外 history。"""

        return None

    def prepare_decode(self, reqs, cu_seqlens=None):
        """在 decode 前准备本轮基准；Native/Compact 默认不需要准备。"""

        return None

    def accept_single_token(self, reqs):
        """提交非 speculative decode 产生的单 token 更新。"""

        return None

    def merge_accepted_updates(self, reqs):
        """把已接受的临时更新合并回 canonical state。"""

        return None

    def decode_indices(self, reqs, speculative):
        """返回 decode 使用的物理槽位；Native MTP 会展开请求槽位。"""

        return reqs

    def get_accepted_state(self, req_idx, accepted_index=None):
        """返回一个请求的 accepted-state 视图或物化快照。"""

        return self.state[:, req_idx]

    def export_accepted_state(self, req_indices, targets, accepted_indices=None):
        """把 accepted state 写入调用方提供的目标页，不负责分配目标。"""

        export_indexed_state(self.state, req_indices, targets)

    def create_commit(self, run_config):
        """把本次 forward 的配置转换为接受阶段可携带的提交 payload。"""

        return None

    def validate_commit(self, acceptance, state_name):
        """校验接受结果是否包含本模式所需的 forward 元数据。"""

        return None

    def commit(self, acceptance, indices, state_name):
        """将接受结果应用到状态索引；具体合并逻辑由子类实现。"""

        update_accepted_indices(acceptance, indices)

    def forward(
        self,
        layer,
        q,
        k,
        v,
        a,
        b,
        a_log,
        bias,
        reqs,
        cu_seqlens=None,
        run_config=None,
        *,
        state_indices=None,
        accepted_token_counts=None,
    ):
        """执行一层 decode，并返回输出及本次实际采用的 run config。"""

        raise NotImplementedError
