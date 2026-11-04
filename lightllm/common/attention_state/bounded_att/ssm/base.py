from lightllm.common.ssm_types import SSMRunConfig

from lightllm.common.basemodel.triton_kernel.linear_att.ssm_acceptance import commit_ssm_prefix
from .export import export_indexed_state


def update_accepted_indices(acceptance, indices, *, replay_cursors=None):
    commit_ssm_prefix(indices, acceptance.req_indices, acceptance.accepted_lengths, replay_cursors)


class SSMExecutor:
    """Decode execution and storage for one SSM mode; prefill stays in the attention backend."""

    slots_per_request = 1
    _run_config: SSMRunConfig | None = None

    @property
    def run_config(self) -> SSMRunConfig | None:
        return self._run_config

    @run_config.setter
    def run_config(self, config: SSMRunConfig | dict | None):
        self._run_config = config if config is None or isinstance(config, SSMRunConfig) else SSMRunConfig(**config)

    def clear_history(self, req):
        pass

    def prepare_decode(self, reqs, cu_seqlens=None):
        pass

    def accept_single_token(self, reqs):
        pass

    def merge_accepted_updates(self, reqs):
        pass

    def decode_indices(self, reqs, speculative):
        return reqs

    def get_accepted_state(self, req_idx, accepted_index=None):
        return self.state[:, req_idx]

    def export_accepted_state(self, req_indices, targets, accepted_indices=None):
        export_indexed_state(self.state, req_indices, targets)

    def create_commit(self, run_config):
        return None

    def validate_commit(self, acceptance, state_name):
        pass

    def commit(self, acceptance, indices, state_name):
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
        raise NotImplementedError
