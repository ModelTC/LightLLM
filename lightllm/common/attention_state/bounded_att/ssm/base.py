from lightllm.common.basemodel.triton_kernel.mtp_utils import linear_att_mtp_state_index_update


def update_accepted_indices(reqs, accepted_steps, indices, context, *, replay_cursors=None):
    if context is None:
        indices[reqs] = accepted_steps - 1
    else:
        linear_att_mtp_state_index_update(
            req_to_mtp_state_index=indices,
            b_req_mtp_start_loc=context.b_req_mtp_start_loc,
            b_req_idx=context.b_req_idx,
            b_mtp_index=context.b_mtp_index,
            accepted_index=context.accepted_index,
            verify_width=context.verify_width,
            replay_cursors=replay_cursors,
        )


class SSMState:
    slots_per_request = 1

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

    def create_commit(self, run_config):
        return None

    def validate_commit(self, context, state_name):
        pass

    def commit(self, reqs, accepted_steps, indices, context, state_name):
        update_accepted_indices(reqs, accepted_steps, indices, context)

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
