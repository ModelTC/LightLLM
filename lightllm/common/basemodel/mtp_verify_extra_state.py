import torch

from lightllm.utils.envs_utils import get_env_start_args


def init_mtp_verify_extra_state(self, model):
    self.b_att_seq_len = self.b_seq_len
    mtp_step = get_env_start_args().mtp_step
    self.b_buffer_idx = self.b_req_idx * (mtp_step + 1) + self.b_mtp_index
    self.b_conv_buffer_idx = self.b_req_idx
    self.is_mtp_verify = (mtp_step > 0) and (not self.is_prefill) and (self.b_mtp_index is not None)
    self.b_gdn_verify_cu_seqlens = None
    self.b_ssm_index_rows = None
    if self.is_mtp_verify:
        step = mtp_step + 1
        n_real = self.b_req_idx.shape[0] // step
        self.b_gdn_verify_cu_seqlens = torch.arange(
            0, (n_real + 1) * step, step, dtype=torch.int32, device=self.b_req_idx.device
        )
        req_first = self.b_req_idx.view(n_real, step)[:, 0]
        base = (req_first * step).view(n_real, 1)
        self.b_ssm_index_rows = base + torch.arange(step, device=base.device, dtype=base.dtype).view(1, step)
        assert self.b_ssm_index_rows.shape == (n_real, step)
        self.b_conv_buffer_idx = req_first
        self.b_num_accepted_tokens = model.req_manager.req_to_accept_len[req_first]
    return
