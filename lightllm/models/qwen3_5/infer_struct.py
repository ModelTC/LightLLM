import torch

from lightllm.models.qwen2_vl.infer_struct import Qwen2VLInferStateInfo
from lightllm.utils.envs_utils import get_env_start_args


class Qwen35InferStateInfo(Qwen2VLInferStateInfo):
    def __init__(self):
        super().__init__()
        self.gate_value = None

    def init_some_extra_state(self, model):
        super().init_some_extra_state(model)
        self.b_att_seq_len = self.b_seq_len
        mtp_step = get_env_start_args().mtp_step

        self.b_buffer_idx = self.b_req_idx
        self.b_conv_buffer_idx = self.b_req_idx
        self.is_decode_with_mtp = False
        is_mtp_draft_model = getattr(model, "is_mtp_draft_model", False)
        if mtp_step <= 0 or is_mtp_draft_model:
            # Draft 模型不走线性层 MTP 状态。
            return
        step = mtp_step + 1
        if self.is_prefill:
            self.b_buffer_idx = self.b_req_idx * step + self.b_mtp_index
            return

        self.is_decode_with_mtp = True
        batch_size = self.batch_size
        att_batch_size = batch_size // step
        assert batch_size % step == 0
        self.b1_mtp_cu_q_seq_len = torch.arange(
            0, batch_size + 1, step, dtype=torch.int32, device=self.b_req_idx.device
        )
        req_first = self.b_req_idx.view(att_batch_size, step)[:, 0]
        base = (req_first * step).view(att_batch_size, 1)
        self.b_ssm_index_rows = base + torch.arange(step, device=base.device, dtype=base.dtype).view(1, step)
        self.b_conv_buffer_idx = req_first
        self.b_num_accepted_tokens = model.req_manager.req_to_accept_len[req_first]
        return
