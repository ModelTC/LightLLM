import torch
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.utils.envs_utils import get_env_start_args


class Qwen3NextInferStateInfo(LlamaInferStateInfo):
    """
    Inference state for Qwen3Next with:
    - gate_value attribute for output gating in full attention layers
    - MTP-aware batching for multi-token prediction
    - Custom buffer management for hybrid attention (full + linear)
    """

    def __init__(self):
        super().__init__()
        # For output gating in full attention layers
        self.gate_value = None
        # MTP-aware attributes
        self.b_att_seq_len = None
        self.att_batch_size = None
        self.real_req_idx = None
        self.mtp_buffer_idx_list = None
        self.b_buffer_idx = None

    def init_some_extra_state(self, model):
        """Initialize Qwen3Next-specific state"""
        super().init_some_extra_state(model)

        args_mtp_step = get_env_start_args().mtp_step
        mtp_size = args_mtp_step + 1

        if self.is_prefill:
            # Prefill: Standard initialization
            self.b_att_seq_len = self.b_seq_len
            self.b_buffer_idx = model.req_manager.req_to_buffer_index[self.b_req_idx, 0].contiguous()
        else:
            # Decode: MTP-aware handling
            # In MTP mode, each request has (mtp_step + 1) tokens
            # att_batch_size is the number of unique requests
            self.att_batch_size = self.batch_size // mtp_size

            # Use only the sequence lengths for the last token of each MTP group
            if args_mtp_step > 0:
                self.b_att_seq_len = self.b_seq_len[args_mtp_step::mtp_size].contiguous()
                self.real_req_idx = self.b_req_idx[args_mtp_step::mtp_size]
            else:
                self.b_att_seq_len = self.b_seq_len
                self.real_req_idx = self.b_req_idx

            # Buffer indices for Mamba cache (conv and SSM states)
            self.b_buffer_idx = model.req_manager.req_to_buffer_index[self.real_req_idx, :].flatten().contiguous()

            # Create per-step buffer indices for MTP
            if args_mtp_step > 0:
                buffer_idx_list = []
                for step_id in range(mtp_size):
                    buffer_idx_list.append(self.b_buffer_idx[step_id::mtp_size].tolist())
                self.mtp_buffer_idx_list = torch.tensor(
                    buffer_idx_list, dtype=torch.int32, device=self.b_buffer_idx.device
                )

        return
