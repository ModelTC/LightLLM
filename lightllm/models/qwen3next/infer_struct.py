import torch
from lightllm.models.llama.flashattention_infer_struct import FlashAttentionStateInfo
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.common.basemodel.triton_kernel.fa3_utils import page_table_copy


class Qwen3NextFlashAttentionStateInfo(FlashAttentionStateInfo):
    """FlashAttentionStateInfo with MTP-aware page_table and cache_seqlens handling for Qwen3Next."""

    def __init__(self):
        super().__init__()
        self.b_att_seq_len = None

    def _init_flash_attention_state(self, model, input_ids: torch.Tensor):
        args_mtp_step = get_env_start_args().mtp_step

        if self.is_prefill:
            # Prefill path - same as parent
            self.cu_seqlens_q = self.b1_cu_q_seq_len.int()
            self.cu_seqlens_k = self.b1_cu_kv_seq_len.int()
            self.page_table = torch.empty(
                (self.batch_size, self.max_seq_len), dtype=torch.int32, device=input_ids.device
            )
            self.page_table.copy_(model.req_manager.req_to_token_indexs[self.b_req_idx, : self.max_seq_len])
            self.b_att_seq_len = self.b_seq_len
        else:
            # Decode path - MTP-aware handling
            self.cu_seqlens_q = self.b1_cu_q_seq_len.int()
            self.cu_seqlens_k = self.b1_cu_kv_seq_len.int()
            max_seq_len_k = self.max_kv_seq_len

            # In MTP mode, each request has (mtp_step + 1) tokens
            # att_batch_size is the number of unique requests
            mtp_size = args_mtp_step + 1
            att_batch_size = self.batch_size // mtp_size

            if self.batch_size <= model.graph_max_batch_size and self.max_len_in_batch <= model.graph_max_len_in_batch:
                page_buffer = Qwen3NextFlashAttentionStateInfo.get_page_table_buffer(
                    model.graph_max_batch_size, model.graph_max_len_in_batch
                )
                # IMPORTANT: Use att_batch_size for page_table, not batch_size
                self.page_table = page_buffer[self.microbatch_index][
                    : att_batch_size * model.graph_max_len_in_batch
                ].view(att_batch_size, model.graph_max_len_in_batch)
            else:
                self.page_table = torch.empty(
                    (att_batch_size, self.max_len_in_batch), dtype=torch.int32, device=input_ids.device
                )

            # Copy page table using only the last token of each MTP group's request index
            # This ensures we get one row per unique request
            page_table_copy(
                page_table=self.page_table[:, :max_seq_len_k],
                req_to_token_indexs=model.req_manager.req_to_token_indexs,
                b_req_idx=self.b_req_idx[args_mtp_step::mtp_size],
            )

            # Use only the sequence lengths for the last token of each MTP group
            # These represent the actual KV cache lengths for each unique request
            if args_mtp_step > 0:
                self.b_att_seq_len = self.b_seq_len[args_mtp_step::mtp_size].contiguous()
            else:
                self.b_att_seq_len = self.b_seq_len

        # Initialize FP8 calibration scales for flash attention
        # This must be done in the parent class's _init_flash_attention_state first
        if "offline_calibration_fp8kv" in model.mode and not self.is_prefill:
            # In MTP mode, k_descale and v_descale need to match b_att_seq_len
            # The parent class expands them to batch_size, but we need att_batch_size
            if args_mtp_step > 0:
                # Sample k_descale and v_descale to match att_batch_size
                if hasattr(self, "k_descale") and self.k_descale is not None:
                    self.k_descale = self.k_descale[:, args_mtp_step::mtp_size, :].contiguous()
                if hasattr(self, "v_descale") and self.v_descale is not None:
                    self.v_descale = self.v_descale[:, args_mtp_step::mtp_size, :].contiguous()

        return
