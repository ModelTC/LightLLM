import os
import torch
import numpy as np
import torch.distributed as dist
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.models.deepseek2.triton_kernel.repack_kv_index import repack_kv_index
from lightllm.common.basemodel.batch_objs import ModelInput
from lightllm.common.basemodel.triton_kernel.fa3_utils import page_table_copy


class FlashAttentionStateInfo(LlamaInferStateInfo):
    _shared_page_table_buffer = None

    def __init__(self):
        super().__init__()

    @classmethod
    def get_page_table_buffer(cls, graph_max_batch_size: int, max_seq_len: int):
        if cls._shared_page_table_buffer is None:
            cls._shared_page_table_buffer = [
                torch.empty(graph_max_batch_size * max_seq_len, dtype=torch.int32).to(get_current_device_id()),
                torch.empty(graph_max_batch_size * max_seq_len, dtype=torch.int32).to(get_current_device_id()),
            ]
        return cls._shared_page_table_buffer

    def _init_flash_attention_state(self, model):
        if self.is_prefill:
            self.cu_seqlens_q = self.b1_cu_q_seq_len.int()
            self.cu_seqlens_k = self.b1_cu_kv_seq_len.int()
            self.page_table = torch.empty(
                (self.batch_size, self.max_seq_len), dtype=torch.int32, device=self.input_ids.device
            )
            self.page_table.copy_(model.req_manager.req_to_token_indexs[self.b_req_idx, : self.max_seq_len])
        else:
            # Meta information of flashattention for decoding
            self.cu_seqlens_q = self.b1_cu_q_seq_len.int()
            self.cu_seqlens_k = self.b1_cu_kv_seq_len.int()
            max_seq_len_k = self.max_kv_seq_len
            args_mtp_step = get_env_start_args().mtp_step
            att_batch_size = self.batch_size // (args_mtp_step + 1)
            if self.batch_size <= model.graph_max_batch_size and self.max_len_in_batch <= model.graph_max_len_in_batch:
                page_buffer = FlashAttentionStateInfo.get_page_table_buffer(
                    model.graph_max_batch_size, model.graph_max_len_in_batch
                )
                self.page_table = page_buffer[self.microbatch_index][
                    : att_batch_size * model.graph_max_len_in_batch
                ].reshape(att_batch_size, model.graph_max_len_in_batch)
            else:
                self.page_table = torch.empty(
                    (att_batch_size, self.max_len_in_batch), dtype=torch.int32, device=self.input_ids.device
                )

            page_table_copy(
                page_table=self.page_table[:, :max_seq_len_k],
                req_to_token_indexs=model.req_manager.req_to_token_indexs,
                b_req_idx=self.b_req_idx[args_mtp_step :: (args_mtp_step + 1)],
            )
            if args_mtp_step > 0:
                self.b_att_seq_len = self.b_seq_len[args_mtp_step :: (args_mtp_step + 1)].contiguous()
            else:
                self.b_att_seq_len = self.b_seq_len

        if "offline_calibration_fp8kv" in model.mode:
            if self.is_prefill:
                device = self.input_ids.device
                # q_scale和token_batch_ids在对q做per head量化使用，为了节省资源在推理外部初始化
                self.q_scale = torch.empty(
                    (self.batch_size, self.mem_manager.head_num), dtype=torch.float32, device=device
                )
                self.token_batch_ids = torch.repeat_interleave(
                    torch.arange(self.batch_size, device=device), self.b_q_seq_len
                )

            offline_scales = self.mem_manager.scales
            head_num = self.mem_manager.head_num
            # 为了减少推理计算量，在推理外部初始化k_descale和v_descale
            self.k_descale = (
                offline_scales[:, :head_num]
                .view(-1, 1, head_num)
                .expand(offline_scales.shape[0], self.batch_size, head_num)
                if offline_scales is not None
                else torch.ones(
                    (self.mem_manager.layer_num, self.batch_size, head_num),
                    dtype=torch.float32,
                    device=self.input_ids.device,
                )
            )
            self.v_descale = (
                offline_scales[:, head_num:]
                .view(-1, 1, head_num)
                .expand(offline_scales.shape[0], self.batch_size, head_num)
                if offline_scales is not None
                else torch.ones(
                    (self.mem_manager.layer_num, self.batch_size, head_num),
                    dtype=torch.float32,
                    device=self.input_ids.device,
                )
            )
        return

    def init_some_extra_state(self, model):
        super().init_some_extra_state(model)
        self._init_flash_attention_state(model)
        return
