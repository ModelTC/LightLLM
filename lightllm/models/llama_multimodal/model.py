import torch

from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight

from lightllm.models.llama_multimodal.layer_infer.pre_layer_infer import LlamaMultiModalPreLayerInfer
from lightllm.models.llama_multimodal.infer_struct import LlamaMultiModalInferStateInfo

from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.common.infer_utils import init_bloc


class LlamaTpPartMulitModal(LlamaTpPartModel):
    # weight class
    pre_and_post_weight_class = LlamaPreAndPostLayerWeight
    transformer_weight_class = LlamaTransformerLayerWeight

    # infer class
    pre_layer_infer_class = LlamaMultiModalPreLayerInfer
    post_layer_infer_class = LlamaPostLayerInfer
    transformer_layer_infer_class = LlamaTransformerLayerInfer

    # infer state class
    infer_state_class = LlamaMultiModalInferStateInfo

    def __init__(self, tp_rank, world_size, weight_dir,
                 max_total_token_num, load_way="HF", mode=""):
        super().__init__(
            tp_rank,
            world_size,
            weight_dir,
            max_total_token_num,
            load_way,
            mode)
        return

    @torch.no_grad()
    def forward(
            self,
            batch_size,
            total_token_num,
            max_len_in_batch,
            input_ids: torch.Tensor,
            b_loc: torch.Tensor,
            b_start_loc: torch.Tensor,
            b_seq_len: torch.Tensor,
            is_prefill=True,
            **kwargs):
        if is_prefill:
            return self._prefill(batch_size, total_token_num, max_len_in_batch,
                                 input_ids, b_loc, b_start_loc, b_seq_len, **kwargs)
        else:
            return self._decode(batch_size, total_token_num, max_len_in_batch,
                                input_ids, b_loc, b_start_loc, b_seq_len, **kwargs)

    def _prefill(self, batch_size, total_token_num, max_len_in_batch,
                 input_ids, b_loc, b_start_loc, b_seq_len, **kwargs):
        infer_state = self.infer_state_class()
        infer_state.is_prefill = True
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch
        assert (
            input_ids.shape[0] == total_token_num), "{} vs {}".format(
            input_ids.shape, total_token_num)
        assert (b_loc.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0])
        infer_state.b_loc = b_loc
        infer_state.b_start_loc = b_start_loc
        infer_state.b_seq_len = b_seq_len

        infer_state.mem_manager = self.mem_manager
        infer_state.prefill_mem_index = self.mem_manager.alloc(
            infer_state.total_token_num)
        infer_state.prefill_key_buffer = torch.empty(
            (infer_state.total_token_num,
             self.tp_k_head_num_,
             self.head_dim_),
            dtype=torch.float16,
            device="cuda")
        infer_state.prefill_value_buffer = torch.empty(
            (infer_state.total_token_num,
             self.tp_v_head_num_,
             self.head_dim_),
            dtype=torch.float16,
            device="cuda")
        init_bloc(
            b_loc,
            b_seq_len,
            max_len_in_batch,
            infer_state.prefill_mem_index)

        infer_state.init_some_extra_state(
            self,
            batch_size,
            total_token_num,
            max_len_in_batch,
            input_ids,
            b_loc,
            b_start_loc,
            b_seq_len,
            True,
            **kwargs)
        predict_logics = self._context_forward(input_ids, infer_state)
        return predict_logics

    def _decode(self, batch_size, total_token_num, max_len_in_batch,
                input_ids, b_loc, b_start_loc, b_seq_len, **kwargs):
        infer_state = self.infer_state_class()
        infer_state.is_prefill = False
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch
        assert (b_loc.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0])
        infer_state.b_loc = b_loc
        infer_state.b_start_loc = b_start_loc
        infer_state.b_seq_len = b_seq_len

        infer_state.mem_manager = self.mem_manager

        alloc_mem = self.mem_manager.alloc_contiguous(batch_size)
        if alloc_mem is not None:
            infer_state.decode_is_contiguous = True
            infer_state.decode_mem_index = alloc_mem[0]
            infer_state.decode_mem_start = alloc_mem[1]
            infer_state.decode_mem_end = alloc_mem[2]
            b_loc[:, max_len_in_batch - 1] = infer_state.decode_mem_index
        else:
            infer_state.decode_is_contiguous = False
            alloc_mem = self.mem_manager.alloc(batch_size)
            infer_state.decode_mem_index = alloc_mem
            infer_state.decode_key_buffer = torch.empty(
                (batch_size,
                 self.tp_k_head_num_,
                 self.head_dim_),
                dtype=torch.float16,
                device="cuda")
            infer_state.decode_value_buffer = torch.empty(
                (batch_size,
                 self.tp_v_head_num_,
                 self.head_dim_),
                dtype=torch.float16,
                device="cuda")
            b_loc[:, max_len_in_batch - 1] = infer_state.decode_mem_index

        infer_state.init_some_extra_state(
            self,
            batch_size,
            total_token_num,
            max_len_in_batch,
            input_ids,
            b_loc,
            b_start_loc,
            b_seq_len,
            False,
            **kwargs)
        predict_logics = self._token_forward(input_ids, infer_state)
        return predict_logics
