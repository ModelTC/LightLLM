import os
import json
from lightllm.models.bloom.layer_infer.pre_layer_inference import PreLayerInfer
from lightllm.models.bloom.layer_infer.post_layer_inference import PostLayerInfer
from lightllm.models.bloom.layer_infer.transformer_layer_inference import TransformerLayerInfer
from lightllm.models.bloom.layer_weights.pre_and_post_layer_weight import *
from lightllm.models.bloom.layer_weights.transformer_layer_weight import *
from lightllm.models.bloom.layer_infer.infer_struct import InferStateInfo
from lightllm.common.mem_manager import MemoryManager
from lightllm.common.infer_utils import init_bloc

class BloomTpPartModel:
    def __init__(self, tp_rank, world_size, weight_dir, max_total_token_num, load_way="HF", mode=""):
        self.tp_rank_ = tp_rank
        self.world_size_ = world_size
        self.weight_dir_ = weight_dir
        with open(os.path.join(weight_dir, "config.json"), 'r') as json_file:
            self.config = json.load(json_file)

        assert mode == "", "future to support int8 int4 ..."
        
        self.mem_manager = MemoryManager(max_total_token_num, 
                                    dtype=torch.float16,
                                    head_num=self.config["num_attention_heads"] // world_size,
                                    head_dim=self.config["n_embed"] // self.config["num_attention_heads"],
                                    layer_num=self.config["n_layer"])

        data_type_str = "fp16"
        self.pre_post_weight = PreAndPostLayerWeight(self.tp_rank_, self.world_size_, data_type_str, network_config=self.config)
        self.trans_layers_weight = [
            TransformerLayerWeight(i, self.tp_rank_, self.world_size_, data_type_str, network_config=self.config)
            for i in range(self.config["n_layer"])
        ]

        if load_way == "FT":
            from lightllm.models.bloom.layer_weights.ft_load_utils import load_ft_weights
            load_ft_weights(
                data_type_str,
                weight_dir=weight_dir,
                pre_post_layer=self.pre_post_weight,
                transformer_layer_list=self.trans_layers_weight)
        if load_way == "HF":
            from lightllm.models.bloom.layer_weights.hf_load_utils import load_hf_weights
            load_hf_weights(
                data_type_str,
                weight_dir=weight_dir,
                pre_post_layer=self.pre_post_weight,
                transformer_layer_list=self.trans_layers_weight)

        self.pre_infer = PreLayerInfer(tp_rank=self.tp_rank_, world_size=self.world_size_, network_config=self.config)
        self.post_infer = PostLayerInfer(tp_rank=self.tp_rank_, world_size=self.world_size_, network_config=self.config)
        self.layers_infer = [
            TransformerLayerInfer(
                i,
                tp_rank=self.tp_rank_,
                world_size=self.world_size_,
                network_config=self.config) for i in range(
                self.config["n_layer"])]

        self.head_num_ = self.config["num_attention_heads"]
        self.head_dim_ = self.config["n_embed"] // self.head_num_
        assert self.head_num_ % self.world_size_ == 0
        self.tp_head_num_ = self.head_num_ // self.world_size_
        self.vocab_size = self.config["vocab_size"]
        return

    @torch.no_grad()
    def forward(
            self,
            batch_size,
            total_token_num,
            max_len_in_batch,
            input_ids,
            b_loc,
            b_start_loc,
            b_seq_len,
            is_prefill=True):
        if is_prefill:
            infer_state = InferStateInfo()
            infer_state.is_prefill = is_prefill
            infer_state.batch_size = batch_size
            infer_state.total_token_num = total_token_num
            infer_state.max_len_in_batch = max_len_in_batch
            assert (input_ids.shape[0] == total_token_num)
            assert (b_loc.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0])
            infer_state.b_loc = b_loc
            infer_state.b_start_loc = b_start_loc
            infer_state.b_seq_len = b_seq_len
            
            infer_state.mem_manager = self.mem_manager
            infer_state.prefill_mem_index = self.mem_manager.alloc(infer_state.total_token_num)
            infer_state.prefill_key_buffer = torch.empty((infer_state.total_token_num, self.tp_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
            infer_state.prefill_value_buffer = torch.empty((infer_state.total_token_num, self.tp_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
            init_bloc(b_loc, b_seq_len, max_len_in_batch, infer_state.prefill_mem_index)

            predict_logics = self._context_forward(input_ids, infer_state)
            return predict_logics
        else:
            infer_state = InferStateInfo()
            infer_state.is_prefill = is_prefill
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
                infer_state.decode_key_buffer = torch.empty((batch_size, self.tp_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
                infer_state.decode_value_buffer = torch.empty((batch_size, self.tp_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
                b_loc[:, max_len_in_batch - 1] = infer_state.decode_mem_index

            predict_logics = self._token_forward(input_ids, infer_state)
            return predict_logics

    def _context_forward(self, input_ids, infer_state: InferStateInfo):
        cuda_input_ids = input_ids
        input_embs = self.pre_infer.context_forward(cuda_input_ids, infer_state, self.pre_post_weight)
        for i in range(self.config["n_layer"]):
            input_embs = self.layers_infer[i].context_forward(input_embs, infer_state, self.trans_layers_weight[i])
        predict_logics = self.post_infer.token_forward(input_embs, infer_state, self.pre_post_weight, return_logics=True)
        return predict_logics

    def _token_forward(self, input_ids, infer_state: InferStateInfo):
        cuda_input_ids = input_ids
        input_embs = self.pre_infer.token_forward(cuda_input_ids, infer_state, self.pre_post_weight)
        for i in range(self.config["n_layer"]):
            input_embs = self.layers_infer[i].token_forward(input_embs, infer_state, self.trans_layers_weight[i])
        predict_logics = self.post_infer.token_forward(input_embs, infer_state, self.pre_post_weight, return_logics=True)
        return predict_logics
