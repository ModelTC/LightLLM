import os
import json
import torch
from lightllm.models.llama.layer_infer.pre_layer_inference import PreLayerInfer
from lightllm.models.llama.layer_infer.post_layer_inference import PostLayerInfer
from lightllm.models.llama.layer_infer.transformer_layer_inference import TransformerLayerInfer
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import *
from lightllm.models.llama.layer_weights.transformer_layer_weight import *
from lightllm.models.llama.layer_infer.infer_struct import InferStateInfo
from lightllm.models.llama.layer_weights.hf_load_utils import load_hf_weights
from lightllm.common.mem_manager import MemoryManager
from lightllm.common.int8kv_mem_manager import INT8KVMemoryManager
from lightllm.common.infer_utils import init_bloc

class LlamaTpPartModel:
    def __init__(self, tp_rank, world_size, weight_dir, max_total_token_num, load_way="HF", mode=""):
        self.tp_rank_ = tp_rank
        self.world_size_ = world_size
        self.weight_dir_ = weight_dir
        with open(os.path.join(weight_dir, "config.json"), 'r') as json_file:
            self.config = json.load(json_file)

        assert load_way == "HF", "llama only support HF format to load Now!"
        assert mode in ["", "int8kv"], "now support int8kv, future to support int8 int4 ..."

        mem_dict = {
            "" : MemoryManager,
            "int8kv" : INT8KVMemoryManager
        }
        
        self.mem_manager = mem_dict[mode](max_total_token_num, 
                                         dtype=torch.float16,
                                         head_num=self.config["num_attention_heads"] // world_size,
                                         head_dim=self.config["hidden_size"] // self.config["num_attention_heads"],
                                         layer_num=self.config["num_hidden_layers"])

        self.pre_post_weight = PreAndPostLayerWeight(self.tp_rank_, self.world_size_, torch.float16, self.config)
        self.trans_layers_weight = [
            TransformerLayerWeight(i, self.tp_rank_, self.world_size_, torch.float16, self.config, mode=mode)
            for i in range(self.config["num_hidden_layers"])
        ]

        load_hf_weights("fp16", weight_dir, pre_post_layer=self.pre_post_weight, transformer_layer_list=self.trans_layers_weight)

        self.pre_infer = PreLayerInfer(tp_rank=self.tp_rank_, world_size=self.world_size_, network_config=self.config)
        self.post_infer = PostLayerInfer(tp_rank=self.tp_rank_, world_size=self.world_size_, network_config=self.config)
        self.layers_infer = [
            TransformerLayerInfer(
                i,
                tp_rank=self.tp_rank_,
                world_size=self.world_size_,
                network_config=self.config,
                mode=mode) for i in range(
                self.config["num_hidden_layers"])]

        self.head_num_ = self.config["num_attention_heads"]
        self.head_dim_ = self.config["hidden_size"] // self.head_num_
        assert self.head_num_ % self.world_size_ == 0
        self.tp_head_num_ = self.head_num_ // self.world_size_
        self.vocab_size = self.config["vocab_size"]
        self.init_to_get_rotary()

    def init_to_get_rotary(self, base=10000):
        if self.config.get("rope_scaling", {}) is None:
            rope_scaling_factor = 1.0
        else:
            rope_scaling_factor = self.config.get("rope_scaling", {}).get("factor", 1.0)
        if "max_sequence_length" in self.config:
            max_seq_len = self.config["max_sequence_length"]
        else:
            max_seq_len = self.config.get("max_position_embeddings", 2048) * rope_scaling_factor
        base = float(base)
        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim_, 2, device="cpu", dtype=torch.float32) / self.head_dim_))
        t = torch.arange(max_seq_len + 1024 * 64, device="cpu", dtype=torch.float32) / rope_scaling_factor
        freqs = torch.outer(t, inv_freq)

        self._cos_cached = torch.cos(freqs).to(torch.float16).cuda()
        self._sin_cached = torch.sin(freqs).to(torch.float16).cuda()
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

            b_seq_len_numpy = b_seq_len.cpu().numpy()
            position_ids = torch.from_numpy(np.concatenate([np.arange(0, b_seq_len_numpy[i])
                                            for i in range(len(b_seq_len_numpy))], axis=0)).cuda()
            infer_state.position_cos = torch.index_select(self._cos_cached, 0, position_ids).view(position_ids.shape[0], -1)
            infer_state.position_sin = torch.index_select(self._sin_cached, 0, position_ids).view(position_ids.shape[0], -1)
            position_ids = None
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
            infer_state.position_cos = torch.index_select(self._cos_cached, 0, b_seq_len - 1).view(b_seq_len.shape[0], -1)
            infer_state.position_sin = torch.index_select(self._sin_cached, 0, b_seq_len - 1).view(b_seq_len.shape[0], -1)
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
        for i in range(self.config["num_hidden_layers"]):
            input_embs = self.layers_infer[i].context_forward(input_embs, infer_state, self.trans_layers_weight[i])
        predict_logics = self.post_infer.token_forward(input_embs, infer_state, self.pre_post_weight, return_logics=True)
        return predict_logics

    def _token_forward(self, input_ids, infer_state: InferStateInfo):
        cuda_input_ids = input_ids
        input_embs = self.pre_infer.token_forward(cuda_input_ids, infer_state, self.pre_post_weight)
        for i in range(self.config["num_hidden_layers"]):
            input_embs = self.layers_infer[i].token_forward(input_embs, infer_state, self.trans_layers_weight[i])
        predict_logics = self.post_infer.token_forward(input_embs, infer_state, self.pre_post_weight, return_logics=True)
        return predict_logics
