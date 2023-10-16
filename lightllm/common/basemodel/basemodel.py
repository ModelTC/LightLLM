import os
import json
import torch
from typing import final

from lightllm.common.basemodel.layer_weights.hf_load_utils import load_hf_weights
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.mem_manager import MemoryManager
from lightllm.common.infer_utils import init_bloc
from lightllm.common.build_utils import repair_config


class TpPartBaseModel:
    # weight class
    pre_and_post_weight_class = None
    transformer_weight_class = None

    # infer class
    pre_layer_infer_class = None
    post_layer_infer_class = None
    transformer_layer_infer_class = None

    # infer state class
    infer_state_class = InferStateInfo

    def __init__(self, tp_rank, world_size, weight_dir, max_total_token_num, load_way="HF", mode=[], weight_dict=None, finetune_config=None):
        self.tp_rank_ = tp_rank
        self.world_size_ = world_size
        self.weight_dir_ = weight_dir
        self.max_total_token_num = max_total_token_num
        self.load_way = load_way
        self.mode = mode
        self.weight_dict = weight_dict
        self.finetune_config = finetune_config

        self._init_config()
        self._verify_must()
        self._verify_params()
        self._init_weights()
        self._init_mem_manager()
        self._init_infer_layer()
        self._init_some_value()
        self._init_custom()
        return
    
    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), 'r') as json_file:
            self.config = json.load(json_file)
        # rename keys
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])
        if self.finetune_config:
            self.config['vocab_size'] = self.finetune_config.vocab_size
        return
    
    @final
    def _verify_must(self):
        assert self.config["num_attention_heads"] % self.world_size_ == 0
        return
    
    def _verify_params(self):
        assert self.load_way == "HF", "only support HF format weights"
        return

    def _init_weights(self):
        self.pre_post_weight = self.pre_and_post_weight_class(self.tp_rank_, self.world_size_, torch.float16, network_config=self.config, mode=self.mode)
        self.trans_layers_weight = [
            self.transformer_weight_class(i, self.tp_rank_, self.world_size_, torch.float16, network_config=self.config, mode=self.mode)
            for i in range(self.config["n_layer"])
        ]
        load_hf_weights(
            "fp16",
            weight_dir=self.weight_dir_,
            pre_post_layer=self.pre_post_weight,
            transformer_layer_list=self.trans_layers_weight,
            weight_dict=self.weight_dict)
        self.pre_post_weight.verify_load()
        [weight.verify_load() for weight in self.trans_layers_weight]
        return 
    
    def _init_mem_manager(self):
        assert self.config["num_attention_heads"] % self.world_size_ == 0
        self.mem_manager = MemoryManager(self.max_total_token_num, 
                            dtype=torch.float16,
                            head_num=self.config["num_attention_heads"] // self.world_size_,
                            head_dim=self.config["n_embed"] // self.config["num_attention_heads"],
                            layer_num=self.config["n_layer"])
        return 
    
    def _init_infer_layer(self):
        self.pre_infer = self.pre_layer_infer_class(tp_rank=self.tp_rank_, world_size=self.world_size_, network_config=self.config, mode=self.mode)
        self.post_infer = self.post_layer_infer_class(tp_rank=self.tp_rank_, world_size=self.world_size_, network_config=self.config, mode=self.mode)
        self.layers_infer = [
            self.transformer_layer_infer_class(
                i,
                tp_rank=self.tp_rank_,
                world_size=self.world_size_,
                network_config=self.config,
                mode=self.mode) for i in range(
                self.config["n_layer"])]
        return
    
    def _init_some_value(self):
        self.head_dim_ = self.config["n_embed"] // self.config["num_attention_heads"]
        self.tp_k_head_num_ = self.config["num_attention_heads"] // self.world_size_
        self.tp_v_head_num_ = self.tp_k_head_num_
        self.layers_num = self.config["n_layer"]
        self.vocab_size = self.config["vocab_size"]
        return
    
    def _init_custom(self):
        pass


    @torch.no_grad()
    def forward(
            self,
            batch_size,
            total_token_num,
            max_len_in_batch,
            input_ids : torch.Tensor,
            b_loc : torch.Tensor,
            b_start_loc : torch.Tensor,
            b_seq_len : torch.Tensor,
            is_prefill=True):
        if is_prefill:
            return self._prefill(batch_size, total_token_num, max_len_in_batch, input_ids, b_loc, b_start_loc, b_seq_len)
        else:
            return self._decode(batch_size, total_token_num, max_len_in_batch, input_ids, b_loc, b_start_loc, b_seq_len)

    
    def _prefill(self, batch_size, total_token_num, max_len_in_batch, input_ids, b_loc, b_start_loc, b_seq_len):
        infer_state = self.infer_state_class()
        infer_state.is_prefill = True
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
        infer_state.prefill_key_buffer = torch.empty((infer_state.total_token_num, self.tp_k_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
        infer_state.prefill_value_buffer = torch.empty((infer_state.total_token_num, self.tp_v_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
        init_bloc(b_loc, b_seq_len, max_len_in_batch, infer_state.prefill_mem_index)

        infer_state.init_some_extra_state(self, batch_size, total_token_num, max_len_in_batch, input_ids, b_loc, b_start_loc, b_seq_len, True)
        predict_logics = self._context_forward(input_ids, infer_state)
        return predict_logics
    
    def _decode(self, batch_size, total_token_num, max_len_in_batch, input_ids, b_loc, b_start_loc, b_seq_len):
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
            infer_state.decode_key_buffer = torch.empty((batch_size, self.tp_k_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
            infer_state.decode_value_buffer = torch.empty((batch_size, self.tp_v_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
            b_loc[:, max_len_in_batch - 1] = infer_state.decode_mem_index

        infer_state.init_some_extra_state(self, batch_size, total_token_num, max_len_in_batch, input_ids, b_loc, b_start_loc, b_seq_len, False)
        predict_logics = self._token_forward(input_ids, infer_state)
        return predict_logics
    
    @final
    def _context_forward(self, input_ids, infer_state: InferStateInfo):
        cuda_input_ids = input_ids
        input_embs = self.pre_infer.context_forward(cuda_input_ids, infer_state, self.pre_post_weight)
        for i in range(self.layers_num):
            input_embs = self.layers_infer[i].context_forward(input_embs, infer_state, self.trans_layers_weight[i])
        predict_logics = self.post_infer.token_forward(input_embs, infer_state, self.pre_post_weight, return_logics=True)
        return predict_logics

    @final
    def _token_forward(self, input_ids, infer_state: InferStateInfo):
        cuda_input_ids = input_ids
        input_embs = self.pre_infer.token_forward(cuda_input_ids, infer_state, self.pre_post_weight)
        for i in range(self.layers_num):
            input_embs = self.layers_infer[i].token_forward(input_embs, infer_state, self.trans_layers_weight[i])
        predict_logics = self.post_infer.token_forward(input_embs, infer_state, self.pre_post_weight, return_logics=True)
        return predict_logics
