import os

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import json
import torch
from typing import final

from lightllm.common.basemodel.layer_weights.hf_load_utils import load_hf_weights
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.basemodel.splitfuse_infer_struct import SplitFuseInferStateInfo
from lightllm.common.mem_manager import MemoryManager
from lightllm.common.req_manager import ReqManager
from lightllm.common.infer_utils import init_req_to_token_indexes
from lightllm.common.build_utils import repair_config
from lightllm.common.basemodel.triton_kernel.copy_kv_index_to_req import copy_kv_index_to_req
from lightllm.common.basemodel.triton_kernel.splitfuse_copy_kv_index_to_req import splitfuse_copy_kv_index_to_req

torch.backends.cudnn.enabled = True


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
    splitfuse_infer_state_class = SplitFuseInferStateInfo

    def __init__(self, kvargs):
        self.tp_rank_ = kvargs["tp_rank"]
        self.world_size_ = kvargs["world_size"]
        self.weight_dir_ = kvargs["weight_dir"]
        self.max_total_token_num = kvargs["max_total_token_num"]
        self.load_way = kvargs.get("load_way", "HF")
        self.mode = [m.replace('int4weight', 'w4a16').replace('int8weight', 'w8a16') for m in kvargs.get("mode", [])]
        self.weight_dict = kvargs.get("weight_dict", None)
        self.finetune_config = kvargs.get("finetune_config", None)
        self.max_req_num = kvargs.get("max_req_num", 1000)
        self.max_seq_length = kvargs.get("max_seq_length", 1024 * 5)
        self.return_all_prompt_logprobs = kvargs.get("return_all_prompt_logprobs", False)
        self.use_dynamic_prompt_cache = kvargs.get("use_dynamic_prompt_cache", False)

        self._init_config()
        self._verify_must()
        self._verify_params()
        self._init_weights()
        self._init_mem_manager()
        self._init_req_manager()
        self._init_infer_layer()
        self._init_some_value()
        self._init_custom()
        return

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), "r") as json_file:
            self.config = json.load(json_file)
        # rename keys
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])
        if self.finetune_config:
            self.config["vocab_size"] = self.finetune_config.vocab_size
        return

    @final
    def _verify_must(self):
        assert self.config["num_attention_heads"] % self.world_size_ == 0
        return

    def _verify_params(self):
        assert self.load_way == "HF", "only support HF format weights"
        assert self.config["num_key_value_heads"] % self.world_size_ == 0
        return

    def _init_weights(self):
        self.pre_post_weight = self.pre_and_post_weight_class(
            self.tp_rank_, self.world_size_, torch.float16, network_config=self.config, mode=self.mode
        )
        self.trans_layers_weight = [
            self.transformer_weight_class(
                i, self.tp_rank_, self.world_size_, torch.float16, network_config=self.config, mode=self.mode
            )
            for i in range(self.config["n_layer"])
        ]
        load_hf_weights(
            "fp16",
            weight_dir=self.weight_dir_,
            pre_post_layer=self.pre_post_weight,
            transformer_layer_list=self.trans_layers_weight,
            weight_dict=self.weight_dict,
        )
        self.pre_post_weight.verify_load()
        [weight.verify_load() for weight in self.trans_layers_weight]
        return

    def _init_mem_manager(self):
        assert self.config["num_attention_heads"] % self.world_size_ == 0
        self.mem_manager = MemoryManager(
            self.max_total_token_num,
            dtype=torch.float16,
            head_num=self.config["num_attention_heads"] // self.world_size_,
            head_dim=self.config["n_embed"] // self.config["num_attention_heads"],
            layer_num=self.config["n_layer"],
        )
        return

    def _init_req_manager(self):
        self.req_manager = ReqManager(self.max_req_num, self.max_seq_length, self.mem_manager)
        return

    def _init_infer_layer(self):
        self.pre_infer = self.pre_layer_infer_class(
            tp_rank=self.tp_rank_, world_size=self.world_size_, network_config=self.config, mode=self.mode
        )
        self.post_infer = self.post_layer_infer_class(
            tp_rank=self.tp_rank_, world_size=self.world_size_, network_config=self.config, mode=self.mode
        )
        self.layers_infer = [
            self.transformer_layer_infer_class(
                i, tp_rank=self.tp_rank_, world_size=self.world_size_, network_config=self.config, mode=self.mode
            )
            for i in range(self.config["n_layer"])
        ]
        return

    def _init_some_value(self):
        self.head_dim_ = self.config["n_embed"] // self.config["num_attention_heads"]
        self.tp_k_head_num_ = self.config["num_key_value_heads"] // self.world_size_
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
        input_ids: torch.Tensor,
        b_req_idx: torch.Tensor,
        b_start_loc: torch.Tensor,
        b_seq_len: torch.Tensor,
        b_ready_cache_len: torch.Tensor = None,
        multimodal_params=None,
        is_prefill=True,
    ):
        if is_prefill:
            return self._prefill(
                batch_size,
                total_token_num,
                max_len_in_batch,
                input_ids,
                b_req_idx,
                b_start_loc,
                b_seq_len,
                b_ready_cache_len,
                multimodal_params,
            )
        else:
            return self._decode(
                batch_size,
                total_token_num,
                max_len_in_batch,
                input_ids,
                b_req_idx,
                b_start_loc,
                b_seq_len,
                multimodal_params,
            )

    def _prefill(
        self,
        batch_size,
        total_token_num,
        max_len_in_batch,
        input_ids,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        b_ready_cache_len,
        multimodal_params,
    ):
        infer_state = self.infer_state_class()
        infer_state.is_prefill = True
        infer_state.return_all_prompt_logprobs = self.return_all_prompt_logprobs
        infer_state.use_dynamic_prompt_cache = self.use_dynamic_prompt_cache
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch
        assert b_req_idx.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0]
        infer_state.b_req_idx = b_req_idx
        infer_state.b_start_loc = b_start_loc
        infer_state.b_seq_len = b_seq_len
        if b_ready_cache_len is not None:
            infer_state.b_ready_cache_len = b_ready_cache_len
        else:
            infer_state.b_ready_cache_len = torch.zeros_like(b_seq_len, dtype=b_seq_len.dtype, device=b_seq_len.device)
        infer_state.multimodal_params = multimodal_params

        infer_state.mem_manager = self.mem_manager
        infer_state.req_manager = self.req_manager

        alloc_mem = self.mem_manager.alloc_contiguous(input_ids.shape[0])
        if alloc_mem is not None:
            infer_state.mem_is_contiguous = True
            infer_state.mem_index = alloc_mem[0]
            infer_state.mem_start = alloc_mem[1]
            infer_state.mem_end = alloc_mem[2]

        else:
            infer_state.mem_is_contiguous = False
            alloc_mem = self.mem_manager.alloc(input_ids.shape[0])
            infer_state.mem_index = alloc_mem
            infer_state.kv_buffer = torch.empty(
                (input_ids.shape[0], self.tp_k_head_num_ + self.tp_v_head_num_, self.head_dim_),
                dtype=torch.float16,
                device="cuda",
            )

        init_req_to_token_indexes(
            self.req_manager.req_to_token_indexs,
            b_req_idx,
            b_seq_len,
            infer_state.b_ready_cache_len,
            max_len_in_batch,
            infer_state.mem_index,
        )

        infer_state.init_some_extra_state(self, input_ids)
        predict_logics = self._context_forward(input_ids, infer_state)
        return predict_logics

    def _decode(
        self,
        batch_size,
        total_token_num,
        max_len_in_batch,
        input_ids,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        multimodal_params,
    ):
        infer_state = self.infer_state_class()
        infer_state.is_prefill = False
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch
        infer_state.use_dynamic_prompt_cache = self.use_dynamic_prompt_cache
        assert b_req_idx.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0]
        infer_state.b_req_idx = b_req_idx
        infer_state.b_start_loc = b_start_loc
        infer_state.b_seq_len = b_seq_len
        infer_state.multimodal_params = multimodal_params

        infer_state.mem_manager = self.mem_manager
        infer_state.req_manager = self.req_manager

        alloc_mem = self.mem_manager.alloc_contiguous(batch_size)
        if alloc_mem is not None:
            infer_state.mem_is_contiguous = True
            infer_state.mem_index = alloc_mem[0]
            infer_state.mem_start = alloc_mem[1]
            infer_state.mem_end = alloc_mem[2]
            copy_kv_index_to_req(self.req_manager.req_to_token_indexs, b_req_idx, b_seq_len, infer_state.mem_index)
        else:
            infer_state.mem_is_contiguous = False
            alloc_mem = self.mem_manager.alloc(batch_size)
            infer_state.mem_index = alloc_mem
            infer_state.kv_buffer = torch.empty(
                (batch_size, self.tp_k_head_num_ + self.tp_v_head_num_, self.head_dim_),
                dtype=torch.float16,
                device="cuda",
            )
            copy_kv_index_to_req(self.req_manager.req_to_token_indexs, b_req_idx, b_seq_len, infer_state.mem_index)

        infer_state.init_some_extra_state(self, input_ids)
        predict_logics = self._token_forward(input_ids, infer_state)
        return predict_logics

    @torch.no_grad()
    def splitfuse_forward(
        self,
        input_ids,
        decode_req_num,
        decode_total_token_num,
        decode_b_req_idx: torch.Tensor,
        decode_b_start_loc: torch.Tensor,
        decode_b_seq_len: torch.Tensor,
        decode_max_len_in_batch,
        prefill_req_num,
        prefill_b_req_idx: torch.Tensor,
        prefill_b_split_start_loc: torch.Tensor,
        prefill_b_split_ready_cache_len: torch.Tensor,
        prefill_max_split_seq_len_in_batch,
        prefill_b_seq_len: torch.Tensor,
    ):

        infer_state = self.splitfuse_infer_state_class()
        infer_state.use_dynamic_prompt_cache = self.use_dynamic_prompt_cache
        infer_state.batch_size = decode_req_num + prefill_req_num

        infer_state.decode_req_num = decode_req_num
        infer_state.decode_total_token_num = decode_total_token_num
        infer_state.decode_b_req_idx = decode_b_req_idx
        infer_state.decode_b_start_loc = decode_b_start_loc
        infer_state.decode_b_seq_len = decode_b_seq_len
        infer_state.decode_max_len_in_batch = decode_max_len_in_batch

        infer_state.prefill_req_num = prefill_req_num
        infer_state.prefill_b_req_idx = prefill_b_req_idx
        infer_state.prefill_b_split_start_loc = prefill_b_split_start_loc
        infer_state.prefill_b_split_ready_cache_len = prefill_b_split_ready_cache_len
        infer_state.prefill_max_split_seq_len_in_batch = prefill_max_split_seq_len_in_batch
        infer_state.prefill_b_seq_len = prefill_b_seq_len
        # infer_state.event = [torch.cuda.Event() for _ in range(self.layers_num)]

        infer_state.mem_manager = self.mem_manager
        infer_state.req_manager = self.req_manager

        alloc_size = len(input_ids)
        alloc_mem = self.mem_manager.alloc_contiguous(alloc_size)
        if alloc_mem is not None:
            infer_state.mem_is_contiguous = True
            infer_state.mem_index = alloc_mem[0]
            infer_state.mem_start = alloc_mem[1]
            infer_state.mem_end = alloc_mem[2]
        else:
            infer_state.mem_is_contiguous = False
            alloc_mem = self.mem_manager.alloc(alloc_size)
            infer_state.mem_index = alloc_mem
            infer_state.kv_buffer = torch.empty(
                (alloc_size, self.tp_k_head_num_ + self.tp_v_head_num_, self.head_dim_),
                dtype=torch.float16,
                device="cuda",
            )

        # decode 部分
        if decode_req_num != 0:
            copy_kv_index_to_req(
                self.req_manager.req_to_token_indexs,
                decode_b_req_idx,
                decode_b_seq_len,
                infer_state.mem_index[0:decode_req_num],
            )

        # split prefill 部分
        if prefill_req_num != 0:
            splitfuse_copy_kv_index_to_req(
                self.req_manager.req_to_token_indexs,
                prefill_b_req_idx,
                prefill_b_split_ready_cache_len,
                prefill_b_seq_len,
                infer_state.mem_index[decode_req_num:],
            )

        infer_state.init_some_extra_state(self, input_ids)
        infer_state.create_inner_decode_infer_status()
        predict_logics = self._splitfuse_forward(input_ids, infer_state)
        return predict_logics

    @final
    def _context_forward(self, input_ids, infer_state: InferStateInfo):
        cuda_input_ids = input_ids
        input_embs = self.pre_infer.context_forward(cuda_input_ids, infer_state, self.pre_post_weight)
        for i in range(self.layers_num):
            input_embs = self.layers_infer[i].context_forward(input_embs, infer_state, self.trans_layers_weight[i])
        predict_logics = self.post_infer.token_forward(
            input_embs, infer_state, self.pre_post_weight, return_logics=True
        )
        return predict_logics

    @final
    def _token_forward(self, input_ids, infer_state: InferStateInfo):
        cuda_input_ids = input_ids
        input_embs = self.pre_infer.token_forward(cuda_input_ids, infer_state, self.pre_post_weight)
        for i in range(self.layers_num):
            input_embs = self.layers_infer[i].token_forward(input_embs, infer_state, self.trans_layers_weight[i])
        predict_logics = self.post_infer.token_forward(
            input_embs, infer_state, self.pre_post_weight, return_logics=True
        )
        return predict_logics

    @final
    def _splitfuse_forward(self, input_ids, infer_state: SplitFuseInferStateInfo):
        cuda_input_ids = input_ids
        input_embs = self.pre_infer.splitfuse_forward(cuda_input_ids, infer_state, self.pre_post_weight)
        for i in range(self.layers_num):
            input_embs = self.layers_infer[i].splitfuse_forward(input_embs, infer_state, self.trans_layers_weight[i])
        predict_logics = self.post_infer.splitfuse_forward(
            input_embs, infer_state, self.pre_post_weight, return_logics=True
        )
        return predict_logics
