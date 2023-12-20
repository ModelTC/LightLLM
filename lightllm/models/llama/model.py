import os
import json
import torch
from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.models.llama.layer_weights.ds_load_utils import load_ds_weights
from lightllm.common.basemodel.layer_weights.hf_load_utils import load_hf_weights

from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.splitfuse_infer_struct import LlamaSplitFuseInferStateInfo
from lightllm.common.basemodel import TpPartBaseModel
from lightllm.common.mem_utils import select_mem_manager_class
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

class LlamaTpPartModel(TpPartBaseModel):
    # weight class
    pre_and_post_weight_class = LlamaPreAndPostLayerWeight
    transformer_weight_class = LlamaTransformerLayerWeight

    # infer class
    pre_layer_infer_class = LlamaPreLayerInfer
    post_layer_infer_class = LlamaPostLayerInfer
    transformer_layer_infer_class = LlamaTransformerLayerInfer

    # infer state class
    infer_state_class = LlamaInferStateInfo
    splitfuse_infer_state_class = LlamaSplitFuseInferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return
    
    def _init_config(self):
        super()._init_config()
        # rename key
        # repair_config()
        self._reset_num_key_value_heads()
        return 
    
    def _reset_num_key_value_heads(self):
        if "num_key_value_heads" not in self.config:
            self.config["num_key_value_heads"] = self.config["num_attention_heads"]
        return

    def _verify_params(self):
        assert self.load_way in ["HF", "DS"], "llama only supports HF and DS format to load Now!"
        assert self.config["num_key_value_heads"] % self.world_size_ == 0
        assert self.config["num_attention_heads"] % self.world_size_ == 0
        return
    
    def _init_mem_manager(self):
        self.mem_manager = select_mem_manager_class(self.mode)(self.max_total_token_num, 
                                                     dtype=torch.float16,
                                                     head_num=self.config["num_key_value_heads"] // self.world_size_,
                                                     head_dim=self.config["hidden_size"] // self.config["num_attention_heads"],
                                                     layer_num=self.config["num_hidden_layers"])
        return

    def _init_custom(self):
        """
        模型特殊的一些初始化
        """
        if self.config.get("use_dynamic_ntk", False):
            self._init_to_get_dynamic_ntk_rotary()
        else:
            self._init_to_get_rotary()
        return

    def _init_weights(self):
        self.pre_post_weight = self.pre_and_post_weight_class(self.tp_rank_, self.world_size_, torch.float16, network_config=self.config, mode=self.mode)
        self.trans_layers_weight = [
            self.transformer_weight_class(i, self.tp_rank_, self.world_size_, torch.float16, network_config=self.config, mode=self.mode)
            for i in range(self.config["n_layer"])
        ]
        if self.load_way == 'HF':
            load_hf_weights(
                "fp16",
                weight_dir=self.weight_dir_,
                pre_post_layer=self.pre_post_weight,
                transformer_layer_list=self.trans_layers_weight,
                weight_dict=self.weight_dict)
        else:
            load_ds_weights(
                "fp16",
                weight_dir=self.weight_dir_,
                pre_post_layer=self.pre_post_weight,
                transformer_layer_list=self.trans_layers_weight,
                weight_dict=self.weight_dict,
                prefix='model.layers.',
                num_layer=self.config["n_layer"])
        self.pre_post_weight.verify_load()
        [weight.verify_load() for weight in self.trans_layers_weight]            
        return 

    def _init_to_get_rotary(self, default_base=10000):
        if self.config.get("rope_scaling", {}) is None:
            rope_scaling_factor = 1.0
        else:
            rope_scaling_factor = self.config.get("rope_scaling", {}).get("factor", 1.0)

        base = self.config.get("rope_theta", float(default_base))

        if "max_sequence_length" in self.config:
            max_seq_len = self.config["max_sequence_length"]
        else:
            max_position_embeddings = self.config.get(
                "max_position_embeddings",
                2048 if base <= 10000.0 + 1e-5 else 16384
            )
            max_seq_len = max_position_embeddings * rope_scaling_factor

        # NTK
        try:
            ntk_alpha = float(os.environ.get("LIGHTLLM_NTK_ALPHA", 1))
            assert ntk_alpha >= 1
            if ntk_alpha > 1:
                logger.info(f"Note: NTK enabled, alpha set to {ntk_alpha}")
            max_seq_len *= ntk_alpha
            base = base * (ntk_alpha ** (self.head_dim_ / (self.head_dim_-2))) #Base change formula
        except:
            pass

        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim_, 2, device="cpu", dtype=torch.float32) / self.head_dim_))
        t = torch.arange(max_seq_len + 1024 * 64, device="cpu", dtype=torch.float32) / rope_scaling_factor
        freqs = torch.outer(t, inv_freq)

        self._cos_cached = torch.cos(freqs).to(torch.float16).cuda()
        self._sin_cached = torch.sin(freqs).to(torch.float16).cuda()
        return

    def _init_to_get_dynamic_ntk_rotary(self):
        max_position_embeddings = self.config.get("max_position_embeddings", 2048)
        base = self.config.get("rope_theta", 10000.0)
        scaling_factor = self.config.get("rope_scaling", {}).get("factor", 1.0)
        max_seq_len = 32 * max_position_embeddings # 64k
        self._cos_cached = torch.zeros((max_seq_len, self.head_dim_ // 2), dtype=torch.float16, device="cuda")
        self._sin_cached = torch.zeros((max_seq_len, self.head_dim_ // 2), dtype=torch.float16, device="cuda")
        
        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim_, 2, device="cpu", dtype=torch.float32) / self.head_dim_))
        t = torch.arange(max_position_embeddings, device="cpu", dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self._cos_cached[0:max_position_embeddings, :] = torch.cos(freqs).to(torch.float16).cuda()
        self._sin_cached[0:max_position_embeddings, :] = torch.sin(freqs).to(torch.float16).cuda()

        for seq_loc_index in range(max_position_embeddings, max_seq_len, 1):
            new_base = base * ((scaling_factor * (seq_loc_index + 1) / max_position_embeddings) -(scaling_factor - 1)) ** (self.head_dim_ / (self.head_dim_ - 2))
            inv_freq = 1.0 / (new_base ** (torch.arange(0, self.head_dim_, 2, device="cpu", dtype=torch.float32) / self.head_dim_))
            t = torch.tensor([seq_loc_index,], device="cpu", dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached[seq_loc_index:seq_loc_index + 1, :] = torch.cos(freqs).to(torch.float16).cuda()
            self._sin_cached[seq_loc_index:seq_loc_index + 1, :] = torch.sin(freqs).to(torch.float16).cuda()
        return

