import torch
from lightllm.models.llama2.model import Llama2TpPartModel
from lightllm.common.ppl_int8kv_mem_manager import PPLINT8KVMemoryManager
from lightllm.models.llama2_ppl.layer_infer.transformer_layer_infer import Llama2PPlTransformerLayerInfer

class Llama2PPlTpPartModel(Llama2TpPartModel):

    transformer_layer_infer_class = Llama2PPlTransformerLayerInfer

    memory_manager_class = PPLINT8KVMemoryManager

    def __init__(self, tp_rank, world_size, weight_dir, max_total_token_num, load_way="HF", mode=[]):
        super().__init__(tp_rank, world_size, weight_dir, max_total_token_num, load_way, mode)
        return
    
    def _verify_params(self):
        assert self.load_way == "HF", "llama only support HF format to load Now!"
        assert self.config["num_key_value_heads"] % self.world_size_ == 0
        assert self.config["num_attention_heads"] % self.world_size_ == 0
        assert "int8kv" in self.mode, "ppl llama2 only support int8kv mode"
        return

    def _init_mem_manager(self):
        self.mem_manager = self.memory_manager_class(self.max_total_token_num, 
                                                     dtype=torch.float16,
                                                     head_num=self.config["num_key_value_heads"] // self.world_size_,
                                                     head_dim=self.config["hidden_size"] // self.config["num_attention_heads"],
                                                     layer_num=self.config["num_hidden_layers"])
        return
