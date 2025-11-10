from lightllm.models.registry import ModelRegistry
from lightllm.models.deepseek2.model import Deepseek2TpPartModel
from lightllm.models.deepseek3_2.layer_weights.transformer_layer_weight import Deepseek3_2TransformerLayerWeight
from lightllm.models.deepseek3_2.layer_infer.transformer_layer_infer import Deepseek3_2TransformerLayerInfer
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.models.deepseek3_2.infer_struct import Deepseek3_2FlashAttentionStateInfo
from lightllm.models.deepseek3_2.mem_manager import Deepseek3_2MemoryManager, Deepseek3_2FP8KVMemoryManager
@ModelRegistry(["deepseek_v32"])
class Deepseek3_2TpPartModel(Deepseek2TpPartModel):
    # weight class
    transformer_weight_class = Deepseek3_2TransformerLayerWeight

    # infer class
    transformer_layer_infer_class = Deepseek3_2TransformerLayerInfer

    # infer state class
    infer_state_class = Deepseek3_2FlashAttentionStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
        self.index_topk = self.config["index_topk"]
        return

    def _init_inferstate_cls(self):
        self.infer_state_class = Deepseek3_2FlashAttentionStateInfo

    def _init_mem_manager(self):
        manager_class = Deepseek3_2MemoryManager
        if "triton_fp8kv" in self.mode:
            manager_class = Deepseek3_2FP8KVMemoryManager

        # mtp 模式下需要在mem manger上扩展draft model使用的layer
        added_mtp_layer_num = 0
        if get_env_start_args().mtp_mode == "deepseekv3_eagle":
            added_mtp_layer_num += 1
        elif get_env_start_args().mtp_mode == "deepseekv3_vanilla":
            added_mtp_layer_num += get_env_start_args().mtp_step

        self.mem_manager = manager_class(
            self.max_total_token_num,
            dtype=self.data_type,
            head_num=1,
            head_dim=self.config["kv_lora_rank"] + self.config["qk_rope_head_dim"],
            layer_num=self.config["num_hidden_layers"] + added_mtp_layer_num,
            mem_fraction=self.mem_fraction,
        )
        return