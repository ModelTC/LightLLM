from lightllm.common.kv_cache_mem_manager.mem_manager import MemoryManager
from lightllm.common.kv_cache_mem_manager.operator import NormalMemOperator
from lightllm.common.speculative.config import normalize_speculative_draft_config
from lightllm.distributed.communication_op import dist_group_manager
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.qwen3_dflash.model import Qwen3DFlashModel
from lightllm.models.qwen3_5_dflash.layer_weights.pre_and_post_layer_weight import (
    Qwen35DFlashPreAndPostLayerWeight,
)


class Qwen35DFlashMemOperator(NormalMemOperator):
    """Map global draft layer indexes onto the draft-owned KV cache.

    Qwen3.5 target models use a Qwen3Next mixed linear/full-attention memory
    manager, which is not compatible with ordinary DFlash KV writes. The
    Qwen3 DFlash layer infer code is reused here and still passes global layer
    indexes, so this operator converts those indexes back to the local
    0..draft_layer_num-1 range used by the independent draft MemoryManager.
    """

    def _local_layer_index(self, layer_index: int) -> int:
        return int(layer_index) - int(self.mem_manager.layer_index_offset)

    def copy_kv_to_mem_manager(self, layer_index: int, mem_index, kv):
        return super().copy_kv_to_mem_manager(self._local_layer_index(layer_index), mem_index, kv)


class Qwen35DFlashMemoryManager(MemoryManager):
    """Ordinary KV cache for the Qwen3.5 DFlash draft model.

    Unlike Qwen3 DFlash, this draft cannot share the target model memory
    manager because the Qwen3.5 target cache contains Qwen3Next linear-attention
    state. The layer_index_offset records where the draft layers sit in the
    global model layer list, while the underlying cache stores only draft-local
    layer rows.
    """

    operator_class = Qwen35DFlashMemOperator

    def __init__(self, *args, layer_index_offset: int = 0, **kwargs):
        self.layer_index_offset = int(layer_index_offset)
        super().__init__(*args, **kwargs)

    def get_att_input_params(self, layer_index: int):
        return super().get_att_input_params(int(layer_index) - self.layer_index_offset)


class Qwen3_5DFlashModel(Qwen3DFlashModel):
    """Qwen3.5 DFlash draft model.

    The checkpoint stores DFlash parameters under `dflash_config` and omits
    token embedding / LM head weights. Those two weights are shared from the
    already-loaded target model. Its query/proposal layout is resolved from the
    checkpoint schema by the generic speculative runtime.
    """

    pre_and_post_weight_class = Qwen35DFlashPreAndPostLayerWeight

    def __init__(self, kvargs: dict):
        super().__init__(kvargs)
        # The request manager is shared with the target, while this draft owns
        # a separate KV cache. TpPartBaseModel wires every request manager to
        # the model's memory manager during construction, which would make
        # request allocation bypass the target/radix-cache allocator. Restore
        # the shared request manager's allocator after draft initialization.
        self._restore_main_mem_manager()
        return

    def _restore_main_mem_manager(self):
        assert self.req_manager is self.main_model.req_manager
        self.req_manager.mem_manager = self.main_model.mem_manager
        return

    def _init_config(self):
        super()._init_config()
        self._normalize_dflash_config()
        return

    def _normalize_dflash_config(self):
        normalize_speculative_draft_config(self.config)

        rope_parameters = self.config.get("rope_parameters")
        if isinstance(rope_parameters, dict):
            if "rope_theta" in rope_parameters and "rope_theta" not in self.config:
                self.config["rope_theta"] = rope_parameters["rope_theta"]
            if "partial_rotary_factor" in rope_parameters and "partial_rotary_factor" not in self.config:
                self.config["partial_rotary_factor"] = rope_parameters["partial_rotary_factor"]
            if "rope_scaling" not in self.config:
                self.config["rope_scaling"] = rope_parameters
        return

    def _init_custom(self):
        # The Qwen3.5 target uses mrope/partial rotary with a different head
        # shape from the ordinary DFlash draft. Build draft-owned rotary caches
        # from the draft config instead of reusing main_model._cos/_sin.
        LlamaTpPartModel._init_custom(self)
        self.dist_group = dist_group_manager.get_default_group()
        self.block_size = int(self.config["block_size"])
        self.mask_token_id = int(self.config["mask_token_id"])
        return

    def _init_mem_manager(self):
        head_dim = self.config.get("head_dim")
        if head_dim is None:
            head_dim = self.config["hidden_size"] // self.config["num_attention_heads"]
        num_kv_heads = max(1, self.config["num_key_value_heads"] // self.tp_world_size_)
        self.draft_layer_start = len(self.main_model.layers_infer)
        self.draft_layer_start += sum(
            len(previous_model.layers_infer) for previous_model in self.mtp_previous_draft_models
        )
        self.mem_manager = Qwen35DFlashMemoryManager(
            size=self.main_model.mem_manager.size,
            dtype=self.data_type,
            head_num=num_kv_heads,
            head_dim=head_dim,
            layer_num=self.config["n_layer"],
            mem_fraction=self.mem_fraction,
            layer_index_offset=self.draft_layer_start,
        )
        register_draft_manager = getattr(
            self.main_model.mem_manager, "register_dflash_draft_mem_manager", None
        )
        assert callable(register_draft_manager), "Qwen3_5DFlashModel requires a Qwen3Next target"
        register_draft_manager(
            self.mem_manager,
            global_kv_heads=int(self.config["num_key_value_heads"]),
        )
        return

    def _init_weights(self, start_layer_index=None):
        super()._init_weights(start_layer_index=start_layer_index)
        self.pre_post_weight.wte_weight_ = self.main_model.pre_post_weight.wte_weight_
        self.pre_post_weight.lm_head_weight_ = self.main_model.pre_post_weight.lm_head_weight_
        return
