from lightllm.common.basemodel.layer_weights.hf_load_utils import load_hf_weights
from lightllm.common.kv_cache_mem_manager.draft_mem_manager import DraftMemoryManagerView
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.qwen3_dflash.model import Qwen3DFlashModel
from lightllm.models.qwen3_dspark.layer_infer.post_layer_infer import Qwen3DSparkPostLayerInfer
from lightllm.models.qwen3_dspark.layer_weights.pre_and_post_layer_weight import Qwen3DSparkPreAndPostLayerWeight
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class Qwen3DSparkModel(Qwen3DFlashModel):
    """Qwen3 DSpark draft model.

    DSpark keeps the DFlash block backbone.  Its extra Markov/confidence heads
    run in the post layer so the proposer can consume corrected logits through
    the same token-generation path as DFlash.
    """

    pre_and_post_weight_class = Qwen3DSparkPreAndPostLayerWeight
    post_layer_infer_class = Qwen3DSparkPostLayerInfer

    def __init__(self, kvargs: dict):
        super().__init__(kvargs)
        # The draft may create a geometry view over target storage.  Keep the
        # target manager as the request manager's shared owner.
        self.main_model.req_manager.mem_manager = self.main_model.mem_manager

    def _verify_params(self):
        self.config.setdefault("block_size", self.args.mtp_step)
        super()._verify_params()

    def _init_custom(self):
        # DSpark has narrower heads than the target and must own its RoPE cache.
        LlamaTpPartModel._init_custom(self)
        self.block_size = int(self.config["block_size"])
        self.mask_token_id = int(self.config["mask_token_id"])

    def _init_mem_manager(self):
        owner = self.main_model.mem_manager
        head_num = max(self.config["num_key_value_heads"] // self.tp_world_size_, 1)
        head_dim = self.config["head_dim"]
        if head_num == owner.head_num and head_dim == owner.head_dim:
            self.mem_manager = owner
        else:
            self.mem_manager = DraftMemoryManagerView(owner, head_num=head_num, head_dim=head_dim)

    def load_weights(self, weight_dict: dict):
        assert weight_dict is None or isinstance(weight_dict, dict), "weight_dict must be a dict or None"
        load_hf_weights(
            data_type=self.data_type,
            weight_dir=self.weight_dir_,
            pre_post_layer=self.pre_post_weight,
            transformer_layer_list=self.trans_layers_weight,
            weight_dict=weight_dict,
        )
        draft_lm_head = self.pre_post_weight.lm_head_weight_
        if not draft_lm_head.verify_load():
            target_lm_head = self.main_model.pre_post_weight.lm_head_weight_
            compatible = (
                target_lm_head.verify_load()
                and target_lm_head.dim == draft_lm_head.dim
                and target_lm_head.vocab_size == draft_lm_head.vocab_size
                and target_lm_head.tp_rank_ == draft_lm_head.tp_rank_
                and target_lm_head.tp_world_size_ == draft_lm_head.tp_world_size_
                and target_lm_head.tp_vocab_start_id == draft_lm_head.tp_vocab_start_id
                and target_lm_head.tp_vocab_end_id == draft_lm_head.tp_vocab_end_id
            )
            if not compatible:
                raise ValueError("DSpark checkpoint lacks lm_head.weight and target LM head is incompatible")
            logger.info("DSpark checkpoint lacks lm_head.weight; reusing the compatible target LM head")
            self.pre_post_weight.lm_head_weight_ = target_lm_head
        self.pre_post_weight.verify_load()
        [weight.verify_load() for weight in self.trans_layers_weight]
