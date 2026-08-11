from lightllm.common.basemodel.attention import (
    Fa3AttBackend,
    Fp8Fa3AttBackend,
)
from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.registry import DraftModelRegistry
from lightllm.models.qwen3_dflash.infer_struct import Qwen3DFlashInferStateInfo
from lightllm.models.qwen3_dflash.layer_infer.post_layer_infer import Qwen3DFlashPostLayerInfer
from lightllm.models.qwen3_dflash.layer_infer.pre_layer_infer import Qwen3DFlashPreLayerInfer
from lightllm.models.qwen3_dflash.layer_infer.transformer_layer_infer import Qwen3DFlashTransformerLayerInfer
from lightllm.models.qwen3_dflash.layer_weights.pre_and_post_layer_weight import Qwen3DFlashPreAndPostLayerWeight
from lightllm.models.qwen3_dflash.layer_weights.transformer_layer_weight import Qwen3DFlashTransformerLayerWeight


@DraftModelRegistry(model_type="qwen3", spec_modes="dflash")
class Qwen3DFlashModel(LlamaTpPartModel):
    """Qwen3 DFlash draft model."""

    is_mtp_draft_model = True

    pre_and_post_weight_class = Qwen3DFlashPreAndPostLayerWeight
    transformer_weight_class = Qwen3DFlashTransformerLayerWeight
    pre_layer_infer_class = Qwen3DFlashPreLayerInfer
    post_layer_infer_class = Qwen3DFlashPostLayerInfer
    transformer_layer_infer_class = Qwen3DFlashTransformerLayerInfer
    infer_state_class = Qwen3DFlashInferStateInfo

    def __init__(self, kvargs: dict):
        self._pre_init(kvargs)
        super().__init__(kvargs)

    def _pre_init(self, kvargs: dict):
        self.main_model: TpPartBaseModel = kvargs.pop("main_model")
        self.mtp_previous_draft_models = kvargs.pop("mtp_previous_draft_models")

    def _verify_params(self):
        super()._verify_params()
        assert not self.enable_tpsp_mix_mode, "Qwen3 DFlash draft model does not support TP-SP"

    def _init_custom(self):
        self._cos_cached = self.main_model._cos_cached
        self._sin_cached = self.main_model._sin_cached
        self.block_size = int(self.config["block_size"])
        self.mask_token_id = int(self.config["mask_token_id"])

    def _init_req_manager(self):
        self.req_manager = self.main_model.req_manager

    def _init_mem_manager(self):
        # Draft KV uses target-owned cache slots.
        self.mem_manager = self.main_model.mem_manager

    def _init_att_backend(self):
        super()._init_att_backend()
        # FA3 is currently the only backend that honors decode_causal=False.
        # TODO: Remove this restriction after Triton and FlashInfer support non-causal block decode.
        if not isinstance(self.decode_att_backend, (Fa3AttBackend, Fp8Fa3AttBackend)):
            raise NotImplementedError("Qwen3 DFlash decode requires FA3")

    def _init_infer_layer(self, start_layer_index=None):
        assert start_layer_index is None
        self.draft_layer_start = len(self.main_model.layers_infer)
        self.draft_layer_start += sum(
            len(previous_model.layers_infer) for previous_model in self.mtp_previous_draft_models
        )
        super()._init_infer_layer(start_layer_index=self.draft_layer_start)

    def _init_weights(self, start_layer_index=None):
        assert start_layer_index is None
        self.pre_post_weight = self.pre_and_post_weight_class(
            self.data_type,
            network_config=self.config,
            quant_cfg=self.quant_cfg,
        )
        self.trans_layers_weight = [
            self.transformer_weight_class(
                i,
                self.data_type,
                network_config=self.config,
                quant_cfg=self.quant_cfg,
            )
            for i in range(self.config["n_layer"])
        ]

    def _gen_special_model_input(self, token_num: int):
        return {"mtp_draft_input_hiddens": None}

    def _autotune_warmup(self):
        return

    def _init_padded_req(self):
        return

    def _init_prefill_cuda_graph(self):
        self.prefill_graph = None
