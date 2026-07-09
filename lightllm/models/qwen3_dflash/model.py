import torch

from lightllm.common.basemodel.attention import (
    BaseAttBackend,
    Fa3AttBackend,
    Fp8Fa3AttBackend,
    get_decode_att_backend_class,
    get_prefill_att_backend_class,
)
from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.cuda_graph import CudaGraph
from lightllm.distributed.communication_op import dist_group_manager
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.qwen3_dflash.infer_struct import Qwen3DFlashInferStateInfo
from lightllm.models.qwen3_dflash.layer_infer.post_layer_infer import Qwen3DFlashPostLayerInfer
from lightllm.models.qwen3_dflash.layer_infer.pre_layer_infer import Qwen3DFlashPreLayerInfer
from lightllm.models.qwen3_dflash.layer_infer.transformer_layer_infer import Qwen3DFlashTransformerLayerInfer
from lightllm.models.qwen3_dflash.layer_weights.pre_and_post_layer_weight import Qwen3DFlashPreAndPostLayerWeight
from lightllm.models.qwen3_dflash.layer_weights.transformer_layer_weight import Qwen3DFlashTransformerLayerWeight


class Qwen3DFlashModel(LlamaTpPartModel):
    """Qwen3 DFlash draft model.

    This is the LightLLM service port of the DeepSpec DFlash/DSpark Qwen3 model.
    The service path enters through `forward(ModelInput)` using the same
    primitive metadata as normal LightLLM prefill: `mem_indexes`,
    `req_to_token_indexs`, sequence lengths, and prefill start locations.

    Target -> draft inputs:
    - target hidden rows are committed into DFlash draft K/V.
    - this model then materializes the next DFlash block from query/mask
      embeddings and scratch KV slots.

    Draft output:
    - logits are returned for the flattened [batch, draft_step] block rows.
      The proposer maps them back to the standard
      [verify_batch, draft_step + 1] speculative proposal shape.

    KV ownership:
    - DFlash intentionally reuses `main_model.req_manager` and
      `main_model.mem_manager` when the target/draft KV shape is compatible.
      The target manager can be over-provisioned with draft layer slots and
      extra token capacity so CPU cache, offload, and free logic remain unified.
    - The DFlash-specific invariant is the layer/token-slot lifecycle: accepted
      target hidden rows become committed draft K/V, while current block K/V is
      scratch and must be released or overwritten after proposal/verification.
    """

    pre_and_post_weight_class = Qwen3DFlashPreAndPostLayerWeight
    transformer_weight_class = Qwen3DFlashTransformerLayerWeight
    pre_layer_infer_class = Qwen3DFlashPreLayerInfer
    post_layer_infer_class = Qwen3DFlashPostLayerInfer
    transformer_layer_infer_class = Qwen3DFlashTransformerLayerInfer
    infer_state_class = Qwen3DFlashInferStateInfo

    def __init__(self, kvargs: dict):
        self._pre_init(kvargs)
        super().__init__(kvargs)
        return

    def _pre_init(self, kvargs: dict):
        self.main_model: TpPartBaseModel = kvargs.pop("main_model")
        self.mtp_previous_draft_models = kvargs.pop("mtp_previous_draft_models")
        kvargs["return_all_prompt_logics"] = True
        return

    def _init_custom(self):
        self._cos_cached = self.main_model._cos_cached
        self._sin_cached = self.main_model._sin_cached
        self.dist_group = dist_group_manager.get_default_group()
        self.block_size = int(self.config["block_size"])
        self.mask_token_id = int(self.config["mask_token_id"])
        return

    def _init_req_manager(self):
        self.req_manager = self.main_model.req_manager
        return

    def _init_mem_manager(self):
        # Intentionally shared with the target model.  DFlash uses compatible
        # KV shapes, so the main manager should be provisioned with the draft
        # layer range and any extra temporary token capacity needed by block
        # proposals.  This keeps req/cpu-cache/offload/free paths unified.
        self.mem_manager = self.main_model.mem_manager
        return

    def _init_att_backend(self):
        self.prefill_att_backend: BaseAttBackend = get_prefill_att_backend_class(index=0)(model=self)
        try:
            self.decode_att_backend: BaseAttBackend = get_decode_att_backend_class(
                index=0,
                priority_list=["fa3"],
            )(model=self)
        except KeyError as exc:
            raise NotImplementedError(
                "Qwen3DFlashModel requires FA3 decode attention: "
                "block draft attention is non-causal and Triton/FlashInfer decode paths do not honor decode_causal."
            ) from exc
        if not isinstance(self.decode_att_backend, (Fa3AttBackend, Fp8Fa3AttBackend)):
            raise NotImplementedError(
                "Qwen3DFlashModel requires FA3 decode attention: "
                "block draft attention is non-causal and Triton/FlashInfer decode paths do not honor decode_causal."
            )
        return

    def _init_infer_layer(self, start_layer_index=None):
        assert start_layer_index is None
        self.draft_layer_start = len(self.main_model.layers_infer)
        self.draft_layer_start += sum(
            len(previous_model.layers_infer) for previous_model in self.mtp_previous_draft_models
        )
        super()._init_infer_layer(start_layer_index=self.draft_layer_start)
        return

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
        return

    def _gen_special_model_input(self, token_num: int):
        return {"mtp_draft_input_hiddens": None}

    def _autotune_warmup(self):
        return

    def _init_padded_req(self):
        return

    def _init_cudagraph(self):
        if self.disable_cudagraph or self.args.enable_decode_microbatch_overlap:
            self.graph = None
            return

        self.graph = CudaGraph(
            max_batch_size=self.graph_max_batch_size,
            max_len_in_batch=self.graph_max_len_in_batch,
            tp_world_size=self.tp_world_size_,
        )
        return

    def _init_prefill_cuda_graph(self):
        self.prefill_graph = None
        return

    def _check_max_len_infer(self):
        return
