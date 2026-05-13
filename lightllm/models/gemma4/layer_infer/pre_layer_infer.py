import math
import torch
import torch.distributed as dist
from lightllm.common.basemodel.triton_kernel.sp_pad_copy import sp_pad_copy
from lightllm.distributed.communication_op import all_reduce
from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer
from lightllm.utils.envs_utils import get_env_start_args


class Gemma4PreLayerInfer(LlamaMultimodalPreLayerInfer):
    def __init__(self, network_config):
        super().__init__(network_config)
        self.embed_scale = float(network_config["hidden_size"]) ** 0.5
        self.multimodal_text_embed_scale_ = self.embed_scale
        self.pad_token_id_ = network_config.get("pad_token_id", 0)

        self.has_ple = bool(network_config.get("hidden_size_per_layer_input"))
        if self.has_ple:
            self.num_layers_ = network_config["num_hidden_layers"]
            self.ple_dim_ = network_config["hidden_size_per_layer_input"]
            self.ple_embed_scale_ = math.sqrt(self.ple_dim_)
            self.ple_proj_scale_ = float(network_config["hidden_size"]) ** -0.5
            self.ple_combine_scale_ = 2.0 ** -0.5
            self.rms_norm_eps_ = network_config.get("rms_norm_eps", 1e-6)

    def _compute_per_layer_embeds(self, input_ids_for_ple, input_embdings, infer_state, layer_weight):
        ple_embeds = layer_weight.embed_tokens_per_layer_weight_(input_ids_for_ple)
        if self.tp_world_size_ > 1:
            all_reduce(ple_embeds, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        ple_embeds = ple_embeds * self.ple_embed_scale_

        ple_proj = layer_weight.per_layer_model_projection_weight_.mm(input_embdings)
        ple_proj = ple_proj * self.ple_proj_scale_
        ple_proj = ple_proj.reshape(*ple_proj.shape[:-1], self.num_layers_, self.ple_dim_)
        ple_proj = layer_weight.per_layer_projection_norm_weight_(
            input=ple_proj, eps=self.rms_norm_eps_, alloc_func=self.alloc_tensor
        )

        ple_embeds = ple_embeds.reshape(*ple_embeds.shape[:-1], self.num_layers_, self.ple_dim_)
        infer_state.per_layer_embeds = (ple_proj + ple_embeds) * self.ple_combine_scale_

    def context_forward(self, input_ids, infer_state, layer_weight):
        input_embdings = LlamaMultimodalPreLayerInfer.context_forward(self, input_ids, infer_state, layer_weight)
        if self.has_ple:
            input_ids_for_ple = input_ids.masked_fill(infer_state.b_image_token_end != 0, self.pad_token_id_)
            self._compute_per_layer_embeds(input_ids_for_ple, input_embdings, infer_state, layer_weight)
        return input_embdings

    def token_forward(self, input_ids, infer_state, layer_weight):
        input_embdings = LlamaPreLayerInfer.token_forward(self, input_ids, infer_state, layer_weight)
        input_embdings = input_embdings * self.embed_scale
        if self.has_ple:
            self._compute_per_layer_embeds(input_ids, input_embdings, infer_state, layer_weight)
        return input_embdings

    def _tpsp_sp_split(self, input: torch.Tensor, infer_state):
        if self.tp_world_size_ > 1 and get_env_start_args().enable_tpsp_mix_mode:
            input = super()._tpsp_sp_split(input=input, infer_state=infer_state)
            if self.has_ple and infer_state.per_layer_embeds is not None:
                ple_shape = infer_state.per_layer_embeds.shape
                per_layer_embeds = infer_state.per_layer_embeds.reshape(ple_shape[0], -1)
                per_layer_embeds = sp_pad_copy(
                    in_tensor=per_layer_embeds,
                    sp_rank_id=self.tp_rank_,
                    sp_world_size=self.tp_world_size_,
                    alloc_func=self.alloc_tensor,
                )
                infer_state.per_layer_embeds = per_layer_embeds.reshape(
                    per_layer_embeds.shape[0], ple_shape[1], ple_shape[2]
                )
            return input
        return input
