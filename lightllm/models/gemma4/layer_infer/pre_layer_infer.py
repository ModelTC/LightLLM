import math
import torch
import torch.distributed as dist
from lightllm.distributed.communication_op import all_reduce
from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.common.basemodel.triton_kernel.multimodal_emb import multimodal_emb


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
        self.ple_static_buffer = None

    def _compute_per_layer_embeds(self, input_ids_for_ple, input_embdings, infer_state, layer_weight):
        # 查表 PLE。
        ple_embeds = layer_weight.embed_tokens_per_layer_weight_(input_ids_for_ple)
        if self.tp_world_size_ > 1:
            all_reduce(ple_embeds, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        ple_embeds = ple_embeds * self.ple_embed_scale_

        # 这个分支本质上只对多模态token存在建模上的意义。
        ple_proj = layer_weight.per_layer_model_projection_weight_.mm(input_embdings)
        ple_proj = ple_proj * self.ple_proj_scale_
        ple_proj = ple_proj.reshape(*ple_proj.shape[:-1], self.num_layers_, self.ple_dim_)
        ple_proj = layer_weight.per_layer_projection_norm_weight_(
            input=ple_proj, eps=self.rms_norm_eps_, alloc_func=self.alloc_tensor
        )
        ple_embeds = ple_embeds.reshape(*ple_embeds.shape[:-1], self.num_layers_, self.ple_dim_)

        handle_len = input_embdings.shape[0]
        torch.add(ple_proj, ple_embeds, out=self.ple_static_buffer[:handle_len])
        self.ple_static_buffer[:handle_len].mul_(self.ple_combine_scale_)
        return

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
            # SP would need a per-rank slice (N/world_size tokens), but the
            # PLE static buffer is sized/written for the full N tokens. If you
            # ever need SP + PLE, refactor _compute_per_layer_embeds to do an
            # sp_pad_copy into a per-rank buffer.
            assert not self.has_ple, "gemma4 PLE + enable_tpsp_mix_mode not implemented"
            return super()._tpsp_sp_split(input=input, infer_state=infer_state)
        return input

    def _multimodal_emb(
        self,
        out: torch.Tensor,
        input_ids: torch.Tensor,
        layer_weight,
        embed_cache: torch.Tensor,
        img_token_lens: torch.Tensor,
        img_start_token_ids: torch.Tensor,
        img_start_locs_in_cache: torch.Tensor,
    ) -> torch.Tensor:
        """
        修改多模态的 embed 计算的细节实现方式,调用本地的 multimodal_text_embed_scale_ 参数。
        """
        multimodal_emb(
            out=out,
            prompt_ids=input_ids,
            text_weight_embs=layer_weight.wte_weight_.weight,
            embed_cache=embed_cache,
            img_token_lens=img_token_lens,
            img_start_token_ids=img_start_token_ids,
            img_start_locs_in_cache=img_start_locs_in_cache,
            tp_text_start_token_id=layer_weight.wte_weight_.tp_vocab_start_id,
            tp_text_end_token_id=layer_weight.wte_weight_.tp_vocab_end_id,
            tp_world_size=self.tp_world_size_,
            text_embed_scale=self.multimodal_text_embed_scale_,
        )
        return
