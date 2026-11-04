import torch
import torch.distributed as dist

from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from lightllm.common.basemodel.triton_kernel.multimodal_emb import multimodal_emb
from lightllm.distributed.communication_op import all_reduce
from lightllm.core.runner.multimodal import MultimodalInputs


"""
infer_state.multimodal_params: batch list of MultimodalParams-dict like:
   {
       "images": [
           {
               "uuid": int,
               "token_id": int, image token start id,
               "token_num": int, image token num,
           },
       ]
       ...
   }
"""


class LlamaMultimodalPreLayerInfer(LlamaPreLayerInfer):
    def __init__(self, network_config):
        super().__init__(network_config)
        return

    def bind_inputs(self, layer_weight, cache):
        return MultimodalInputs.bind(layer_weight.wte_weight_.weight, cache).prepare

    def context_forward(self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight):
        device = layer_weight.wte_weight_.weight.device
        dtype = layer_weight.wte_weight_.weight.dtype
        hidden_size = layer_weight.wte_weight_.weight.shape[1]
        out = torch.zeros((len(input_ids), hidden_size), dtype=dtype, device=device)

        self._multimodal_emb(
            out=out,
            input_ids=input_ids,
            layer_weight=layer_weight,
            embed_cache=infer_state.cpu_embed_cache_tensor,
            img_token_lens=infer_state.img_token_lens,
            img_start_token_ids=infer_state.img_start_token_ids,
            img_start_locs_in_cache=infer_state.img_start_locs_in_cache,
        )
        if self.tp_world_size_ > 1:
            all_reduce(out, group=infer_state.dist_group, op=dist.ReduceOp.SUM, async_op=False)
        return out

    def _multimodal_emb(
        self,
        out: torch.Tensor,
        input_ids: torch.Tensor,
        layer_weight: LlamaPreAndPostLayerWeight,
        embed_cache: torch.Tensor,
        img_token_lens: torch.Tensor,
        img_start_token_ids: torch.Tensor,
        img_start_locs_in_cache: torch.Tensor,
    ) -> torch.Tensor:
        """
        方便子类继承修改多模态的embed计算的细节实现方式。
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
        )
        return
