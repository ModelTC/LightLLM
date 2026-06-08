import torch
import torch.distributed as dist
from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from lightllm.distributed.communication_op import all_reduce
from ..infer_struct import DeepseekV4InferStateInfo


class DeepseekV4PreLayerInfer(LlamaPreLayerInfer):
    """Token embedding, then expand to the hc_mult parallel residual streams [T, hc_mult*hidden]."""

    def _embed_and_expand(self, input_ids, infer_state: DeepseekV4InferStateInfo, layer_weight):
        emb = layer_weight.wte_weight_(input_ids=input_ids, alloc_func=self.alloc_tensor)  # [T, hidden]
        if self.tp_world_size_ > 1:
            all_reduce(emb, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        hc_mult = layer_weight.network_config_["hc_mult"]
        t, hidden = emb.shape
        return emb.unsqueeze(1).expand(t, hc_mult, hidden).reshape(t, hc_mult * hidden).contiguous()

    def context_forward(self, input_ids, infer_state: DeepseekV4InferStateInfo, layer_weight):
        return self._embed_and_expand(input_ids, infer_state, layer_weight)

    def token_forward(self, input_ids, infer_state: DeepseekV4InferStateInfo, layer_weight):
        return self._embed_and_expand(input_ids, infer_state, layer_weight)
