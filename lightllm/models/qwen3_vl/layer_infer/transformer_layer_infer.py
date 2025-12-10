import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from functools import partial
from typing import Tuple
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.qwen2_vl.triton_kernel.mrope import mrope_triton
from lightllm.models.qwen3.layer_infer.transformer_layer_infer import Qwen3TransformerLayerInfer
from lightllm.models.qwen3.layer_weights.transformer_layer_weight import Qwen3TransformerLayerWeight
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.qwen3_vl.infer_struct import Qwen3VLInferStateInfo
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.llama.triton_kernel.silu_and_mul import silu_and_mul_fwd
from lightllm.distributed import all_reduce
from lightllm.utils.dist_utils import get_global_world_size


class Qwen3VLTransformerLayerInfer(Qwen3TransformerLayerInfer):
    def __init__(self, layer_num, network_config, mode=[]):
        super().__init__(layer_num, network_config, mode)
        self.mrope_section = network_config["rope_scaling"]["mrope_section"]
        axis_map = []
        for i, n in enumerate(self.mrope_section * 2):
            axis_map += [i % 3] * n
        self.axis_map = torch.tensor(axis_map, dtype=torch.int32, device="cuda")

    def context_forward(self, input_embdings, infer_state: Qwen3VLInferStateInfo, layer_weight):
        input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        q, cache_kv = self._get_qkv(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._context_attention_kernel(q, cache_kv, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.tp_world_size_ > 1:
            all_reduce(o, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(o.view(-1, self.embed_dim_))
        o = None

        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        if self.tp_world_size_ > 1:
            all_reduce(ffn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        if infer_state.deepstack_features:
            for i in range(len(infer_state.img_first_token_locs)):
                start = infer_state.img_first_token_locs[i]
                end = infer_state.img_last_token_locs[i]
                deepstack_features = infer_state.deepstack_features[i]
                if end <= input_embdings.shape[0] and self.layer_num_ in range(len(deepstack_features)):
                    deepstack_features_cur_layer = deepstack_features[self.layer_num_].to(
                        device=input_embdings.device, non_blocking=True
                    )
                    print(
                        f"self.layer_num_ is {self.layer_num_}, i is{i} ,"
                        f"deepstack_features_cur_layer is {deepstack_features_cur_layer}"
                    )
                    input_embdings[
                        start:end,
                    ].add_(deepstack_features_cur_layer)
            if self.layer_num_ == len(deepstack_features):
                infer_state.img_first_token_locs = []
                infer_state.img_last_token_locs = []
                infer_state.deepstack_features = []
        return input_embdings
