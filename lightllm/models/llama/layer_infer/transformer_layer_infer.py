import torch
import triton
import torch.distributed as dist
from functools import partial
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.common.basemodel import TransformerLayerInferTpl
from lightllm.distributed.communication_op import all_gather_into_tensor, reduce_scatter_tensor
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class LlamaTransformerLayerInfer(TransformerLayerInferTpl):
    """ """

    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        self.eps_ = network_config["rms_norm_eps"]
        self.tp_q_head_num_ = network_config["num_attention_heads"] // self.tp_world_size_
        self.tp_k_head_num_ = max(network_config["num_key_value_heads"] // self.tp_world_size_, 1)
        self.tp_v_head_num_ = max(network_config["num_key_value_heads"] // self.tp_world_size_, 1)
        self.tp_o_head_num_ = self.tp_q_head_num_
        self.head_dim_ = network_config["hidden_size"] // network_config["num_attention_heads"]
        self.embed_dim_ = network_config["hidden_size"]
        self._bind_func()
        return

    def _bind_func(self):
        self._bind_norm()
        return

    def _bind_norm(self):
        self._att_norm = partial(LlamaTransformerLayerInfer._att_norm, self)
        self._ffn_norm = partial(LlamaTransformerLayerInfer._ffn_norm, self)
        return

    def _context_attention_kernel(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: LlamaTransformerLayerWeight,
    ) -> torch.Tensor:
        _k, _v = infer_state.mem_manager.get_att_input_params(layer_index=self.layer_num_)
        _q = q.view(-1, self.tp_q_head_num_, self.head_dim_)
        o_tensor = infer_state.prefill_att_state.prefill_att(
            q=_q,
            k=_k,
            v=_v,
            alloc_func=self.alloc_tensor,
        )
        o_tensor = o_tensor.view(q.shape)
        return o_tensor

    def _token_attention_kernel(
        self,
        q: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: LlamaTransformerLayerWeight,
    ) -> torch.Tensor:
        _k, _v = infer_state.mem_manager.get_att_input_params(layer_index=self.layer_num_)
        _q = q.view(-1, self.tp_q_head_num_, self.head_dim_)
        o_tensor = infer_state.decode_att_state.decode_att(q=_q, k=_k, v=_v, alloc_func=self.alloc_tensor)
        return o_tensor.view(q.shape)

    def _att_norm(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight
    ) -> torch.Tensor:
        return layer_weight.att_norm_weight_(input=input, eps=self.eps_, alloc_func=self.alloc_tensor)

    def _ffn_norm(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight
    ) -> torch.Tensor:
        return layer_weight.ffn_norm_weight_(input=input, eps=self.eps_, alloc_func=self.alloc_tensor)

    def _get_qkv(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight
    ) -> torch.Tensor:
        input = self._tpsp_allgather(input, infer_state)
        q = layer_weight.q_proj.mm(input)
        cache_kv = layer_weight.kv_proj.mm(input).view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)

        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, 0 : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
        )

        if infer_state.need_dp_prefill_balance:
            q = infer_state._all_to_all_unbalance_get(data=q)
            cache_kv = infer_state._all_to_all_unbalance_get(data=cache_kv)

        return q, cache_kv

    def _get_o(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight
    ) -> torch.Tensor:
        if infer_state.need_dp_prefill_balance:
            input = infer_state._all_to_all_balance_get(data=input)

        input = input.view(-1, self.tp_o_head_num_ * self.head_dim_)
        o_tensor = layer_weight.o_proj.mm(input)

        o_tensor = self._tpsp_reduce(input=o_tensor, infer_state=infer_state)
        return o_tensor

    def _ffn(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight) -> torch.Tensor:
        input = input.view(-1, self.embed_dim_)
        input = self._tpsp_allgather(input=input, infer_state=infer_state)
        ffn2_out = self._ffn_tp(input=input, infer_state=infer_state, layer_weight=layer_weight)
        ffn2_out = self._tpsp_reduce(input=ffn2_out, infer_state=infer_state)
        return ffn2_out

    def _ffn_tp(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight
    ) -> torch.Tensor:
        input = input.view(-1, self.embed_dim_)
        up_gate_out = layer_weight.gate_up_proj.mm(input)
        ffn1_out = self.alloc_tensor((input.size(0), up_gate_out.size(1) // 2), input.dtype)
        silu_and_mul_fwd(up_gate_out, ffn1_out)
        input = None
        up_gate_out = None
        ffn2_out = layer_weight.down_proj.mm(ffn1_out)
        ffn1_out = None
        return ffn2_out

    # # keep code
    # def _ffn(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight)->torch.Tensor:
    #     gate_up_out = torch.mm(input.view(-1, self.embed_dim_), layer_weight.gate_up_proj)
    #     size = gate_up_out.shape[1]
    #     gate_out, up_out = gate_up_out[:, 0: size // 2], gate_up_out[:, size // 2:]
    #     torch.nn.functional.silu(gate_out, inplace=True)
    #     gate_out.mul_(up_out)
    #     input = None
    #     ffn2_out = torch.mm(gate_out, layer_weight.down_proj)
    #     gate_out, up_out = None, None
    #     return ffn2_out

    def overlap_tpsp_token_forward(
        self,
        input_embdings: torch.Tensor,
        input_embdings1: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        infer_state1: LlamaInferStateInfo,
        layer_weight: LlamaTransformerLayerWeight,
    ):
        input_embdings = self.token_forward(input_embdings, infer_state, layer_weight=layer_weight)
        input_embdings1 = self.token_forward(input_embdings1, infer_state1, layer_weight=layer_weight)
        return input_embdings, input_embdings1

    def overlap_tpsp_context_forward(
        self,
        input_embdings: torch.Tensor,
        input_embdings1: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        infer_state1: LlamaInferStateInfo,
        layer_weight: LlamaTransformerLayerWeight,
    ):
        input_embdings = self.context_forward(input_embdings, infer_state, layer_weight=layer_weight)
        input_embdings1 = self.context_forward(input_embdings1, infer_state1, layer_weight=layer_weight)
        return input_embdings, input_embdings1
