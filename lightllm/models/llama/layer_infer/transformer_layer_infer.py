import torch
import triton
import torch.distributed as dist
from functools import partial
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.common.basemodel import TransformerLayerInferTpl
from lightllm.distributed.communication_op import all_gather_into_tensor, reduce_scatter_tensor
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def npu_silu_and_mul_fwd(
    input: torch.Tensor,
    layout="blocked",
    limit=None,
    alpha=None,
) -> torch.Tensor:
    assert input.is_contiguous()
    assert input.dim() == 2
    assert (limit is None and alpha is None) or (limit is not None and alpha is not None)
    N = input.shape[1] // 2

    if layout == "blocked":
        gate = input[:, :N]
        up   = input[:, N:]
    elif layout == "interleaved":
        gate = input[:, 0::2]
        up   = input[:, 1::2]
    else:
        raise ValueError(f"unknown layout: {layout}")

    if limit is not None and alpha is not None:
        gate_fp32_limit = torch.minimum(
            gate.float(),
            torch.tensor(limit, device=gate.device, dtype=torch.float32),
        )

        gate_act = torch.sigmoid(gate_fp32_limit * alpha) * gate_fp32_limit
        gate_act = gate_act.to(input.dtype)

        up_clip = torch.clamp(up, -limit, limit)
        out = (up_clip + 1) * gate_act
    else:
        import torch_npu

        out = torch_npu.npu_swiglu(input, dim=-1)

    return out


def npu_ffn_fwd(
    input: torch.Tensor,
    layer_weight: LlamaTransformerLayerWeight,
    embed_dim: int,
) -> torch.Tensor:
    import torch.nn.functional as F

    input = input.view(-1, embed_dim)
    # up
    gate_up_proj_bias = [layer_weight.gate_up_proj.bias] if layer_weight.gate_up_proj.bias is not None else None
    weight = layer_weight.gate_up_proj.mm_param.weight
    # up_gate_out = torch_npu.npu_grouped_matmul(
    #     x=[input],
    #     weight=[weight],
    #     bias=gate_up_proj_bias,
    #     split_item=0,
    #     group_type=-1,
    #     group_list=None,
    # )[0]
    up_gate_out = F.linear(input, weight, bias=gate_up_proj_bias)

    # activation
    ffn1_out = npu_silu_and_mul_fwd(up_gate_out)

    # down
    down_proj_bias = [layer_weight.down_proj.bias] if layer_weight.down_proj.bias is not None else None
    weight = layer_weight.down_proj.mm_param.weight
    # ffn2_out = torch_npu.npu_grouped_matmul(
    #     x=[ffn1_out],
    #     weight=[weight],
    #     bias=down_proj_bias,
    #     split_item=0,
    #     group_type=-1,
    #     group_list=None,
    # )[0]
    ffn2_out = F.linear(ffn1_out, weight, bias=down_proj_bias)

    return ffn2_out


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

        self.platform_backend.ops.infer.rotary_emb(
            is_prefill=infer_state.is_prefill,
            batch_size=infer_state.batch_size,
            q=q.view(-1, self.tp_q_head_num_, self.head_dim_),
            k=cache_kv[:, 0 : self.tp_k_head_num_, :],
            cos=infer_state.position_cos,
            sin=infer_state.position_sin,
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
        return self._tpsp_reduce(input=ffn2_out, infer_state=infer_state)

    def _ffn_tp(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight
    ) -> torch.Tensor:
        return self.platform_backend.ops.infer.ffn(
            input=input,
            layer_weight=layer_weight,
            alloc_func=self.alloc_tensor,
            embed_dim=self.embed_dim_,
        )

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
