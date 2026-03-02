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
        q = layer_weight.q_proj.mm(input)
        cache_kv = layer_weight.kv_proj.mm(input).view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)

        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, 0 : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
        )
        return q, cache_kv

    def _tpsp_get_qkv(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight
    ) -> torch.Tensor:
        if self.tp_world_size_ > 1:
            sp_token_num, hidden_dim = input.shape
            gather_input = self.alloc_tensor(
                (sp_token_num * self.tp_world_size_, hidden_dim), dtype=input.dtype, device=input.device
            )
            all_gather_into_tensor(gather_input, input, group=infer_state.dist_group, async_op=False)
            input = gather_input[0 : len(infer_state.input_ids), :]

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
        input = input.view(-1, self.tp_o_head_num_ * self.head_dim_)
        o_tensor = layer_weight.o_proj.mm(input)
        return o_tensor

    def _tpsp_get_o(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight
    ) -> torch.Tensor:
        if infer_state.need_dp_prefill_balance:
            input = infer_state._all_to_all_balance_get(data=input)

        input = input.view(-1, self.tp_o_head_num_ * self.head_dim_)
        dest_size = triton.cdiv(input.shape[0], self.tp_world_size_) * self.tp_world_size_
        o_tensor = self.alloc_tensor((dest_size, self.embed_dim_), dtype=input.dtype, device=input.device)
        layer_weight.o_proj.mm(input, out=o_tensor[0 : len(infer_state.input_ids), :])
        e_o_tensor = o_tensor[len(infer_state.input_ids) :, :]
        if e_o_tensor.shape[0] > 0:
            e_o_tensor.fill_(0)

        if self.tp_world_size_ > 1:
            sp_token_num = o_tensor.shape[0] // self.tp_world_size_
            reduce_o_tensor = self.alloc_tensor((sp_token_num, self.embed_dim_), dtype=input.dtype, device=input.device)
            reduce_scatter_tensor(
                output=reduce_o_tensor,
                input=o_tensor,
                op=dist.ReduceOp.SUM,
                group=infer_state.dist_group,
                async_op=False,
            )
            o_tensor = reduce_o_tensor

        return o_tensor

    def _ffn(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight) -> torch.Tensor:
        input = input.view(-1, self.embed_dim_)
        up_gate_out = layer_weight.gate_up_proj.mm(input)
        ffn1_out = self.alloc_tensor((input.size(0), up_gate_out.size(1) // 2), input.dtype)
        silu_and_mul_fwd(up_gate_out, ffn1_out)

        if not torch.cuda.is_current_stream_capturing() and not infer_state.is_prefill:
            from lightllm.utils.wandb_utils import get_wandb_run
            import numpy as np
            import wandb

            # run = get_wandb_run()
            a = ffn1_out.float().detach().cpu().abs().flatten()
            split_params = [0, 0.5, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00]
            fens = list(np.quantile(a, split_params))
            fens = [float(fen) for fen in fens]
            if not hasattr(infer_state, "ffn_act_table"):
                cols_name = ["layer_index"] + [f"ffn1_up_key_value_{int(sp*100)}" for sp in split_params]

                infer_state.ffn_act_table = wandb.Table(columns=cols_name)
            fens = [int(self.layer_num_)] + fens
            infer_state.ffn_act_table.add_data(*fens)

            up_gate_weight_norm = (
                torch.norm(layer_weight.gate_up_proj.mm_param.weight, dim=-1).float().detach().cpu().numpy()
            )
            down_weight_norm = torch.norm(layer_weight.down_proj.mm_param.weight, dim=-1).float().detach().cpu().numpy()

            if not hasattr(infer_state, "ffn_weight_table"):
                infer_state.ffn_weight_table = wandb.Table(
                    columns=[
                        "layer_index",
                        "gate_weight_max",
                        "gate_weight_mean",
                        "gate_weight_min",
                        "up_weight_max",
                        "up_weight_mean",
                        "up_weight_min",
                        "ffn_down_weight_max",
                        "ffn_down_weight_mean",
                        "ffn_down_weight_min",
                    ]
                )
            gate_weight_norm = up_gate_weight_norm[0 : up_gate_weight_norm.shape[0] // 2]
            up_weight_norm = up_gate_weight_norm[up_gate_weight_norm.shape[0] // 2 :]

            infer_state.ffn_weight_table.add_data(
                int(self.layer_num_),
                float(gate_weight_norm.max()),
                float(gate_weight_norm.mean()),
                float(gate_weight_norm.min()),
                float(up_weight_norm.max()),
                float(up_weight_norm.mean()),
                float(up_weight_norm.min()),
                float(down_weight_norm.max()),
                float(down_weight_norm.mean()),
                float(down_weight_norm.min()),
            )
            energy_thresholds = [0.3, 0.5, 0.7, 0.8, 0.9]
            if not hasattr(infer_state, "ffn_rms_norm_table"):
                cols_name = [
                    "layer_index",
                    "ffn_rms_abs_sum",
                    "ffn_input_rms_norm",
                    "ffn_input_rms_abs_mean",
                    "ffn_input_rms_abs_max",
                    "ffn_input_rms_abs_min",
                ]
                cols_name += [f"ffn_rms_norm_energy{int(t*100)}" for t in energy_thresholds]
                infer_state.ffn_rms_norm_table = wandb.Table(columns=cols_name)
            ffn_rms_weight_norm = (
                torch.norm(layer_weight.ffn_norm_weight_.weight, dim=-1).float().detach().cpu().numpy().item()
            )
            ffn_input_rms_mean = layer_weight.ffn_norm_weight_.weight.abs().mean().item()
            ffn_input_rms_max = layer_weight.ffn_norm_weight_.weight.abs().max().item()
            ffn_input_rms_min = layer_weight.ffn_norm_weight_.weight.abs().min().item()
            ffn_input_rms_abs_sum = layer_weight.ffn_norm_weight_.weight.abs().sum().item()
            ffn_norm_weight = layer_weight.ffn_norm_weight_.weight.abs().flatten().detach().cpu()
            ffn_norm_weight = ffn_norm_weight.sort(dim=-1, descending=True)[0].cumsum(dim=0)
            ffn_norm_weight = ffn_norm_weight / ffn_norm_weight[-1]
            ffn_rms_norm_energy_indices = [
                int(torch.searchsorted(ffn_norm_weight, torch.tensor(t, device=ffn_norm_weight.device)).item())
                / ffn_norm_weight.shape[0]
                for t in energy_thresholds
            ]

            infer_state.ffn_rms_norm_table.add_data(
                int(self.layer_num_),
                float(ffn_input_rms_abs_sum),
                float(ffn_rms_weight_norm),
                float(ffn_input_rms_mean),
                float(ffn_input_rms_max),
                float(ffn_input_rms_min),
                *ffn_rms_norm_energy_indices,
            )
            # 找到 a 中第一个大于0.3， 0.5, 0.7, 0.8, 0.9 的位置对应索引

            if not hasattr(infer_state, "ffn_up_gate_energy_table"):
                infer_state.ffn_up_gate_energy_table = wandb.Table(
                    columns=[
                        "layer_index",
                        "ffn_up_gate_energy_30",
                        "ffn_up_gate_energy_50",
                        "ffn_up_gate_energy_70",
                        "ffn_up_gate_energy_80",
                        "ffn_up_gate_energy_90",
                    ]
                )
            a = a.sort(dim=-1, descending=True)[0].cumsum(dim=0)
            a = a / a[-1]
            energy_indices = [
                int(torch.searchsorted(a, torch.tensor(t, device=a.device)).item()) / a.shape[0]
                for t in energy_thresholds
            ]
            infer_state.ffn_up_gate_energy_table.add_data(int(self.layer_num_), *energy_indices)

        input = None
        up_gate_out = None
        ffn2_out = layer_weight.down_proj.mm(ffn1_out)
        if not torch.cuda.is_current_stream_capturing() and not infer_state.is_prefill:
            abs_ffn1_out = ffn1_out.abs().flatten()
            sorted_ffn1_out = abs_ffn1_out.sort(dim=-1, descending=True)[0]
            dims = sorted_ffn1_out.view(-1).shape[0]
            energy_thresholds = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]
            cols_names = ["layer_index"]
            cols_datas = []
            for t in energy_thresholds:
                cols_names += [f"ffn1_out_energy_{int(t*100)}_cos_sim", f"ffn1_out_energy_{int(t*100)}_l2_dist"]
                split_value = sorted_ffn1_out[int(dims * t) - 1].detach().item()
                mask = abs_ffn1_out < split_value
                new_ffn1_out = ffn1_out.masked_fill(mask.view(1, -1), 0)
                new_ffn2_out = layer_weight.down_proj.mm(new_ffn1_out)
                # 计算新旧ffn2_out的cosine相似度
                cos_sim = torch.nn.functional.cosine_similarity(ffn2_out.view(-1), new_ffn2_out.view(-1), dim=0).item()
                # 计算新旧ffn2_out的l2距离
                l2_dist = torch.norm(ffn2_out.view(-1) - new_ffn2_out.view(-1)).item()
                cols_datas.append(cos_sim)
                cols_datas.append(l2_dist)

            if not hasattr(infer_state, "ffn2_out_energy_table"):
                infer_state.ffn2_out_energy_table = wandb.Table(columns=cols_names)
            infer_state.ffn2_out_energy_table.add_data(int(self.layer_num_), *cols_datas)

        ffn1_out = None
        return ffn2_out

    def _tpsp_ffn(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight
    ) -> torch.Tensor:
        input = input.view(-1, self.embed_dim_)
        if self.tp_world_size_ > 1:
            sp_token_num, hidden_dim = input.shape
            gather_input = self.alloc_tensor(
                (sp_token_num * self.tp_world_size_, hidden_dim), dtype=input.dtype, device=input.device
            )
            all_gather_into_tensor(gather_input, input, group=infer_state.dist_group, async_op=False)
            input = gather_input

        up_gate_out = layer_weight.gate_up_proj.mm(input)
        ffn1_out = self.alloc_tensor((input.size(0), up_gate_out.size(1) // 2), input.dtype)
        silu_and_mul_fwd(up_gate_out, ffn1_out)
        input = None
        up_gate_out = None
        ffn2_out = layer_weight.down_proj.mm(ffn1_out)
        ffn1_out = None
        if self.tp_world_size_ > 1:
            sp_token_num = ffn2_out.shape[0] // self.tp_world_size_
            reduce_o_tensor = self.alloc_tensor(
                (sp_token_num, self.embed_dim_), dtype=ffn2_out.dtype, device=ffn2_out.device
            )
            reduce_scatter_tensor(
                reduce_o_tensor, ffn2_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False
            )
            ffn2_out = reduce_o_tensor
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
        input_embdings = self.tpsp_token_forward(input_embdings, infer_state, layer_weight=layer_weight)
        input_embdings1 = self.tpsp_token_forward(input_embdings1, infer_state1, layer_weight=layer_weight)
        return input_embdings, input_embdings1

    def overlap_tpsp_context_forward(
        self,
        input_embdings: torch.Tensor,
        input_embdings1: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        infer_state1: LlamaInferStateInfo,
        layer_weight: LlamaTransformerLayerWeight,
    ):
        input_embdings = self.tpsp_context_forward(input_embdings, infer_state, layer_weight=layer_weight)
        input_embdings1 = self.tpsp_context_forward(input_embdings1, infer_state1, layer_weight=layer_weight)
        return input_embdings, input_embdings1
