import os
import torch
import torch.distributed as dist
from ..transformer_layer_infer import TransformerLayerInfer
from ...infer_struct import InferStateInfo
from lightllm.distributed import all_reduce
from typing import Tuple
from lightllm.utils.tensor_utils import tensor_to_no_ref_tensor


class TransformerLayerInferTpl(TransformerLayerInfer):
    """ """

    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        # need to set by subclass
        self.eps_ = 1e-5
        self.tp_q_head_num_ = -1
        self.tp_k_head_num_ = -1
        self.tp_v_head_num_ = -1
        self.tp_o_head_num_ = -1
        self.head_dim_ = -1
        self.embed_dim_ = -1
        return

    def _att_norm(self, input, infer_state: InferStateInfo, layer_weight) -> torch.Tensor:
        raise Exception("need to impl")

    def _ffn_norm(self, input, infer_state: InferStateInfo, layer_weight) -> torch.Tensor:
        raise Exception("need to impl")

    def _get_qkv(self, input, infer_state: InferStateInfo, layer_weight) -> Tuple[torch.Tensor, torch.Tensor]:
        raise Exception("need to impl")

    def _tpsp_get_qkv(self, input, infer_state: InferStateInfo, layer_weight) -> Tuple[torch.Tensor, torch.Tensor]:
        raise Exception("need to impl")

    def _post_cache_kv(self, cache_kv, infer_state: InferStateInfo, layer_weight):
        mem_manager = infer_state.mem_manager
        mem_manager.copy_kv_to_mem_manager(
            layer_index=self.layer_num_,
            mem_index=infer_state.mem_index,
            kv=cache_kv,
        )
        return

    def _context_attention_kernel(self, q, kv, infer_state: InferStateInfo, layer_weight, out=None) -> torch.Tensor:
        raise Exception("need to impl")

    def _token_attention_kernel(self, q, infer_state: InferStateInfo, layer_weight, out=None) -> torch.Tensor:
        raise Exception("need to impl")

    def _get_o(self, input, infer_state: InferStateInfo, layer_weight) -> torch.Tensor:
        raise Exception("need to impl")

    def _tpsp_get_o(self, input, infer_state: InferStateInfo, layer_weight) -> torch.Tensor:
        raise Exception("need to impl")

    def _ffn(self, input, infer_state: InferStateInfo, layer_weight) -> torch.Tensor:
        raise Exception("need to impl")

    def _tpsp_ffn(self, input, infer_state: InferStateInfo, layer_weight) -> torch.Tensor:
        raise Exception("need to impl")

    def context_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        q, cache_kv = self._get_qkv(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)

        o = self._context_attention_wrapper_run(
            q=q, cache_kv=cache_kv, infer_state=infer_state, layer_weight=layer_weight
        )

        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.tp_world_size_ > 1:
            all_reduce(o, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(o.view(-1, self.embed_dim_))
        o = None

        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        # if self.layer_num_ == 0:
        #     input_embdings.fill_(0)
        input1 = None
        if self.tp_world_size_ > 1:
            all_reduce(ffn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings

    def energy_10_percentile(self, input: torch.Tensor):
        input = input.view(-1)
        value = (-input.view(-1).float().abs()).kthvalue(k=int(input.numel() * 0.1), dim=-1)[0].item()
        tmp_input = input.masked_fill(input.abs() < (-value), 0)
        return torch.norm(tmp_input, dim=-1).item()

    def token_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        need_wandb = not torch.cuda.is_current_stream_capturing() and not infer_state.is_prefill
        if need_wandb:
            att_input_norm = torch.norm(input_embdings.view(-1), dim=-1).float().detach().cpu().numpy().item()
            att_input_norm_10_percentile_energy = self.energy_10_percentile(input_embdings)

        input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        if need_wandb:
            att_rmsnorm_out_norm = torch.norm(input1.view(-1), dim=-1).float().detach().cpu().numpy().item()
        q, cache_kv = self._get_qkv(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.tp_world_size_ > 1:
            all_reduce(o, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)

        if need_wandb:
            att_delta_norm = torch.norm(o.view(-1), dim=-1).float().detach().cpu().numpy().item()

        input_embdings.add_(o.view(-1, self.embed_dim_))

        if need_wandb:
            att_out_norm = torch.norm(input_embdings.view(-1), dim=-1).float().detach().cpu().numpy().item()

        o = None

        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        if need_wandb:
            ffn_rmsnorm_out_norm = torch.norm(input1.view(-1), dim=-1).float().detach().cpu().numpy().item()
            ffn_rmsnorm_out_norm_10_percentile_energy = self.energy_10_percentile(input1)

        ffn_out = self._ffn(input1, infer_state, layer_weight)

        if need_wandb:
            ffn_delta_norm = torch.norm(ffn_out.view(-1), dim=-1).float().detach().cpu().numpy().item()

        input1 = None
        if self.tp_world_size_ > 1:
            all_reduce(ffn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))

        if need_wandb:
            ffn_out_norm = torch.norm(input_embdings.view(-1), dim=-1).float().detach().cpu().numpy().item()

            import wandb

            if not hasattr(infer_state, "resnet_table"):
                infer_state.resnet_table = wandb.Table(
                    columns=[
                        "layer_index",
                        "att_input_norm",
                        "att_input_norm_10_percentile_energy",
                        "att_rmsnorm_out_norm",
                        "att_delta_norm",
                        "att_out_norm",
                        "ffn_rmsnorm_out_norm",
                        "ffn_rmsnorm_out_norm_10_percentile_energy",
                        "ffn_delta_norm",
                        "ffn_out_norm",
                    ]
                )

            infer_state.resnet_table.add_data(
                int(self.layer_num_),
                float(att_input_norm),
                float(att_input_norm_10_percentile_energy),
                float(att_rmsnorm_out_norm),
                float(att_delta_norm),
                float(att_out_norm),
                float(ffn_rmsnorm_out_norm),
                float(ffn_rmsnorm_out_norm_10_percentile_energy),
                float(ffn_delta_norm),
                float(ffn_out_norm),
            )
        return input_embdings

    def tpsp_context_forward(self, input_embdings: torch.Tensor, infer_state: InferStateInfo, layer_weight):
        input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        q, cache_kv = self._tpsp_get_qkv(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)

        o = self._context_attention_wrapper_run(
            q=q, cache_kv=cache_kv, infer_state=infer_state, layer_weight=layer_weight
        )

        q = None
        o = self._tpsp_get_o(o, infer_state, layer_weight)
        input_embdings.add_(o.view(-1, self.embed_dim_))
        o = None

        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._tpsp_ffn(input1, infer_state, layer_weight)
        input1 = None
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings

    def tpsp_token_forward(self, input_embdings: torch.Tensor, infer_state: InferStateInfo, layer_weight):
        input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        q, cache_kv = self._tpsp_get_qkv(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._tpsp_get_o(o, infer_state, layer_weight)
        input_embdings.add_(o.view(-1, self.embed_dim_))
        o = None

        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._tpsp_ffn(input1, infer_state, layer_weight)
        input1 = None
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings

    def _context_attention_wrapper_run(
        self, q: torch.Tensor, cache_kv: torch.Tensor, infer_state: InferStateInfo, layer_weight
    ) -> torch.Tensor:
        if torch.cuda.is_current_stream_capturing():
            q = q.contiguous()
            cache_kv = cache_kv.contiguous()
            _q, _cache_kv = (
                tensor_to_no_ref_tensor(q),
                tensor_to_no_ref_tensor(cache_kv),
            )
            pre_capture_graph = infer_state.prefill_cuda_graph_get_current_capture_graph()
            pre_capture_graph.__exit__(None, None, None)

            def get_o_shape_dtype_device():
                # 在一个新的 graph 中尝试运行，并不是为了捕获图，是为了尝试得到 o 的形状等信息
                with torch.cuda.graph(cuda_graph=torch.cuda.CUDAGraph()):
                    __o = self._context_attention_kernel(_q, _cache_kv, infer_state, layer_weight)
                    o_shape = __o.shape
                    o_dtype = __o.dtype
                    o_device = __o.device
                    del __o

                    import gc

                    gc.collect()
                    torch.cuda.empty_cache()
                return o_shape, o_dtype, o_device

            o_shape, o_dtype, o_device = get_o_shape_dtype_device()
            infer_state.prefill_cuda_graph_create_graph_obj()
            infer_state.prefill_cuda_graph_get_current_capture_graph().__enter__()
            o = torch.empty(o_shape, dtype=o_dtype, device=o_device)
            _o = tensor_to_no_ref_tensor(o)

            def att_func(new_infer_state: InferStateInfo):
                tmp_o = self._context_attention_kernel(_q, _cache_kv, new_infer_state, layer_weight)
                assert tmp_o.shape == _o.shape
                _o.copy_(tmp_o)
                return

            infer_state.prefill_cuda_graph_add_cpu_runnning_func(func=att_func, after_graph=pre_capture_graph)
        else:
            o = self._context_attention_kernel(q, cache_kv, infer_state, layer_weight)

        return o
