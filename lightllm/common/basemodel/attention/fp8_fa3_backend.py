import dataclasses
import torch
from .base_att import BaseAttBackend, BasePrefillAttState, BaseDecodeAttState, AttControl
from typing import Optional, TYPE_CHECKING
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.sgl_utils import flash_attn_with_kvcache
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.common.basemodel.triton_kernel.fa3_utils import page_table_copy
from lightllm.common.basemodel.triton_kernel.q_per_head_fp8_quant import q_per_head_fp8_quant
from lightllm.common.basemodel.triton_kernel.gen_prefill_params import gen_cumsum_pad0_tensor
from lightllm.utils.vllm_utils import HAS_VLLM, vllm_ops

if HAS_VLLM:
    scaled_fp8_quant = vllm_ops.scaled_fp8_quant
else:
    scaled_fp8_quant = None


class Fp8Fa3AttBackend(BaseAttBackend):
    def __init__(self, model):
        super().__init__(model=model)
        self.get_page_table_buffer()  # init

    def get_page_table_buffer(self):
        """
        用于减少 decode graph 捕获的时候, 造成显存二次方增长的情况.
        """
        model = self.model
        if self._shared_page_table_buffer is None:
            self._shared_page_table_buffer = [
                torch.empty(model.graph_max_batch_size * model.graph_max_len_in_batch, dtype=torch.int32).to(
                    get_current_device_id()
                ),
                torch.empty(model.graph_max_batch_size * model.graph_max_len_in_batch, dtype=torch.int32).to(
                    get_current_device_id()
                ),
            ]
        return self._shared_page_table_buffer

    def create_att_prefill_state(self, infer_state) -> "Fp8Fa3PrefillAttState":
        return Fp8Fa3PrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state) -> "Fp8Fa3DecodeAttState":
        return Fp8Fa3DecodeAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class Fp8Fa3PrefillAttState(BasePrefillAttState):
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    page_table: torch.Tensor = None
    # 临时共享变量
    mid_token_batch_ids: torch.Tensor = None
    k_descale: torch.Tensor = None
    v_descale: torch.Tensor = None

    def init_state(self):
        self.cu_seqlens_q = self.infer_state.b1_cu_q_seq_len.int()
        self.cu_seqlens_k = self.infer_state.b1_cu_kv_seq_len.int()
        self.page_table = torch.empty(
            (self.infer_state.batch_size, self.infer_state.max_kv_seq_len),
            dtype=torch.int32,
            device=self.infer_state.input_ids.device,
        )
        self.page_table.copy_(
            self.infer_state.req_manager.req_to_token_indexs[
                self.infer_state.b_req_idx, : self.infer_state.max_kv_seq_len
            ]
        )

        device = self.infer_state.input_ids.device
        batch_size = self.infer_state.batch_size
        mem_manager = self.backend.model.mem_manager

        offline_scales: torch.Tensor = mem_manager.scales
        head_num = mem_manager.head_num
        self.mid_token_batch_ids = torch.repeat_interleave(
            torch.arange(batch_size, device=device), self.infer_state.b_q_seq_len
        )
        # 为了减少推理计算量，在推理外部初始化k_descale和v_descale
        self.k_descale = (
            offline_scales[:, :head_num].view(-1, 1, head_num).expand(offline_scales.shape[0], batch_size, head_num)
            if offline_scales is not None
            else torch.ones(
                (mem_manager.layer_num, batch_size, head_num),
                dtype=torch.float32,
                device=device,
            )
        )
        self.v_descale = (
            offline_scales[:, head_num:].view(-1, 1, head_num).expand(offline_scales.shape[0], batch_size, head_num)
            if offline_scales is not None
            else torch.ones(
                (mem_manager.layer_num, batch_size, head_num),
                dtype=torch.float32,
                device=device,
            )
        )

    def copy_for_prefill_cuda_graph(self, new_state: "Fp8Fa3PrefillAttState"):
        pass

    def prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_weight,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        assert att_control.use_alibi is False
        return self._fp8_prefill_att(
            q=q,
            k=k,
            v=v,
            layer_weight=layer_weight,
            alloc_func=alloc_func,
        )

    def _fp8_prefill_att(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_weight, alloc_func=torch.empty
    ) -> torch.Tensor:
        self.backend: Fp8Fa3AttBackend = self.backend  # for typing

        q, q_scale = q_per_head_fp8_quant(
            q,
            self.infer_state.b_seq_len,
            self.cu_seqlens_q,
            self.mid_token_batch_ids,
        )
        k_head_num = k.shape[1]
        k_head_dim = k.shape[2]
        cache_k = k.view(-1, 1, k_head_num, k_head_dim).view(torch.float8_e4m3fn)
        cache_v = v.view(-1, 1, k_head_num, k_head_dim).view(torch.float8_e4m3fn)
        o = flash_attn_with_kvcache(
            q=q,
            k_cache=cache_k,
            v_cache=cache_v,
            page_table=self.page_table,
            cache_seqlens=self.infer_state.b_seq_len,
            cu_seqlens_q=self.cu_seqlens_q,
            cu_seqlens_k_new=self.cu_seqlens_k,
            max_seqlen_q=self.infer_state.max_q_seq_len,
            causal=True,
            window_size=(-1, -1),
            softcap=0.0,
            q_descale=q_scale,
            k_descale=self.k_descale[layer_weight.layer_num_],
            v_descale=self.v_descale[layer_weight.layer_num_],
            return_softmax_lse=False,
        )
        return o


@dataclasses.dataclass
class Fp8Fa3DecodeAttState(BaseDecodeAttState):
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    page_table: torch.Tensor = None
    b_att_seq_len: torch.Tensor = None
    # 在是否开启mtp 的不同模式下，其设置不同的值，可以加速算子的运行。
    decode_max_q_seq_len: int = None

    k_descale: torch.Tensor = None
    v_descale: torch.Tensor = None

    def init_state(self):
        self.backend: Fp8Fa3AttBackend = self.backend

        args_mtp_step = get_env_start_args().mtp_step
        if args_mtp_step > 0:
            # 修正 mtp 在 fa3 下的输入。
            mtp_size = args_mtp_step + 1
            b_q_seq_len = torch.full(
                (self.infer_state.b_seq_len.shape[0] // mtp_size,),
                fill_value=mtp_size,
                dtype=torch.int32,
                device=self.infer_state.b_seq_len.device,
            )
            b_kv_seq_len = self.infer_state.b_seq_len[mtp_size - 1 :: mtp_size]
            b1_cu_q_seq_len, b1_cu_kv_seq_len = gen_cumsum_pad0_tensor(
                b_q_seq_len, b_kv_seq_len[mtp_size - 1 :: mtp_size]
            )
            self.cu_seqlens_q = b1_cu_q_seq_len.int()
            self.cu_seqlens_k = b1_cu_kv_seq_len.int()
        else:
            self.cu_seqlens_q = self.infer_state.b1_cu_q_seq_len.int()
            self.cu_seqlens_k = self.infer_state.b1_cu_kv_seq_len.int()

        att_batch_size = self.infer_state.batch_size // (args_mtp_step + 1)
        assert self.infer_state.batch_size % (args_mtp_step + 1) == 0

        model = self.backend.model
        # 可以使用 cuda graph的时候从 buffer中申请
        if (
            self.infer_state.batch_size <= model.graph_max_batch_size
            and self.infer_state.max_kv_seq_len <= model.graph_max_len_in_batch
        ):
            page_buffer = self.backend.get_page_table_buffer(model.graph_max_batch_size, model.graph_max_len_in_batch)
            self.page_table = page_buffer[self.infer_state.microbatch_index][
                : att_batch_size * model.graph_max_len_in_batch
            ].reshape(att_batch_size, model.graph_max_len_in_batch)
        else:
            self.page_table = torch.empty(
                (att_batch_size, self.infer_state.max_kv_seq_len),
                dtype=torch.int32,
                device=self.infer_state.input_ids.device,
            )

        if args_mtp_step > 0:
            page_table_copy(
                page_table=self.page_table[:, : self.infer_state.max_kv_seq_len],
                req_to_token_indexs=model.req_manager.req_to_token_indexs,
                b_req_idx=self.infer_state.b_req_idx[args_mtp_step :: (args_mtp_step + 1)],
            )
            self.b_att_seq_len = self.infer_state.b_seq_len[args_mtp_step :: (args_mtp_step + 1)].contiguous()
            self.decode_max_q_seq_len = args_mtp_step + 1
        else:
            page_table_copy(
                page_table=self.page_table[:, : self.infer_state.max_kv_seq_len],
                req_to_token_indexs=model.req_manager.req_to_token_indexs,
                b_req_idx=self.infer_state.b_req_idx,
            )
            self.b_att_seq_len = self.infer_state.b_seq_len
            self.decode_max_q_seq_len = 1

        device = self.infer_state.input_ids.device
        batch_size = att_batch_size
        mem_manager = self.backend.model.mem_manager

        offline_scales: torch.Tensor = mem_manager.scales
        head_num = mem_manager.head_num

        # 为了减少推理计算量，在推理外部初始化k_descale和v_descale
        self.k_descale = (
            offline_scales[:, :head_num].view(-1, 1, head_num).expand(offline_scales.shape[0], batch_size, head_num)
            if offline_scales is not None
            else torch.ones(
                (mem_manager.layer_num, batch_size, head_num),
                dtype=torch.float32,
                device=device,
            )
        )
        self.v_descale = (
            offline_scales[:, head_num:].view(-1, 1, head_num).expand(offline_scales.shape[0], batch_size, head_num)
            if offline_scales is not None
            else torch.ones(
                (mem_manager.layer_num, batch_size, head_num),
                dtype=torch.float32,
                device=device,
            )
        )
        return

    def copy_for_decode_cuda_graph(self, new_state: "Fp8Fa3DecodeAttState"):
        pass

    def decode_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_weight,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ):
        assert att_control.use_alibi is False
        return self._fp8_decode_att(
            q=q,
            k=k,
            v=v,
            layer_weight=layer_weight,
            alloc_func=alloc_func,
        )

    def _fp8_decode_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_weight,
        alloc_func=torch.empty,
    ):
        k_head_num = k.shape[1]
        k_head_dim = k.shape[2]

        cache_k = k.view(-1, 1, k_head_num, k_head_dim).view(torch.float8_e4m3fn)
        cache_v = v.view(-1, 1, k_head_num, k_head_dim).view(torch.float8_e4m3fn)

        q_head_num = q.shape[1]
        q, q_scale = scaled_fp8_quant(q.view(q.shape[0] * k_head_num, -1), use_per_token_if_dynamic=True)
        o = flash_attn_with_kvcache(
            q=q.view(-1, q_head_num, k_head_dim),
            k_cache=cache_k,
            v_cache=cache_v,
            page_table=self.page_table,
            cache_seqlens=self.infer_state.b_seq_len,
            cu_seqlens_q=self.cu_seqlens_q,
            cu_seqlens_k_new=self.cu_seqlens_k,
            max_seqlen_q=self.decode_max_q_seq_len,
            causal=False,
            window_size=(-1, -1),
            softcap=0.0,
            q_descale=q_scale.view(self.infer_state.batch_size, k_head_num),
            k_descale=self.k_descale[layer_weight.layer_num_],
            v_descale=self.v_descale[layer_weight.layer_num_],
            return_softmax_lse=False,
        )
        return o
