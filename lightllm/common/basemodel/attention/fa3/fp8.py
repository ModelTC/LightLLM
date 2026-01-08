import dataclasses
import torch
from ..base_att import AttControl
from typing import Optional, TYPE_CHECKING
from lightllm.utils.sgl_utils import flash_attn_with_kvcache
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.common.basemodel.triton_kernel.q_per_head_fp8_quant import q_per_head_fp8_quant
from lightllm.utils.vllm_utils import HAS_VLLM, vllm_ops
from typing import Union
from .fp import Fa3AttBackend, Fa3PrefillAttState, Fa3DecodeAttState

if HAS_VLLM:
    scaled_fp8_quant = vllm_ops.scaled_fp8_quant
else:
    scaled_fp8_quant = None


class Fp8Fa3AttBackend(Fa3AttBackend):
    def __init__(self, model):
        super().__init__(model=model)

    def create_att_prefill_state(self, infer_state) -> "Fp8Fa3PrefillAttState":
        return Fp8Fa3PrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state) -> "Fp8Fa3DecodeAttState":
        return Fp8Fa3DecodeAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class Fp8Fa3PrefillAttState(Fa3PrefillAttState):
    # 临时共享变量
    mid_token_batch_ids: torch.Tensor = None
    k_descale: torch.Tensor = None
    v_descale: torch.Tensor = None

    def init_state(self):
        super().init_state()
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

    def prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        assert (
            att_control.use_alibi is False
            and att_control.use_sliding_window is False
            and att_control.use_att_sink is False
        )
        return self._fp8_prefill_att(
            q=q,
            k=k,
            v=v,
            alloc_func=alloc_func,
        )

    def _fp8_prefill_att(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alloc_func=torch.empty
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
        layer_index = self.backend._find_layer_index(k=cache_k, v=cache_v, att_state=self)
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
            k_descale=self.k_descale[layer_index],
            v_descale=self.v_descale[layer_index],
            return_softmax_lse=False,
        )
        return o


@dataclasses.dataclass
class Fp8Fa3DecodeAttState(Fa3DecodeAttState):
    k_descale: torch.Tensor = None
    v_descale: torch.Tensor = None

    def init_state(self):
        super().init_state()
        self.backend: Fp8Fa3AttBackend = self.backend

        args_mtp_step = get_env_start_args().mtp_step
        att_batch_size = self.infer_state.batch_size // (args_mtp_step + 1)
        assert self.infer_state.batch_size % (args_mtp_step + 1) == 0

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
        super().copy_for_decode_cuda_graph(new_state)

    def decode_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ):
        assert (
            att_control.use_alibi is False
            and att_control.use_sliding_window is False
            and att_control.use_att_sink is False
        )
        return self._fp8_decode_att(
            q=q,
            k=k,
            v=v,
            alloc_func=alloc_func,
        )

    def _fp8_decode_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alloc_func=torch.empty,
    ):
        k_head_num = k.shape[1]
        k_head_dim = k.shape[2]

        cache_k = k.view(-1, 1, k_head_num, k_head_dim).view(torch.float8_e4m3fn)
        cache_v = v.view(-1, 1, k_head_num, k_head_dim).view(torch.float8_e4m3fn)

        layer_index = self.backend._find_layer_index(k=cache_k, v=cache_v, att_state=self)

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
            k_descale=self.k_descale[layer_index],
            v_descale=self.v_descale[layer_index],
            return_softmax_lse=False,
        )
        return o
