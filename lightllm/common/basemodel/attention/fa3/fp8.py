import dataclasses
import weakref
import torch
from ..base_att import AttControl
from lightllm.utils.sgl_utils import flash_attn_with_kvcache
from lightllm.common.basemodel.triton_kernel.quantization.q_per_head_fp8_quant import q_per_head_static_fp8_quant
from lightllm.utils.log_utils import init_logger
from .fp import Fa3AttBackend, Fa3PrefillAttState, Fa3DecodeAttState


logger = init_logger(__name__)


class Fp8Fa3AttBackend(Fa3AttBackend):
    def _init_infer_page_size(self):
        # TODO: FP8 FA3 完成多 token 推理页适配后，改为继承模型 page_size。
        self.infer_page_size = 1
        assert self.model.args.page_size % self.infer_page_size == 0, (
            f"model page_size {self.model.args.page_size} "
            f"must be divisible by infer_page_size {self.infer_page_size}"
        )
        logger.warning(
            f"Fp8Fa3AttBackend temporarily uses infer_page_size=1 with model page_size={self.model.args.page_size}; "
            "multi-token FP8 inference pages are not implemented yet."
        )

    def create_att_prefill_state(self, infer_state) -> "Fp8Fa3PrefillAttState":
        return Fp8Fa3PrefillAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))

    def create_att_decode_state(self, infer_state) -> "Fp8Fa3DecodeAttState":
        return Fp8Fa3DecodeAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))


@dataclasses.dataclass
class Fp8Fa3PrefillAttState(Fa3PrefillAttState):
    k_descale: torch.Tensor = None
    v_descale: torch.Tensor = None

    def init_state(self):
        super().init_state()
        batch_size = self.infer_state.batch_size
        state = getattr(
            self.infer_state.att_resource_manager,
            f"{self.infer_state.att_resource_role}_gqa_fp8_static_per_head_state",
        )

        offline_scales: torch.Tensor = state.scales
        head_num = state.config.head_num
        # 为了减少推理计算量，在推理外部初始化k_descale和v_descale
        self.k_descale = (
            offline_scales[:, :head_num].view(-1, 1, head_num).expand(offline_scales.shape[0], batch_size, head_num)
        )
        self.v_descale = (
            offline_scales[:, head_num:].view(-1, 1, head_num).expand(offline_scales.shape[0], batch_size, head_num)
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
            logical_layer_id=att_control.logical_layer_id,
        )

    def _fp8_prefill_att(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alloc_func=torch.empty,
        *, logical_layer_id: int,
    ) -> torch.Tensor:
        self.backend: Fp8Fa3AttBackend = self.backend  # for typing

        q_head_num = q.shape[1]
        q_head_dim = q.shape[2]
        k_head_num = k.shape[1]
        k_head_dim = k.shape[2]
        cache_k = k.view(-1, 1, k_head_num, k_head_dim).view(torch.float8_e4m3fn)
        cache_v = v.view(-1, 1, k_head_num, k_head_dim).view(torch.float8_e4m3fn)
        state = getattr(
            self.infer_state.att_resource_manager,
            f"{self.infer_state.att_resource_role}_gqa_fp8_static_per_head_state",
        )
        storage_layer_index = state.get_storage_layer_index(logical_layer_id=logical_layer_id)
        static_q_scales = state.q_scales[storage_layer_index]
        q = q_per_head_static_fp8_quant(q.reshape(q.shape[0], k_head_num, -1), static_q_scales)
        q_scale = static_q_scales.view(1, k_head_num).expand(self.infer_state.b_seq_len.shape[0], k_head_num)
        o = flash_attn_with_kvcache(
            q=q.reshape(-1, q_head_num, q_head_dim),
            k_cache=cache_k,
            v_cache=cache_v,
            page_table=self.page_table,
            cache_seqlens=self.infer_state.b_seq_len,
            cu_seqlens_q=self.cu_seqlens_q,
            cu_seqlens_k_new=self.cu_seqlens_k,
            max_seqlen_q=self.infer_state.max_q_seq_len,
            causal=self.causal,
            window_size=(-1, -1),
            softcap=0.0,
            q_descale=q_scale,
            k_descale=self.k_descale[storage_layer_index],
            v_descale=self.v_descale[storage_layer_index],
            return_softmax_lse=False,
        )
        return o


@dataclasses.dataclass
class Fp8Fa3DecodeAttState(Fa3DecodeAttState):
    k_descale: torch.Tensor = None
    v_descale: torch.Tensor = None

    def init_state(self):
        self.backend: Fp8Fa3AttBackend = self.backend

        state = getattr(
            self.infer_state.att_resource_manager,
            f"{self.infer_state.att_resource_role}_gqa_fp8_static_per_head_state",
        )
        super().init_state()

        att_batch_size = self.b_att_seq_len.shape[0]
        offline_scales: torch.Tensor = state.scales
        head_num = state.config.head_num

        # 为了减少推理计算量，在推理外部初始化k_descale和v_descale
        self.k_descale = (
            offline_scales[:, :head_num].view(-1, 1, head_num).expand(offline_scales.shape[0], att_batch_size, head_num)
        )
        self.v_descale = (
            offline_scales[:, head_num:].view(-1, 1, head_num).expand(offline_scales.shape[0], att_batch_size, head_num)
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
            logical_layer_id=att_control.logical_layer_id,
        )

    def _fp8_decode_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alloc_func=torch.empty,
        *,
        logical_layer_id: int,
    ):
        k_head_num = k.shape[1]
        k_head_dim = k.shape[2]

        cache_k = k.view(-1, 1, k_head_num, k_head_dim).view(torch.float8_e4m3fn)
        cache_v = v.view(-1, 1, k_head_num, k_head_dim).view(torch.float8_e4m3fn)

        state = getattr(
            self.infer_state.att_resource_manager,
            f"{self.infer_state.att_resource_role}_gqa_fp8_static_per_head_state",
        )
        storage_layer_index = state.get_storage_layer_index(logical_layer_id=logical_layer_id)

        q_head_num = q.shape[1]
        att_batch_size = self.b_att_seq_len.shape[0]
        static_q_scales = state.q_scales
        q = q_per_head_static_fp8_quant(q.reshape(q.shape[0], k_head_num, -1), static_q_scales[storage_layer_index])
        q_scale = static_q_scales[storage_layer_index].view(1, k_head_num).expand(att_batch_size, k_head_num)
        o = flash_attn_with_kvcache(
            q=q.reshape(-1, q_head_num, k_head_dim),
            k_cache=cache_k,
            v_cache=cache_v,
            page_table=self.page_table,
            cache_seqlens=self.b_att_seq_len,
            cu_seqlens_q=self.cu_seqlens_q,
            cu_seqlens_k_new=self.cu_seqlens_k,
            max_seqlen_q=self.decode_max_q_seq_len,
            causal=self.causal,
            window_size=(-1, -1),
            softcap=0.0,
            q_descale=q_scale.view(att_batch_size, k_head_num),
            k_descale=self.k_descale[storage_layer_index],
            v_descale=self.v_descale[storage_layer_index],
            return_softmax_lse=False,
        )
        return o
