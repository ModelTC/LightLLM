import dataclasses
import weakref
import torch
from typing import TYPE_CHECKING, Tuple

from ..base_att import AttControl, BaseAttBackend, BaseDecodeAttState, BasePrefillAttState
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.common.attention.backends.sparse_mla.flashmla import (
    FlashMLASparsePrefillBatch,
    FlashMLASparsePrefillBackend,
    SparseMLAPrefillInputs,
    FlashMLASparseDecodeBatch,
    FlashMLASparseDecodeBackend,
    SparseMLADecodeInputs,
)

if TYPE_CHECKING:
    from lightllm.common.basemodel.infer_struct import InferStateInfo


class NsaFlashMlaFp8SparseAttBackend(BaseAttBackend):
    def __init__(self, model):
        super().__init__(model=model)
        self.prefill_backend = FlashMLASparsePrefillBackend()
        self.decode_backend = FlashMLASparseDecodeBackend()
        device = get_current_device_id()
        self.ragged_mem_buffers = [
            torch.empty(model.graph_max_batch_size * model.max_seq_length, dtype=torch.int32, device=device)
            for _ in range(2)
        ]

    def create_att_prefill_state(self, infer_state: "InferStateInfo") -> "NsaFlashMlaFp8SparsePrefillAttState":
        return NsaFlashMlaFp8SparsePrefillAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))

    def create_att_decode_state(self, infer_state: "InferStateInfo") -> "NsaFlashMlaFp8SparseDecodeAttState":
        return NsaFlashMlaFp8SparseDecodeAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))


@dataclasses.dataclass
class NsaFlashMlaFp8SparsePrefillAttState(BasePrefillAttState):
    ks: torch.Tensor = None
    ke: torch.Tensor = None
    lengths: torch.Tensor = None
    ragged_mem_index: torch.Tensor = None

    def init_state(self):
        self.backend: NsaFlashMlaFp8SparseAttBackend = self.backend
        self.ragged_mem_index = torch.empty(
            self.infer_state.total_token_num,
            dtype=torch.int32,
            device=get_current_device_id(),
        )
        from lightllm.common.basemodel.triton_kernel.gen_nsa_ks_ke import gen_nsa_ks_ke

        self.ks, self.ke, self.lengths = gen_nsa_ks_ke(
            b_seq_len=self.infer_state.b_seq_len,
            b_q_seq_len=self.infer_state.b_q_seq_len,
            b_req_idx=self.infer_state.b_req_idx,
            req_to_token_index=self.infer_state.req_manager.req_to_token_indexs,
            q_token_num=self.infer_state.input_ids.shape[0],
            ragged_mem_index=self.ragged_mem_index,
            hold_req_idx=self.infer_state.req_manager.HOLD_REQUEST_ID,
        )
        return

    def prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        assert att_control.nsa_prefill, "nsa_prefill must be True for NSA prefill attention"
        assert att_control.nsa_prefill_dict is not None, "nsa_prefill_dict is required"
        nsa_dict = att_control.nsa_prefill_dict
        topk_indices = nsa_dict["topk_indices"]
        softmax_scale = nsa_dict["softmax_scale"]
        kv_lora_rank = nsa_dict["kv_lora_rank"]
        topk_mem_indices = nsa_dict["topk_mem_indices"]
        prefill_cache_kv = nsa_dict["prefill_cache_kv"]

        if self.infer_state.max_cache_len > 0:
            # 当前推理生成的token kv部分从 prefill_cache_kv 中获取，历史
            # 部分kv 从 packed_kv 中获取, 并进行反量化，这样可以避免 prefill_cache_kv
            # 部分的数据进行重复的反量化操作，提升整体的性能。
            from lightllm.common.basemodel.triton_kernel.kv_copy.prefill_compact_kv_flashmla_fp8 import (
                get_prefill_kv_cache_and_remap_indices_triton,
            )

            kv, topk_indices = get_prefill_kv_cache_and_remap_indices_triton(
                packed_kv=k,
                topk_mem_indices=topk_mem_indices,
                prefill_mem_index=self.infer_state.mem_index,
                prefill_cache_kv=prefill_cache_kv,
                prefill_dtype=prefill_cache_kv.dtype,
            )
        else:
            kv = prefill_cache_kv

        out = alloc_func((q.shape[0], q.shape[1], kv_lora_rank), dtype=q.dtype, device=q.device)
        return self.backend.prefill_backend.forward(
            batch=FlashMLASparsePrefillBatch(indices=topk_indices),
            inputs=SparseMLAPrefillInputs(q=q, kv=kv, softmax_scale=softmax_scale),
            out=out,
            alloc_func=alloc_func,
        )


@dataclasses.dataclass
class NsaFlashMlaFp8SparseDecodeAttState(BaseDecodeAttState):
    ks: torch.Tensor = None
    ke: torch.Tensor = None
    lengths: torch.Tensor = None
    ragged_mem_index: torch.Tensor = None
    flashmla_sched_meta: object = None

    def init_state(self):
        self.backend: NsaFlashMlaFp8SparseAttBackend = self.backend
        model = self.backend.model
        use_cuda_graph = (
            self.infer_state.batch_size <= model.graph_max_batch_size
            and self.infer_state.max_kv_seq_len <= model.graph_max_len_in_batch
        )

        if use_cuda_graph:
            self.ragged_mem_index = self.backend.ragged_mem_buffers[self.infer_state.microbatch_index]
        else:
            self.ragged_mem_index = torch.empty(
                self.infer_state.total_token_num,
                dtype=torch.int32,
                device=get_current_device_id(),
            )

        from lightllm.common.basemodel.triton_kernel.gen_nsa_ks_ke import gen_nsa_ks_ke

        self.ks, self.ke, self.lengths = gen_nsa_ks_ke(
            b_seq_len=self.infer_state.b_seq_len,
            b_q_seq_len=self.infer_state.b_q_seq_len,
            b_req_idx=self.infer_state.b_req_idx,
            req_to_token_index=self.infer_state.req_manager.req_to_token_indexs,
            q_token_num=self.infer_state.b_seq_len.shape[0],
            ragged_mem_index=self.ragged_mem_index,
            hold_req_idx=self.infer_state.req_manager.HOLD_REQUEST_ID,
        )
        import flash_mla

        self.flashmla_sched_meta, _ = flash_mla.get_mla_metadata()
        return

    def decode_att(
        self,
        q: Tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        assert att_control.nsa_decode, "nsa_decode must be True for NSA decode attention"
        assert att_control.nsa_decode_dict is not None, "nsa_decode_dict is required"
        params = att_control.nsa_decode_dict
        q_nope, q_rope = q
        out = alloc_func(
            (q_nope.shape[0], q_nope.shape[1], params["kv_lora_rank"]), dtype=q_nope.dtype, device=q_nope.device
        )
        return self.backend.decode_backend.forward(
            batch=FlashMLASparseDecodeBatch(indices=params["topk_mem_indices"], sched_meta=self.flashmla_sched_meta),
            inputs=SparseMLADecodeInputs(
                q_nope=q_nope, q_rope=q_rope, packed_kv=k, softmax_scale=params["softmax_scale"]
            ),
            out=out,
            alloc_func=alloc_func,
        )
