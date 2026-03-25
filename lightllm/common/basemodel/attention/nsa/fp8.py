import dataclasses
from typing import TYPE_CHECKING, Tuple

import torch

from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.flashmla_utils import import_flash_mla

from ..base_att import AttControl, BaseAttBackend, BaseDecodeAttState, BasePrefillAttState

if TYPE_CHECKING:
    from lightllm.common.basemodel.infer_struct import InferStateInfo


class NsaFlashMlaFp8AttBackend(BaseAttBackend):
    def __init__(self, model):
        super().__init__(model=model)
        device = get_current_device_id()
        self.ragged_mem_buffers = [
            torch.empty(model.graph_max_batch_size * model.max_seq_length, dtype=torch.int32, device=device)
            for _ in range(2)
        ]

    def create_att_prefill_state(self, infer_state: "InferStateInfo") -> "NsaFlashMlaFp8PrefillAttState":
        return NsaFlashMlaFp8PrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state: "InferStateInfo") -> "NsaFlashMlaFp8DecodeAttState":
        return NsaFlashMlaFp8DecodeAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class NsaFlashMlaFp8PrefillAttState(BasePrefillAttState):
    ks: torch.Tensor = None
    ke: torch.Tensor = None
    lengths: torch.Tensor = None
    ragged_mem_index: torch.Tensor = None

    def init_state(self):
        self.backend: NsaFlashMlaFp8AttBackend = self.backend
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
            q_token_num=self.infer_state.total_token_num - self.infer_state.prefix_total_token_num,
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
        return self._nsa_prefill_att(q=q, att_control=att_control)

    def _nsa_prefill_att(
        self,
        q: torch.Tensor,
        att_control: AttControl,
    ) -> torch.Tensor:
        flash_mla = import_flash_mla()

        nsa_dict = att_control.nsa_prefill_dict
        layer_index = nsa_dict["layer_index"]
        topk_indices = nsa_dict["topk_indices"]
        softmax_scale = nsa_dict["softmax_scale"]
        kv_lora_rank = nsa_dict["kv_lora_rank"]

        kv = self.infer_state.mem_manager.get_prefill_kv_cache(layer_index)
        if topk_indices.ndim == 2:
            topk_indices = topk_indices.unsqueeze(1)

        topk_length = torch.sum(topk_indices != -1, dim=-1, dtype=torch.int32)
        if topk_length.ndim == 2 and topk_length.shape[1] == 1:
            topk_length = topk_length[:, 0].contiguous()

        mla_out, _, _ = flash_mla.flash_mla_sparse_fwd(
            q=q.contiguous(),
            kv=kv.contiguous(),
            indices=topk_indices.contiguous(),
            sm_scale=softmax_scale,
            d_v=kv_lora_rank,
            topk_length=topk_length,
        )
        return mla_out


@dataclasses.dataclass
class NsaFlashMlaFp8DecodeAttState(BaseDecodeAttState):
    ks: torch.Tensor = None
    ke: torch.Tensor = None
    lengths: torch.Tensor = None
    ragged_mem_index: torch.Tensor = None
    flashmla_sched_meta: object = None

    def init_state(self):
        self.backend: NsaFlashMlaFp8AttBackend = self.backend
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
        flash_mla = import_flash_mla()
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
        return self._nsa_decode_att(q=q, kv=k, att_control=att_control)

    def _nsa_decode_att(
        self,
        q: Tuple[torch.Tensor, torch.Tensor],
        kv: torch.Tensor,
        att_control: AttControl,
    ) -> torch.Tensor:
        flash_mla = import_flash_mla()

        nsa_dict = att_control.nsa_decode_dict
        topk_indices = nsa_dict["topk_indices"]
        softmax_scale = nsa_dict["softmax_scale"]
        kv_lora_rank = nsa_dict["kv_lora_rank"]

        if topk_indices.ndim == 2:
            topk_indices = topk_indices.unsqueeze(1)
        assert topk_indices.shape[1] == 1, "FlashMLA sparse decode path currently expects seq_len_q == 1"

        q_nope, q_rope = q
        q_all = torch.cat([q_nope, q_rope], dim=-1).unsqueeze(1).contiguous()

        o_tensor, _ = flash_mla.flash_mla_with_kvcache(
            q=q_all,
            k_cache=kv.contiguous(),
            block_table=None,
            cache_seqlens=None,
            head_dim_v=kv_lora_rank,
            tile_scheduler_metadata=self.flashmla_sched_meta,
            num_splits=None,
            softmax_scale=softmax_scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=topk_indices.contiguous(),
        )
        return o_tensor[:, 0, :, :]
