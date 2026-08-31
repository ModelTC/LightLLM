# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/nsa_backend.py
# Uses sgl_kernel.flash_mla and sgl_kernel.flash_attn from the sglang kernel library.

import dataclasses
import torch
import torch.distributed as dist
from typing import Tuple, TYPE_CHECKING

from ..base_att import BaseAttBackend, BasePrefillAttState, BaseDecodeAttState, AttControl
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.dist_utils import get_current_rank_in_dp
from lightllm.utils.envs_utils import get_env_start_args

if TYPE_CHECKING:
    from lightllm.common.basemodel.infer_struct import InferStateInfo


_TP_HEAD_TOKEN_TRANSPOSE_MIN_TOKENS = 4096


def _copy_received_head_shards(received: torch.Tensor, output: torch.Tensor, world_size: int) -> None:
    """Transpose all-to-all receive order from rank-major to token-major heads."""

    tokens, local_heads, head_dim = received.shape
    tokens_per_rank = tokens // world_size
    output.view(tokens_per_rank, world_size, local_heads, head_dim).copy_(
        received.view(world_size, tokens_per_rank, local_heads, head_dim).permute(1, 0, 2, 3)
    )


def _copy_token_shard_for_head_scatter(output: torch.Tensor, send: torch.Tensor, world_size: int) -> None:
    """Transpose token-major global heads into all-to-all destination order."""

    tokens_per_rank, global_heads, head_dim = output.shape
    local_heads = global_heads // world_size
    send.view(world_size, tokens_per_rank, local_heads, head_dim).copy_(
        output.view(tokens_per_rank, world_size, local_heads, head_dim).permute(1, 0, 2, 3)
    )


def _should_use_tp_head_token_transpose(
    q: torch.Tensor,
    infer_state: "InferStateInfo",
    required_heads: int,
) -> bool:
    """Select the exact TP transpose only for its validated serving layout."""

    args = get_env_start_args()
    world_size = infer_state.dist_group.dp_world_size
    return (
        world_size > 1
        and q.is_contiguous()
        and q.shape[0] >= _TP_HEAD_TOKEN_TRANSPOSE_MIN_TOKENS
        and q.shape[0] % world_size == 0
        and q.shape[1] * world_size == required_heads
        and infer_state.max_cache_len == 0
        and not infer_state.need_dp_prefill_balance
        and not infer_state.use_replicated_attention_ep
        and not args.enable_tpsp_mix_mode
        and not args.enable_prefill_cudagraph
        and not args.enable_prefill_microbatch_overlap
        and not args.enable_prefill_decode_mixed
    )


def _alloc_like(input_: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    return torch.empty(shape, dtype=input_.dtype, device=input_.device)


class NsaFlashMlaSparseAttBackend(BaseAttBackend):
    def __init__(self, model):
        super().__init__(model=model)
        device = get_current_device_id()
        self.ragged_mem_buffers = [
            torch.empty(model.graph_max_batch_size * model.max_seq_length, dtype=torch.int32, device=device)
            for _ in range(2)
        ]

    def create_att_prefill_state(self, infer_state: "InferStateInfo") -> "NsaFlashMlaSparsePrefillAttState":
        return NsaFlashMlaSparsePrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state: "InferStateInfo") -> "NsaFlashMlaSparseDecodeAttState":
        return NsaFlashMlaSparseDecodeAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class NsaFlashMlaSparsePrefillAttState(BasePrefillAttState):
    """Prefill attention state for NSA using flash_mla_sparse_fwd."""

    ks: torch.Tensor = None
    ke: torch.Tensor = None
    lengths: torch.Tensor = None
    ragged_mem_index: torch.Tensor = None

    def init_state(self):
        self.backend: NsaFlashMlaSparseAttBackend = self.backend
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

        return self._nsa_prefill_att(q=q, kv=k, att_control=att_control)

    def _nsa_prefill_att(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        att_control: AttControl,
    ) -> torch.Tensor:
        from sgl_kernel.flash_mla import flash_mla_sparse_fwd

        nsa_dict = att_control.nsa_prefill_dict
        topk_mem_indices = nsa_dict["topk_mem_indices"]
        softmax_scale = nsa_dict["softmax_scale"]
        kv_lora_rank = nsa_dict["kv_lora_rank"]

        if topk_mem_indices.ndim == 2:
            topk_mem_indices = topk_mem_indices.unsqueeze(1)

        # The FlashMLA sparse kernels require 64 query heads on Hopper and
        # 128 on Blackwell. Tensor parallelism can leave fewer local heads
        # (GLM-5.3 has 64 / TP8 = 8), so pad the inactive heads and trim the
        # result just as SGLang's DSA backend does.
        num_tokens, num_heads, head_dim = q.shape
        device_sm_major = torch.cuda.get_device_capability(q.device)[0]
        required_heads = 128 if device_sm_major >= 10 else 64
        need_padding = num_heads % required_heads != 0
        if need_padding and _should_use_tp_head_token_transpose(q, self.infer_state, required_heads):
            world_size = self.infer_state.dist_group.dp_world_size
            rank = get_current_rank_in_dp()
            tokens_per_rank = num_tokens // world_size

            received_q = _alloc_like(q, q.shape)
            dist.all_to_all_single(
                received_q,
                q,
                group=self.infer_state.dist_group.device_group,
            )
            transposed_q = _alloc_like(q, (tokens_per_rank, world_size * num_heads, head_dim))
            _copy_received_head_shards(received_q, transposed_q, world_size)
            del received_q

            token_start = rank * tokens_per_rank
            local_indices = topk_mem_indices[token_start : token_start + tokens_per_rank]
            transposed_out, _, _ = flash_mla_sparse_fwd(
                q=transposed_q,
                kv=kv,
                indices=local_indices,
                sm_scale=softmax_scale,
                d_v=kv_lora_rank,
            )

            send_out = _alloc_like(q, q.shape)
            _copy_token_shard_for_head_scatter(transposed_out, send_out, world_size)
            del transposed_q, transposed_out
            output = _alloc_like(q, q.shape)
            dist.all_to_all_single(
                output,
                send_out,
                group=self.infer_state.dist_group.device_group,
            )
            del send_out
            return output

        if need_padding:
            assert required_heads % num_heads == 0, (
                f"num_heads {num_heads} cannot be padded to {required_heads}; "
                "the tensor-parallel size is unsupported"
            )
            q_input = q.new_zeros((num_tokens, required_heads, head_dim))
            q_input[:, :num_heads, :] = q
        else:
            q_input = q

        mla_out, _, _ = flash_mla_sparse_fwd(
            q=q_input,
            kv=kv,
            indices=topk_mem_indices,
            sm_scale=softmax_scale,
            d_v=kv_lora_rank,
        )
        if need_padding:
            mla_out = mla_out[:, :num_heads, :]
        return mla_out


@dataclasses.dataclass
class NsaFlashMlaSparseDecodeAttState(BaseDecodeAttState):

    ks: torch.Tensor = None
    ke: torch.Tensor = None
    lengths: torch.Tensor = None
    ragged_mem_index: torch.Tensor = None
    nsa_cache_seqlens: torch.Tensor = None
    nsa_cu_seqlens_k_new: torch.Tensor = None

    def init_state(self):
        self.backend: NsaFlashMlaSparseAttBackend = self.backend
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
        self.nsa_cache_seqlens = torch.minimum(
            torch.full(size=(self.infer_state.batch_size,), fill_value=2048, dtype=torch.int32, device="cuda"),
            self.infer_state.b_seq_len,
        )
        padded_seq_lens = torch.zeros(size=(self.nsa_cache_seqlens.shape[0] + 1,), dtype=torch.int32, device="cuda")
        # 进行 cumsum 操作
        padded_seq_lens[1:].copy_(self.nsa_cache_seqlens, non_blocking=True)
        self.nsa_cu_seqlens_k_new = padded_seq_lens.cumsum(dim=0, dtype=torch.int32)

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
        from sgl_kernel.flash_attn import flash_attn_with_kvcache

        nsa_dict = att_control.nsa_decode_dict
        topk_mem_indices = nsa_dict["topk_mem_indices"]
        softmax_scale = nsa_dict["softmax_scale"]
        kv_lora_rank = nsa_dict["kv_lora_rank"]
        qk_rope_head_dim = nsa_dict["qk_rope_head_dim"]

        q_nope, q_rope = q

        # Extract k_rope and kv_nope from the KV buffer
        only_qv = qk_rope_head_dim == 0
        if only_qv:
            k_rope = None
            kv_nope = kv[:, :, :kv_lora_rank].view(-1, 1, 1, kv_lora_rank)
        else:
            k_rope = kv[:, :, -qk_rope_head_dim:].view(-1, 1, 1, qk_rope_head_dim)
            kv_nope = kv[:, :, :-qk_rope_head_dim].view(-1, 1, 1, kv_lora_rank)

        o_tensor = flash_attn_with_kvcache(
            q=None if only_qv else q_rope,
            k_cache=k_rope,
            v_cache=kv_nope,
            qv=q_nope,
            only_qv=only_qv,
            page_table=topk_mem_indices,
            cache_seqlens=self.nsa_cache_seqlens,
            cu_seqlens_q=self.infer_state.b1_cu_q_seq_len,
            cu_seqlens_k_new=self.nsa_cu_seqlens_k_new,
            max_seqlen_q=self.infer_state.max_q_seq_len,
            softmax_scale=softmax_scale,
            causal=True,
        )
        return o_tensor
