import math
import os

import torch

from lightllm.common.kv_cache_mem_manager.deepseek4_mem_manager import DSV4_C4_PAGE_SIZE
from lightllm.utils.envs_utils import get_env_start_args


C4_PREFILL_LOGITS_BUDGET_BYTES = 512 * 1024 * 1024
C4_LOGITS_ALIGNMENT = 256


def _compress_cap(max_kv_seq_len: int, ratio: int) -> int:
    entries = max(1, int(max_kv_seq_len) // ratio)
    return ((entries + 63) // 64) * 64


class DeepseekV4C4PrefillWorkspace:
    """Persistent scratch used exclusively by the C4 prefill auxiliary stream."""

    def __init__(
        self,
        token_capacity: int,
        max_seq_length: int,
        max_request_num: int,
        index_n_heads: int,
        index_head_dim: int,
    ):
        self.token_capacity = int(token_capacity)
        self.max_request_num = int(max_request_num)
        self.index_n_heads = int(index_n_heads)
        self.index_head_dim = int(index_head_dim)
        self.max_page_cap = _compress_cap(max_seq_length, 4) // DSV4_C4_PAGE_SIZE

        # The same byte arena serves the sequential indexer-K, indexer-Q and logits
        # phases. Those phases run on one CUDA stream and therefore never overlap.
        self.scratch = torch.empty((C4_PREFILL_LOGITS_BUDGET_BYTES,), dtype=torch.uint8, device="cuda")
        self.idx_q_fp8 = torch.empty(
            (self.token_capacity, self.index_n_heads, self.index_head_dim),
            dtype=torch.float8_e4m3fn,
            device="cuda",
        )
        self.weights = torch.empty((self.token_capacity, self.index_n_heads, 1), dtype=torch.float32, device="cuda")
        self.page_table = torch.empty((self.max_request_num * self.max_page_cap,), dtype=torch.int32, device="cuda")
        self.row_page_table = torch.empty((self.token_capacity * self.max_page_cap,), dtype=torch.int32, device="cuda")
        self.c4_len = torch.empty((self.max_request_num,), dtype=torch.int32, device="cuda")
        self.valid_len = torch.empty((self.token_capacity,), dtype=torch.int32, device="cuda")
        self.ctx_lens = torch.empty((self.token_capacity, 1), dtype=torch.int32, device="cuda")
        self.topk_lengths = torch.empty((self.token_capacity,), dtype=torch.int32, device="cuda")

        max_indexer_k_phase = self.token_capacity * self.index_head_dim * (4 * 4 + 2 + 2)
        max_indexer_q_phase = self.token_capacity * self.index_n_heads * (self.index_head_dim * 2 + 2)
        assert max(max_indexer_k_phase, max_indexer_q_phase) <= C4_PREFILL_LOGITS_BUDGET_BYTES

    @staticmethod
    def _flat_view(buffer: torch.Tensor, shape) -> torch.Tensor:
        size = math.prod(shape)
        assert size <= buffer.numel()
        return buffer[:size].view(shape)

    def _scratch_view(self, shape, dtype: torch.dtype, byte_offset: int = 0) -> torch.Tensor:
        nbytes = math.prod(shape) * torch._utils._element_size(dtype)
        assert byte_offset % torch._utils._element_size(dtype) == 0
        assert byte_offset + nbytes <= self.scratch.numel()
        return self.scratch[byte_offset : byte_offset + nbytes].view(dtype).view(shape)

    def indexer_k_buffers(self, token_num: int):
        kv_score_shape = (token_num, 4 * self.index_head_dim)
        indexer_k_shape = (token_num, self.index_head_dim)
        kv_score = self._scratch_view(kv_score_shape, torch.float32)
        offset = kv_score.numel() * kv_score.element_size()
        indexer_k = self._scratch_view(indexer_k_shape, torch.bfloat16, offset)
        offset += indexer_k.numel() * indexer_k.element_size()
        hadamard = self._scratch_view(indexer_k_shape, torch.bfloat16, offset)
        return kv_score, indexer_k, hadamard

    def indexer_q_inputs(self, token_num: int):
        idx_q_shape = (token_num, self.index_n_heads * self.index_head_dim)
        raw_weights_shape = (token_num, self.index_n_heads)
        idx_q = self._scratch_view(idx_q_shape, torch.bfloat16)
        offset = idx_q.numel() * idx_q.element_size()
        raw_weights = self._scratch_view(raw_weights_shape, torch.bfloat16, offset)
        return idx_q, raw_weights

    def indexer_q_outputs(self, token_num: int):
        return self.idx_q_fp8[:token_num], self.weights[:token_num]

    def page_tables(self, batch_size: int, token_num: int, page_cap: int):
        assert page_cap <= self.max_page_cap
        page_table = self._flat_view(self.page_table, (batch_size, page_cap))
        row_page_table = self._flat_view(self.row_page_table, (token_num, page_cap))
        return page_table, row_page_table

    def metadata(self, batch_size: int, token_num: int):
        return (
            self.c4_len[:batch_size],
            self.valid_len[:token_num],
            self.ctx_lens[:token_num],
            self.topk_lengths[:token_num],
        )

    @staticmethod
    def aligned_c4_cap(c4_cap: int) -> int:
        return ((int(c4_cap) + C4_LOGITS_ALIGNMENT - 1) // C4_LOGITS_ALIGNMENT) * C4_LOGITS_ALIGNMENT

    def rows_per_logits_chunk(self, c4_cap: int) -> int:
        return max(1, C4_PREFILL_LOGITS_BUDGET_BYTES // (self.aligned_c4_cap(c4_cap) * 4))

    def logits(self, row_num: int, c4_cap: int) -> torch.Tensor:
        aligned_c4_cap = self.aligned_c4_cap(c4_cap)
        logits = self._scratch_view((row_num, aligned_c4_cap), torch.float32)
        return logits[:, :c4_cap]


class DeepseekV4Workspace:
    def __init__(self, model):
        self.token_capacity = int(model.batch_max_tokens)
        self.sliding_window = int(model.config["sliding_window"])
        args = get_env_start_args()
        self.vision_swa_width = self.sliding_window + int(model.config.get("vision_max_n_token", 0))
        self.swa_capacity = self.vision_swa_width
        if args.mtp_mode == "dspark":
            dspark_width = self.sliding_window + int(args.mtp_step)
            # FlashMLA sparse decode requires the physical top-k width to be block aligned.
            # 128 covers both supported padded Q-head configurations; swa_lengths keeps
            # the actual number of visible history + draft-block entries.
            self.swa_capacity = max(self.swa_capacity, ((dspark_width + 127) // 128) * 128)
        self.index_topk = int(model.config["index_topk"])
        self.c128_cap = self.compress_cap(model.max_seq_length, 128)
        overlap = args.enable_decode_microbatch_overlap or args.enable_prefill_microbatch_overlap
        self.microbatch_count = 1 + int(overlap)

        self.swa_indices = self._alloc(self.swa_capacity)
        self.swa_lengths = torch.empty((self.microbatch_count, self.token_capacity), dtype=torch.int32, device="cuda")
        self.dspark_swa_write_slots = torch.empty(
            (self.microbatch_count, self.token_capacity),
            dtype=torch.int32,
            device="cuda",
        )
        self.c4_indices = self._alloc(self.index_topk)
        self.c4_lengths = torch.empty((self.microbatch_count, self.token_capacity), dtype=torch.int32, device="cuda")
        self.c128_indices = self._alloc(self.c128_cap)
        self.c128_lengths = torch.empty((self.microbatch_count, self.token_capacity), dtype=torch.int32, device="cuda")
        self.flashmla_prefill_q = None
        self.flashmla_prefill_full_out = None
        self.flashmla_prefill_o_accum = None
        self.flashmla_prefill_lse_accum = None
        self.c4_prefill_aux = None

    def init_c4_prefill_aux(self, model):
        assert self.needs_c4_prefill_aux(model)
        if self.c4_prefill_aux is None:
            self.c4_prefill_aux = DeepseekV4C4PrefillWorkspace(
                token_capacity=self.token_capacity,
                max_seq_length=model.max_seq_length,
                max_request_num=model.max_req_num,
                index_n_heads=model.config["index_n_heads"],
                index_head_dim=model.config["index_head_dim"],
            )

    def init_flashmla_prefill_q(self, real_q_head_num: int, padded_q_head_num: int, head_dim: int, dtype: torch.dtype):
        if self.flashmla_prefill_q is None:
            self.flashmla_prefill_q = torch.empty(
                (self.token_capacity, padded_q_head_num, head_dim), dtype=dtype, device="cuda"
            )
            self.flashmla_prefill_q[:, real_q_head_num:, :].zero_()

    def init_flashmla_prefill_full_out(self, q_head_num: int, head_dim_v: int, dtype: torch.dtype):
        if self.flashmla_prefill_full_out is None:
            self.flashmla_prefill_full_out = torch.empty(
                (self.token_capacity, 1, q_head_num, head_dim_v), dtype=dtype, device="cuda"
            )

    def init_flashmla_prefill_split_kv_workspace(self, q_head_num: int, head_dim_v: int):
        if self.flashmla_prefill_o_accum is None:
            sm_count = torch.cuda.get_device_properties().multi_processor_count
            # FlashMLA stores one row per query plus at most one split row per SM.
            split_capacity = self.token_capacity + sm_count
            self.flashmla_prefill_o_accum = torch.empty(
                (split_capacity, 1, q_head_num, head_dim_v), dtype=torch.float32, device="cuda"
            )
            self.flashmla_prefill_lse_accum = torch.empty(
                (split_capacity, 1, q_head_num), dtype=torch.float32, device="cuda"
            )

    @staticmethod
    def compress_cap(max_kv_seq_len: int, ratio: int) -> int:
        return _compress_cap(max_kv_seq_len, ratio)

    @staticmethod
    def needs_c4_prefill_aux(model) -> bool:
        return (
            model.run_mode == "prefill"
            and os.getenv("LIGHTLLM_DSV4_PREFILL_OVERLAP", "1") == "1"
            and not model.args.enable_prefill_microbatch_overlap
            and 4 in model.config["compress_ratios"]
        )

    def _alloc(self, width: int) -> torch.Tensor:
        return torch.empty((self.microbatch_count, self.token_capacity * width), dtype=torch.int32, device="cuda")

    @staticmethod
    def _view(buffer: torch.Tensor, token_num: int, width: int) -> torch.Tensor:
        return torch.as_strided(buffer, (token_num, width), (width, 1))

    def swa(self, microbatch_index: int, token_num: int, width: int = None):
        width = self.sliding_window if width is None else int(width)
        assert width <= self.swa_capacity, f"swa width {width} exceeds allocated {self.swa_capacity}"
        return (
            self._view(self.swa_indices[microbatch_index], token_num, width),
            self.swa_lengths[microbatch_index, :token_num],
        )

    def dspark_swa(self, microbatch_index: int, token_num: int):
        indices, lengths = self.swa(
            microbatch_index,
            token_num,
            width=self.swa_capacity,
        )
        return (
            indices,
            lengths,
            self.dspark_swa_write_slots[microbatch_index, :token_num],
        )

    def c4(self, microbatch_index: int, token_num: int, width: int):
        assert width <= self.index_topk, f"c4 width {width} exceeds allocated {self.index_topk}"
        return (
            self._view(self.c4_indices[microbatch_index], token_num, width),
            self.c4_lengths[microbatch_index, :token_num],
        )

    def c128(self, microbatch_index: int, token_num: int, width: int):
        return (
            self._view(self.c128_indices[microbatch_index], token_num, width),
            self.c128_lengths[microbatch_index, :token_num],
        )
