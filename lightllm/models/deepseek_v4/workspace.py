import torch
from lightllm.utils.envs_utils import get_env_start_args


class DeepseekV4Workspace:
    def __init__(self, model):
        self.token_capacity = int(model.batch_max_tokens)
        self.sliding_window = int(model.config["sliding_window"])
        self.index_topk = int(model.config["index_topk"])
        self.c128_cap = self.compress_cap(model.max_seq_length, 128)
        args = get_env_start_args()
        overlap = args.enable_decode_microbatch_overlap or args.enable_prefill_microbatch_overlap
        self.microbatch_count = 1 + int(overlap)

        self.swa_indices = self._alloc(self.sliding_window)
        self.swa_lengths = torch.empty((self.microbatch_count, self.token_capacity), dtype=torch.int32, device="cuda")
        self.c4_indices = self._alloc(self.index_topk)
        self.c4_lengths = torch.empty((self.microbatch_count, self.token_capacity), dtype=torch.int32, device="cuda")
        self.c128_indices = self._alloc(self.c128_cap)
        self.c128_lengths = torch.empty((self.microbatch_count, self.token_capacity), dtype=torch.int32, device="cuda")

    @staticmethod
    def compress_cap(max_kv_seq_len: int, ratio: int) -> int:
        entries = max(1, int(max_kv_seq_len) // ratio)
        return ((entries + 63) // 64) * 64

    def _alloc(self, width: int) -> torch.Tensor:
        return torch.empty((self.microbatch_count, self.token_capacity * width), dtype=torch.int32, device="cuda")

    @staticmethod
    def _view(buffer: torch.Tensor, token_num: int, width: int) -> torch.Tensor:
        return torch.as_strided(buffer, (token_num, width), (width, 1))

    def swa(self, microbatch_index: int, token_num: int):
        return (
            self._view(self.swa_indices[microbatch_index], token_num, self.sliding_window),
            self.swa_lengths[microbatch_index, :token_num],
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
