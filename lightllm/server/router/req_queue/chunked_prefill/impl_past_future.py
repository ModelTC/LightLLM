import bisect
from collections import deque
import random
from typing import List, Tuple
import numpy as np
from ...batch import Batch, Req
from .impl import ChunkedPrefillQueue


class PastFutureQueue(ChunkedPrefillQueue):
    WINDOW_SIZE = 200
    MINIMUM_SAMPLES = 200
    MAXIMUM_LISTS = 5
    REVERSED = 0.05
    COMPLIANCE_IS_BUSY_FLAG = False

    def __init__(self, args, router, dp_index, dp_size_in_node) -> None:
        super().__init__(args, router, dp_index, dp_size_in_node)
        initial_len = args.router_max_new_token_len
        self.history_output_len = deque([initial_len] * (self.WINDOW_SIZE // 2), maxlen=self.WINDOW_SIZE)

    def _sample_cache_list(self, reqs: List[Req], is_busy, samples=1) -> List[List[Tuple[int, int]]]:
        cache_len_lists = [[] for _ in range(samples)]
        his_Lo = sorted(self.history_output_len)
        for req in reqs:
            dl = req.shm_cur_output_len
            pos = bisect.bisect(his_Lo, dl)

            sample_range = [dl] + his_Lo[pos:] + [req.sample_params.max_new_tokens]  # at least 2 value

            for i in range(samples):
                random_p = np.random.random() * (len(sample_range) - 1)
                l_pos = int(random_p)
                l_val, r_val = sample_range[l_pos : l_pos + 2]

                # Linear interpolation
                sampled = round(l_val + (r_val - l_val) * (random_p - l_pos))
                cache_len_lists[i].append(
                    req.get_tuple_tokens(is_busy and self.COMPLIANCE_IS_BUSY_FLAG, sampled, has_out_len_factor=1.0)
                )

        return cache_len_lists

    def _calc_max_token_num_needed(self, cache_len_list: List[Tuple[int, int]]) -> int:
        cache_len_list.sort(key=lambda x: -x[1])

        left_out_len_array = np.array([e[1] for e in cache_len_list])
        has_run_len_array = np.array([e[0] for e in cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(cache_len_list) + 1, 1)

        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
        return need_max_token_num

    def _init_cache_list(self, current_batch: Batch, is_busy):
        if current_batch is not None:
            n_lists = min(self.MAXIMUM_LISTS, int(self.MINIMUM_SAMPLES / len(current_batch.reqs)) + 1)
            local_reqs = [req for req in current_batch.reqs if req.sample_params.suggested_dp_index == self.dp_index]
            self._cache_len_lists = self._sample_cache_list(local_reqs, is_busy, samples=n_lists)
        else:
            self._cache_len_lists = [[]]
        self.cache_len_list = self._cache_len_lists[0]  # keep compatibility

    def _update_cache_len_list(self, req: Req, is_busy):
        need_max_token_nums = []
        for li in self._cache_len_lists:
            newreq_output_len_sample = random.choice(self.history_output_len)
            li.append(
                req.get_tuple_tokens(
                    is_busy and self.COMPLIANCE_IS_BUSY_FLAG, newreq_output_len_sample, has_out_len_factor=1.0
                )
            )
            need_max_token_nums.append(self._calc_max_token_num_needed(li))
        need_max_token_num = np.max(need_max_token_nums)
        return need_max_token_num

    def record_finished_len_from_batch(self, batch: Batch):
        for req in batch.reqs:
            if req.shm_infer_released:
                self.history_output_len.append(req.shm_cur_output_len)
