import os
import time
import threading
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from typing import List, Tuple
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.utils.infer_utils import set_random_seed
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.server.router.model_infer.infer_batch import InferReq, InferSamplingParams, g_infer_context
from lightllm.server.core.objs import FinishStatus
from lightllm.server.pd_io_struct import KVMoveTask, DecodeNodeInfo
from lightllm.utils.log_utils import init_logger
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
from lightllm.common.basemodel.infer_lock import g_infer_state_lock
from rpyc.utils.server import ThreadedServer
from .prefill_task_cache import g_kv_move_task_cache
from lightllm.utils.device_utils import kv_trans_use_p2p
from lightllm.utils.envs_utils import get_unique_server_name, get_env_start_args
from .prefill_impl import ChunckedPrefillForPrefillNode
from lightllm.server.router.model_infer.mode_backend.dp_backend.impl import DPChunkedPrefillBackend

logger = init_logger(__name__)


class DPChunkedForPrefillNode(ChunckedPrefillForPrefillNode):
    def __init__(self, info_queue: mp.Queue, mem_queue: mp.Queue) -> None:
        super().__init__(info_queue=info_queue, mem_queue=mem_queue)
        self.enable_decode_microbatch_overlap = get_env_start_args().enable_decode_microbatch_overlap

    def init_custom(self):
        super().init_custom()
        self.reduce_tensor = torch.tensor([0], dtype=torch.int32, device="cuda", requires_grad=False)
        return

    def prefill(self, reqs: List[Tuple]):
        self._init_reqs(reqs)
        return

    def decode(self):
        uinit_reqs, aborted_reqs, ok_finished_reqs, prefill_reqs, decode_reqs = self._get_classed_reqs(
            g_infer_context.infer_req_ids,
            no_decode=True,
        )
        assert len(uinit_reqs) == 0
        assert len(decode_reqs) == 0

        self._filter_reqs(aborted_reqs)

        if ok_finished_reqs:
            self.prefill_req_frozen_tokens_and_put_to_kvmove_taskqueue(ok_finished_reqs)
            self._filter_reqs(ok_finished_reqs)

        # 进行 chuncked prefill
        current_dp_prefill_num = len(prefill_reqs)
        self.reduce_tensor.fill_(current_dp_prefill_num)
        dist.all_reduce(self.reduce_tensor, op=dist.ReduceOp.MAX, group=None, async_op=False)
        max_prefill_num = self.reduce_tensor.item()
        if max_prefill_num != 0:
            from lightllm.server.router.model_infer.mode_backend.dp_backend.pre_process import (
                padded_prepare_prefill_inputs,
            )

            kwargs, run_reqs, padded_req_num = padded_prepare_prefill_inputs(
                prefill_reqs, max_prefill_num, is_multimodal=False
            )
            logits = self.model.forward(**kwargs)
            if len(run_reqs) != 0:
                logits = logits[0 : len(run_reqs), :]
                next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
                next_token_ids = next_token_ids.detach().cpu().numpy()
                next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
                self._post_handle(
                    run_reqs, next_token_ids, next_token_logprobs, is_chuncked_mode=True, do_filter_finished_reqs=False
                )
        return
