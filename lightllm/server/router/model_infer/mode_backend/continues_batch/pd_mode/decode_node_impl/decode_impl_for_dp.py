import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import threading
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from typing import List, Tuple
from lightllm.utils.infer_utils import set_random_seed
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq, InferSamplingParams
from lightllm.server.core.objs import FinishStatus
from lightllm.server.pd_io_struct import UpKVStatus
from lightllm.utils.log_utils import init_logger
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
from .up_status import UpStatusManager
from rpyc.utils.server import ThreadedServer
from lightllm.common.basemodel.infer_lock import g_infer_state_lock
from .decode_task_cache import g_success_kv_move_task_cache, KVMoveTask
from lightllm.utils.device_utils import kv_trans_use_p2p
from lightllm.utils.envs_utils import get_unique_server_name, get_env_start_args
from .decode_impl import ContinuesBatchBackendForDecodeNode

logger = init_logger(__name__)


class DPForDecodeNode(ContinuesBatchBackendForDecodeNode):
    def __init__(self, info_queue: mp.Queue, mem_queue: mp.Queue) -> None:
        super().__init__(info_queue, mem_queue)
        self.enable_decode_microbatch_overlap = get_env_start_args().enable_decode_microbatch_overlap
        return

    def init_custom(self):
        super().init_custom()
        self.reduce_tensor = torch.tensor([0], dtype=torch.int32, device="cuda", requires_grad=False)
        # 这个地方预先进行一次 prefill 推理，主要是为了填充后续fake请求的第一个token位置，因为填充的decode请求
        # 在推理的时候至少是两个token，1个是已经有kv的token，一个是等待计算kv的token，然后生成第三个token，这几个
        # token 实际引用的都是 g_infer_context.req_manager.mem_manager.HOLD_TOKEN_MEMINDEX，但是需要初始化排除
        # nan 值，避免后续构建的fake请求在计算的过程中出现计算错误。
        from lightllm.server.router.model_infer.mode_backend.dp_backend.pre_process import padded_prepare_prefill_inputs

        kwargs, run_reqs, padded_req_num = padded_prepare_prefill_inputs([], 1, is_multimodal=self.is_multimodal)
        self.model.forward(**kwargs)
        assert len(run_reqs) == 0 and padded_req_num == 1

        return

    def prefill(self, reqs: List[Tuple]):
        self._init_reqs(reqs, init_req_obj=False)
        return

    def decode(self):
        uninit_reqs, aborted_reqs, ok_finished_reqs, prefill_reqs, decode_reqs = self._get_classed_reqs(
            g_infer_context.infer_req_ids
        )
        assert len(prefill_reqs) == 0

        self._filter_reqs(aborted_reqs)

        self.reduce_tensor.fill_(len(decode_reqs))
        dist.all_reduce(self.reduce_tensor, op=dist.ReduceOp.MAX, group=None, async_op=False)
        max_decode_num = self.reduce_tensor.item()
        if max_decode_num != 0:
            if not self.enable_decode_microbatch_overlap:
                self.normal_decode(decode_reqs, max_decode_num, uninit_reqs, ok_finished_reqs)
            else:
                self.overlap_decode(decode_reqs, max_decode_num, uninit_reqs, ok_finished_reqs)

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        return

    def normal_decode(self, decode_reqs: List[InferReq], max_decode_num: int, uninit_reqs, ok_finished_reqs):
        from lightllm.server.router.model_infer.mode_backend.dp_backend.pre_process import padded_prepare_decode_inputs

        kwargs, run_reqs, padded_req_num = padded_prepare_decode_inputs(
            decode_reqs, max_decode_num, is_multimodal=self.is_multimodal
        )
        logits = self.model.forward(**kwargs)
        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        if len(run_reqs) != 0:
            logits = logits[0 : len(run_reqs), :]
            next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
            self._post_handle(
                run_reqs, next_token_ids, next_token_logprobs, is_chuncked_mode=False, do_filter_finished_reqs=False
            )
        return

    def overlap_decode(self, decode_reqs: List[InferReq], max_decode_num: int, uninit_reqs, ok_finished_reqs):
        from lightllm.server.router.model_infer.mode_backend.dp_backend.pre_process import (
            padded_overlap_prepare_decode_inputs,
        )

        (
            micro_batch,
            run_reqs,
            padded_req_num,
            micro_batch1,
            run_reqs1,
            padded_req_num1,
        ) = padded_overlap_prepare_decode_inputs(decode_reqs, max_decode_num, is_multimodal=self.is_multimodal)

        logits, logits1 = self.model.microbatch_overlap_decode(micro_batch, micro_batch1)
        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        req_num, req_num1 = len(run_reqs), len(run_reqs1)
        all_logits = torch.empty((req_num + req_num1, logits.shape[1]), dtype=logits.dtype, device=logits.device)

        all_logits[0:req_num, :].copy_(logits[0:req_num, :], non_blocking=True)
        all_logits[req_num : (req_num + req_num1), :].copy_(logits1[0:req_num1, :], non_blocking=True)

        all_run_reqs = run_reqs + run_reqs1
        if all_run_reqs:
            next_token_ids, next_token_probs = sample(all_logits, all_run_reqs, self.eos_id)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
            self._post_handle(
                all_run_reqs, next_token_ids, next_token_logprobs, is_chuncked_mode=False, do_filter_finished_reqs=False
            )
        return
