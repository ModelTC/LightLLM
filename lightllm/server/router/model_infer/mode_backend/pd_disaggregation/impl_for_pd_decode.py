import torch
import torch.multiprocessing as mp
import threading
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from typing import List, Tuple, Dict
from lightllm.utils.infer_utils import set_random_seed
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq
from lightllm.server.core.objs import FinishStatus
from lightllm.utils.log_utils import init_logger
from lightllm.server.router.model_infer.mode_backend.generic_pre_process import prepare_decode_inputs, prepare_remote_prefill_inputs
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample

from .pd_remote_prefill_obj import (
    RemotePrefillTask,
    RemotePrefillServerInfo,
    RemotePrefillRequest)

logger = init_logger(__name__)


class PDBackendForDecodeNode(ModeBackend):
    def __init__(self,
                 prefill_task_queue: mp.Queue,
                 prefill_done_queue: mp.Queue,
                 mem_queue: mp.Queue) -> None:
        super().__init__()
        self.prefill_task_queue = prefill_task_queue
        self.prefill_done_queue = prefill_done_queue
        self.mem_queue = mem_queue
        self.remote_prefilled_reqs: Dict[str, InferReq] = {}

    def wait_prefill_done_loop(self):
        while True:
            prefill_done_id = self.prefill_done_queue.get()
            if prefill_done_id is None: # None means exit
                logger.info("wait prefill done loop exits")
                break
            if run_req := self.remote_prefilled_reqs.get(prefill_done_id, None):
                # remote prefill and transfer done, we need set kv cache to prompt len

                run_req.remote_prefilling = False
                self.remote_prefilled_reqs.pop(prefill_done_id)
            else:
                logger.warning(f"wait prefill done loop: cannot find run_req with id {prefill_done_id}")


    def init_custom(self):

        self.mem_queue.put((self.rank_in_dp, self.model.mem_manager.kv_buffer))

        threading.Thread(target=self.wait_prefill_done_loop, daemon=True).start()

        return

    def prefill(self, reqs: List[Tuple]):
        self._init_reqs(reqs, init_req_obj=False)
        return

    def _build_remote_prefill_task(self, index: int, kwargs: Dict, req: InferReq):
        prefill_node = req.shm_req.sample_params.move_kv_to_decode_node.to_dict()
        prefill_node_info = RemotePrefillServerInfo(
            perfill_server_id=prefill_node['node_id'],
            prefill_server_ip=prefill_node['ip'],
            prefill_server_port=prefill_node['rpyc_port'],
        )

        mem_indexes = kwargs.get('mem_indexes')
        b_start_loc = kwargs.get('b_start_loc')

        prefill_request = RemotePrefillRequest(
            group_request_id=req.shm_req.group_req_id,
            prompt = req.shm_req.get_prompt_ids(),
            sampling_params=req.shm_req.sample_params,
            multimodal_params=req.multimodal_params,
            local_cached_len=req.cur_kv_len,
            token_ids=mem_indexes[b_start_loc[index]: b_start_loc[index+1]],
        )
        return RemotePrefillTask(server_info=prefill_node_info, prefill_request=prefill_request)

    def _get_classed_reqs(self, req_ids: List[int], no_decode: bool = False):
        uninit_reqs, aborted_reqs, ok_finished_reqs, prefill_reqs, decode_reqs = super()._get_classed_reqs(
            req_ids,
            no_decode,
        )
        new_prefill_reqs = []
        # filter remote prefill requests
        for r in prefill_reqs:
            if r.remote_prefilling:
                continue
            new_prefill_reqs.append(r)
        return uninit_reqs, aborted_reqs, ok_finished_reqs, new_prefill_reqs, decode_reqs

    def decode(self):
        uninit_reqs, aborted_reqs, ok_finished_reqs, prefill_reqs, decode_reqs = self._get_classed_reqs(
            g_infer_context.infer_req_ids,
            no_decode=False,
        )

        self._filter_reqs(aborted_reqs)

        # allocate kv cache, do remote prefill
        if prefill_reqs:
            # TODO: we could allocate cache later after remote prefill done and get a signal from remote
            #       but it will have a risk to not have enough cache for this request.
            kwargs, run_reqs = prepare_remote_prefill_inputs(prefill_reqs)
            for idx, run_req in enumerate(run_reqs):
                run_req: InferReq = run_req
                # forward each req to remote prefill
                # since the token index are the same across TPs, we only need to trigger prefill on master
                if self.is_master_in_dp:
                    self.prefill_task_queue.put(self._build_remote_prefill_task(idx, kwargs, run_req))

                run_req.remote_prefilling = True
                self.remote_prefilled_reqs[run_req.req_id] = run_req

        if decode_reqs:
            kwargs, run_reqs = prepare_decode_inputs(decode_reqs)
            logits = self.model.forward(**kwargs)

            self._overlap_req_init_and_filter(
                uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True
            )

            next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

            self._post_handle(
                run_reqs, next_token_ids, next_token_logprobs, is_chuncked_mode=False, do_filter_finished_reqs=False
            )

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        return

