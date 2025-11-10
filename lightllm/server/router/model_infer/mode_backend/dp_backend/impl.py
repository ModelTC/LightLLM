import torch
import time
import numpy as np
import os
import torch.nn.functional as F
import torch.distributed as dist
from typing import List, Tuple, Optional, Callable
from lightllm.common.kv_trans_kernel.kv_trans_v2 import kv_trans_for_dp
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.common.basemodel.batch_objs import ModelOutput, ModelInput
from lightllm.server.router.model_infer.infer_batch import InferSamplingParams, g_infer_context, InferReq
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
from lightllm.server.router.model_infer.mode_backend.pre import (
    padded_prepare_prefill_inputs,
    padded_prepare_decode_inputs,
    padded_overlap_prepare_prefill_inputs,
    padded_overlap_prepare_decode_inputs,
)
from lightllm.server.router.model_infer.mode_backend.overlap_events import OverlapEventPack
from lightllm.common.basemodel.infer_lock import g_infer_state_lock
from lightllm.server.router.model_infer.mode_backend.mtp_pre_process import (
    prepare_mtp_prefill_inputs,
)
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager
from lightllm.common.basemodel.triton_kernel.mtp_utils import mtp_scatter_next_token_ids
from .control_state import DPControlState
from lightllm.common.mem_manager import MemoryManager
import torch.multiprocessing as mp

min_trans_token_num = int(os.getenv("LIGHTLLM_MIN_TRANS_TOKEN_NUM", "512"))
dp_kv_transfer_req_num = int(os.getenv("LIGHTLLM_DP_KV_TRANSFER_REQ_NUM", "16"))


class DPChunkedPrefillBackend(ModeBackend):
    def __init__(self) -> None:
        super().__init__()

        # 用于控制每一步是执行prefill 和 decode 还是跳过
        self.control_state_machine = DPControlState(backend=self)
        self.enable_dp_prompt_cache_fetch = get_env_start_args().enable_dp_prompt_cache_fetch
        self.min_trans_token_num = min_trans_token_num

        # 在 mtp 模式下切换绑定的prefill 和 decode 函数
        if get_env_start_args().mtp_mode:
            self.is_mtp_eagle = get_env_start_args().mtp_mode == "deepseekv3_eagle"
            self.num_mtp_models = 1 if self.is_mtp_eagle else get_env_start_args().mtp_step
            if self.enable_prefill_microbatch_overlap:
                self.prefill = self.prefill_overlap_mtp
            else:
                self.prefill = self.prefill_mtp
            if self.enable_decode_microbatch_overlap:
                self.decode = self.decode_overlap_mtp
                self._draft_decode_overlap_func = (
                    self._draft_decode_eagle_overlap if self.is_mtp_eagle else self._draft_decode_vanilla_overlap
                )
            else:
                self.decode = self.decode_mtp
                self._draft_decode_func = self._draft_decode_eagle if self.is_mtp_eagle else self._draft_decode_vanilla
        else:
            if self.enable_prefill_microbatch_overlap:
                self.prefill = self.prefill_overlap
            else:
                self.prefill = self.prefill_normal

            if self.enable_decode_microbatch_overlap:
                self.decode = self.decode_overlap
            else:
                self.decode = self.decode_normal

        self.classed_req_strict_prefill = False
        return

    def init_custom(self):
        if self.enable_dp_prompt_cache_fetch:
            torch.cuda.set_device(get_current_device_id())

            from lightllm.server.router.model_infer.mode_backend.continues_batch.pd_mode.p2p_fix import reduce_tensor
            from lightllm.server.core.objs.shm_array import ShmArray
            from lightllm.utils.envs_utils import get_unique_server_name

            mp.reductions.reduce_tensor.__code__ = reduce_tensor.__code__

            # Create shared memory for mem_manager
            self.model.mem_manager.create_shm()

            # Create shared ShmArray for kv_indexes transfer
            # Use a small buffer to save shared memory
            self.dp_kv_transfer_req_num = dp_kv_transfer_req_num
            max_len = get_env_start_args().max_req_total_len + 8
            self.shared_kv_indexes_name = f"{get_unique_server_name()}_shared_kv_indexes_global"
            self.shared_kv_indexes = ShmArray(
                self.shared_kv_indexes_name, (self.dp_kv_transfer_req_num, max_len), dtype=np.int64
            )
            # Only rank_in_node == 0 creates the shared memory
            if self.rank_in_node == 0:
                self.shared_kv_indexes.create_shm()

            dist.barrier(group=self.node_nccl_group)

            # Collect mem_managers from all ranks
            self.mem_managers = []
            for rank_idx in range(self.node_world_size):
                if rank_idx != self.rank_in_node:
                    self.mem_managers.append(MemoryManager.from_shm(rank_idx))
                else:
                    self.mem_managers.append(self.model.mem_manager)

            # Other ranks link to the shared memory
            if self.rank_in_node != 0:
                self.shared_kv_indexes.link_shm()
            return

    def _init_reqs(self, reqs: List[Tuple]):
        my_reqs = reqs
        other_reqs = []
        if self.dp_size_in_node != 1:
            dp_rank_in_node = self.dp_rank_in_node
            my_reqs = [req for req in reqs if req[3] == dp_rank_in_node]
            other_reqs = [req for req in reqs if req[3] != dp_rank_in_node]

        g_infer_state_lock.acquire()
        infer_reqs = g_infer_context.add_reqs(my_reqs, init_prefix_cache=not self.enable_dp_prompt_cache_fetch)
        if self.enable_dp_prompt_cache_fetch:
            self._fetch_dp_prompt_cache(infer_reqs, other_reqs=other_reqs, origin_reqs=reqs)
            for r in infer_reqs:
                r._match_radix_cache()

        g_infer_state_lock.release()
        req_ids = [e[0] for e in my_reqs]
        return req_ids

    def _match_radix_cache(self, shm_req):
        input_token_ids = shm_req.shm_prompt_ids.arr[0 : shm_req.input_len]
        key = torch.tensor(input_token_ids, dtype=torch.int64, device="cpu")
        key = key[0 : len(key) - 1]  # 最后一个不需要，因为需要一个额外的token，让其在prefill的时候输出下一个token的值
        _, kv_len, value_tensor = g_infer_context.radix_cache.match_prefix(key, update_refs=False)
        return kv_len, value_tensor

    def _fetch_dp_prompt_cache(
        self, infer_reqs: List[InferReq], other_reqs: List[Tuple] = [], origin_reqs: List[Tuple] = []
    ):
        # shm_index_2_index = {r[1]: i for i, r in enumerate(origin_reqs)}
        my_match = []
        other_match = []
        # match all the reqs in this dp rank.
        for r in infer_reqs:
            if r.sampling_param.disable_prompt_cache:
                continue
            shm_req = r.shm_req

            kv_len, value_tensor = self._match_radix_cache(shm_req)
            # only the first rank is ok
            if self.rank_in_dp == 0:
                with g_infer_context.shm_req_manager.get_req_lock_by_index(shm_req.index_in_shm_mem):
                    shm_req.dp_origin_kv_len = kv_len
                    if kv_len > shm_req.dp_max_kv_len:
                        shm_req.dp_max_kv_len = kv_len
                        shm_req.dp_max_kv_rank = self.dp_rank_in_node  # 单机
            my_match.append((shm_req, kv_len, value_tensor))

        # match all the reqs in other dp ranks.
        if self.rank_in_dp == 0:
            for r in other_reqs:
                _, shm_index, _, _ = r
                shm_req = g_infer_context.shm_req_manager.get_req_obj_by_index(shm_index)
                sampling_param = InferSamplingParams(shm_req, g_infer_context.vocab_size)
                if sampling_param.disable_prompt_cache:
                    continue
                shm_req.link_prompt_ids_shm_array()

                kv_len, value_tensor = self._match_radix_cache(shm_req)
                with g_infer_context.shm_req_manager.get_req_lock_by_index(shm_req.index_in_shm_mem):
                    if kv_len > shm_req.dp_max_kv_len:
                        shm_req.dp_max_kv_len = kv_len
                        shm_req.dp_max_kv_rank = self.dp_rank_in_node  # 单机
                    other_match.append((shm_req, kv_len, value_tensor))

        # wait all the ranks to finish the match
        dist.barrier(group=self.node_nccl_group)

        # 创建 shm_index 到匹配结果的映射
        shm_index_to_match = {match[0].index_in_shm_mem: match for match in my_match}
        shm_index_to_match.update({match[0].index_in_shm_mem: match for match in other_match})

        my_trans_match = []
        other_trans_match = []
        transfer_count = 0
        for r in origin_reqs:
            _, shm_index, _, suggested_dp_index = r
            shm_req, kv_len, value_tensor = shm_index_to_match[shm_index]
            match = (shm_req, kv_len, value_tensor, suggested_dp_index)

            if suggested_dp_index != shm_req.dp_max_kv_rank:
                if suggested_dp_index == self.dp_rank_in_node:
                    my_trans_match.append((match, transfer_count))
                else:
                    other_trans_match.append((match, transfer_count))
                transfer_count += 1

                if transfer_count == self.dp_kv_transfer_req_num:
                    self._transfer_dp_kv_cache(my_trans_match, other_trans_match)
                    my_trans_match = []
                    other_trans_match = []
                    transfer_count = 0

        if transfer_count > 0:
            self._transfer_dp_kv_cache(my_trans_match, other_trans_match)

    def _transfer_dp_kv_cache(self, my_match: List[Tuple], other_match: List[Tuple]):
        other_shm_reqs = []
        for match, index in other_match:
            shm_req, kv_len, value_tensor, _ = match
            trans_len = kv_len - shm_req.dp_origin_kv_len
            if shm_req.dp_max_kv_rank == self.dp_rank_in_node:
                self.shared_kv_indexes.arr[index, 0:trans_len] = value_tensor[shm_req.dp_origin_kv_len : kv_len]
            other_shm_reqs.append(shm_req)

        self.release_all_shm_reqs(other_shm_reqs)
        dist.barrier(group=self.node_nccl_group)

        if not my_match:
            return

        move_token_indexes = []
        token_dp_indexes = []
        trans_info = []
        alloc_size = 0
        for match, index in my_match:
            shm_req, kv_len, value_tensor, _ = match
            trans_len = shm_req.dp_max_kv_len - kv_len
            if trans_len > 0 and shm_req.dp_max_kv_rank != self.dp_rank_in_node:
                move_token_indexes.extend(self.shared_kv_indexes.arr[index, 0:trans_len])
                token_dp_indexes.extend([shm_req.dp_max_kv_rank] * trans_len)
                trans_info.append((shm_req, kv_len, value_tensor, trans_len))
                alloc_size += trans_len

        if alloc_size < self.min_trans_token_num:
            return

        g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(alloc_size)
        mem_indexes = self.model.mem_manager.alloc(alloc_size).cuda()

        move_token_indexes = torch.tensor(move_token_indexes, dtype=torch.int64, device="cuda")
        token_dp_indexes = torch.tensor(token_dp_indexes, dtype=torch.int32, device="cuda")

        self.model.mem_manager.copy_kv_from_other_dp_ranks(
            mem_managers=self.mem_managers,
            move_token_indexes=move_token_indexes,
            token_dp_indexes=token_dp_indexes,
            mem_indexes=mem_indexes,
            dp_size_in_node=self.dp_size_in_node,
            rank_in_dp=self.rank_in_dp,
        )
        self.logger.info(f"dp_i {self.dp_rank_in_node} transfer kv tokens num: {alloc_size}")

        start_index = 0
        for shm_req, kv_len, value_tensor, trans_len in trans_info:
            new_value_tensor = mem_indexes[start_index : start_index + trans_len].cpu()
            start_index += trans_len
            key = torch.tensor(shm_req.shm_prompt_ids.arr[0 : shm_req.dp_max_kv_len], dtype=torch.int64, device="cpu")
            value_tensor = (
                torch.cat((value_tensor, new_value_tensor), dim=0) if value_tensor is not None else new_value_tensor
            )
            g_infer_context.radix_cache.insert(key, value_tensor)

    def release_all_shm_reqs(self, shm_reqs):
        for shm_req in shm_reqs:
            g_infer_context.shm_req_manager.put_back_req_obj(shm_req)

    def infer_loop(self):
        torch.cuda.set_device(get_current_device_id())
        try:
            while True:
                event_pack = self.overlap_event_manager.get_overlap_event_pack()
                if not self.support_overlap:
                    event_pack._close_overlap()

                event_pack.wait_to_forward()

                self._try_read_new_reqs()

                prefill_reqs, decode_reqs = self._get_classed_reqs(
                    no_decode=self.classed_req_no_decode,
                    strict_prefill=self.classed_req_strict_prefill,
                    recover_paused=self.control_state_machine.try_recover_paused_reqs(),
                )

                dp_prefill_req_nums, dp_decode_req_nums = self._dp_all_gather_prefill_and_decode_req_num(
                    prefill_reqs=prefill_reqs, decode_reqs=decode_reqs
                )

                run_way = self.control_state_machine.select_run_way(
                    dp_prefill_req_nums=dp_prefill_req_nums,
                    dp_decode_req_nums=dp_decode_req_nums,
                    prefill_reqs=prefill_reqs,
                    decode_reqs=decode_reqs,
                )

                if run_way.is_prefill():
                    self.prefill(
                        event_pack=event_pack,
                        prefill_reqs=prefill_reqs,
                    )
                    continue
                elif run_way.is_decode():
                    self.decode(
                        event_pack=event_pack,
                        decode_reqs=decode_reqs,
                    )
                    continue
                elif run_way.is_pass():
                    event_pack.notify_post_handle_and_wait_pre_post_handle()
                    event_pack.notify_forward_and_wait_post_handle()
                    event_pack.notify_pre_post_handle()
                    time.sleep(0.02)
                    continue

        except BaseException as e:
            self.logger.exception(str(e))
            raise e

    def prefill_normal(
        self,
        event_pack: OverlapEventPack,
        prefill_reqs: List[InferReq],
    ):
        model_input, run_reqs, _ = padded_prepare_prefill_inputs(prefill_reqs, is_multimodal=self.is_multimodal)
        run_reqs_num = len(run_reqs)
        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            model_output = self.model.forward(model_input)
            if run_reqs_num > 0:
                _, next_token_ids_cpu, next_token_logprobs_cpu = self._sample_and_scatter_token(
                    logits=model_output.logits[:run_reqs_num],
                    b_req_idx=model_input.b_req_idx[:run_reqs_num],
                    b_mtp_index=model_input.b_mtp_index[:run_reqs_num],
                    run_reqs=run_reqs,
                    is_prefill=True,
                    b_prefill_has_output_cpu=model_input.b_prefill_has_output_cpu[:run_reqs_num],
                    mask_func=None,
                )
                sync_event = torch.cuda.Event()
                sync_event.record()

        if run_reqs_num > 0:
            # 第二阶段
            event_pack.notify_post_handle_and_wait_pre_post_handle()
            update_packs = self._pre_post_handle(run_reqs, is_chuncked_mode=not self.disable_chunked_prefill)

            # 第三阶段
            event_pack.notify_forward_and_wait_post_handle()
            sync_event.synchronize()
            self._post_handle(
                run_reqs=run_reqs,
                next_token_ids=next_token_ids_cpu,
                next_token_logprobs=next_token_logprobs_cpu,
                run_reqs_update_packs=update_packs,
                extra_post_req_handle_func=self.extra_post_req_handle_func,
                nixl_prefill_chuncked_handle_func=self.nixl_prefill_chuncked_handle_func,
            )
            # 第四阶段
            event_pack.notify_pre_post_handle()
        else:
            event_pack.notify_post_handle_and_wait_pre_post_handle()
            event_pack.notify_forward_and_wait_post_handle()
            event_pack.notify_pre_post_handle()
        return

    def decode_normal(self, event_pack: OverlapEventPack, decode_reqs: List[InferReq]):
        model_input, run_reqs, padded_req_num = padded_prepare_decode_inputs(req_objs=decode_reqs)
        model_input: ModelInput = model_input
        run_reqs_num = len(run_reqs)
        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            model_output = self.model.forward(model_input)
            if run_reqs_num > 0:
                _, next_token_ids_cpu, next_token_logprobs_cpu = self._sample_and_scatter_token(
                    logits=model_output.logits[:run_reqs_num],
                    b_req_idx=model_input.b_req_idx[:run_reqs_num],
                    b_mtp_index=model_input.b_mtp_index[:run_reqs_num],
                    run_reqs=run_reqs,
                    is_prefill=False,
                    mask_func=None,
                )
                sync_event = torch.cuda.Event()
                sync_event.record()

        if run_reqs_num > 0:
            # 第二阶段
            event_pack.notify_post_handle_and_wait_pre_post_handle()
            update_packs = self._pre_post_handle(run_reqs, is_chuncked_mode=False)

            # 第三阶段
            event_pack.notify_forward_and_wait_post_handle()
            sync_event.synchronize()
            self._post_handle(
                run_reqs=run_reqs,
                next_token_ids=next_token_ids_cpu,
                next_token_logprobs=next_token_logprobs_cpu,
                run_reqs_update_packs=update_packs,
                extra_post_req_handle_func=self.extra_post_req_handle_func,
            )

            # 第四阶段
            event_pack.notify_pre_post_handle()
        else:
            event_pack.notify_post_handle_and_wait_pre_post_handle()
            event_pack.notify_forward_and_wait_post_handle()
            event_pack.notify_pre_post_handle()
        return

    def prefill_overlap(self, event_pack: OverlapEventPack, prefill_reqs: List[InferReq]):
        (
            model_input0,
            run_reqs0,
            _,
            model_input1,
            run_reqs1,
            _,
        ) = padded_overlap_prepare_prefill_inputs(prefill_reqs, is_multimodal=self.is_multimodal)

        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            model_output0, model_output1 = self.model.microbatch_overlap_prefill(model_input0, model_input1)
            logits0 = model_output0.logits
            logits1 = model_output1.logits

            req_num0, req_num1 = len(run_reqs0), len(run_reqs1)
            logits = torch.empty((req_num0 + req_num1, logits0.shape[1]), dtype=logits0.dtype, device=logits0.device)

            logits[0:req_num0, :].copy_(logits0[0:req_num0, :], non_blocking=True)
            logits[req_num0 : (req_num0 + req_num1), :].copy_(logits1[0:req_num1, :], non_blocking=True)

            run_reqs = run_reqs0 + run_reqs1
            b_has_out_cpu = (
                model_input0.b_prefill_has_output_cpu[0:req_num0] + model_input1.b_prefill_has_output_cpu[0:req_num1]
            )
            b_mtp_index = torch.cat((model_input0.b_mtp_index[0:req_num0], model_input1.b_mtp_index[0:req_num1]), dim=0)
            b_req_idx = torch.cat((model_input0.b_req_idx[0:req_num0], model_input1.b_req_idx[0:req_num1]), dim=0)

            if (req_num0 + req_num1) > 0:

                _, next_token_ids_cpu, next_token_logprobs_cpu = self._sample_and_scatter_token(
                    logits=logits,
                    b_req_idx=b_req_idx,
                    b_mtp_index=b_mtp_index,
                    run_reqs=run_reqs,
                    is_prefill=True,
                    b_prefill_has_output_cpu=b_has_out_cpu,
                    mask_func=None,
                )
                sync_event = torch.cuda.Event()
                sync_event.record()

        if (req_num0 + req_num1) > 0:
            # 第二阶段
            event_pack.notify_post_handle_and_wait_pre_post_handle()
            update_packs = self._pre_post_handle(run_reqs, is_chuncked_mode=not self.disable_chunked_prefill)

            # 第三阶段
            event_pack.notify_forward_and_wait_post_handle()
            sync_event.synchronize()

            self._post_handle(
                run_reqs=run_reqs,
                next_token_ids=next_token_ids_cpu,
                next_token_logprobs=next_token_logprobs_cpu,
                run_reqs_update_packs=update_packs,
                extra_post_req_handle_func=self.extra_post_req_handle_func,
                nixl_prefill_chuncked_handle_func=self.nixl_prefill_chuncked_handle_func,
            )
            # 第四阶段
            event_pack.notify_pre_post_handle()
        else:
            event_pack.notify_post_handle_and_wait_pre_post_handle()
            event_pack.notify_forward_and_wait_post_handle()
            event_pack.notify_pre_post_handle()
        return

    def decode_overlap(self, event_pack: OverlapEventPack, decode_reqs: List[InferReq]):
        (
            model_input0,
            run_reqs0,
            _,
            model_input1,
            run_reqs1,
            _,
        ) = padded_overlap_prepare_decode_inputs(req_objs=decode_reqs)
        model_input0: ModelInput = model_input0
        model_input1: ModelInput = model_input1

        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            model_output0, model_output1 = self.model.microbatch_overlap_decode(model_input0, model_input1)
            logits0 = model_output0.logits
            logits1 = model_output1.logits

            req_num0, req_num1 = len(run_reqs0), len(run_reqs1)
            logits = torch.empty((req_num0 + req_num1, logits0.shape[1]), dtype=logits0.dtype, device=logits0.device)

            logits[0:req_num0, :].copy_(logits0[0:req_num0, :], non_blocking=True)
            logits[req_num0 : (req_num0 + req_num1), :].copy_(logits1[0:req_num1, :], non_blocking=True)
            b_mtp_index = torch.cat((model_input0.b_mtp_index[0:req_num0], model_input1.b_mtp_index[0:req_num1]), dim=0)
            b_req_idx = torch.cat((model_input0.b_req_idx[0:req_num0], model_input1.b_req_idx[0:req_num1]), dim=0)

            run_reqs = run_reqs0 + run_reqs1
            if (req_num0 + req_num1) > 0:
                _, next_token_ids_cpu, next_token_logprobs_cpu = self._sample_and_scatter_token(
                    logits=logits,
                    b_req_idx=b_req_idx,
                    b_mtp_index=b_mtp_index,
                    run_reqs=run_reqs,
                    is_prefill=False,
                    mask_func=None,
                )
                sync_event = torch.cuda.Event()
                sync_event.record()

        if (req_num0 + req_num1) > 0:
            # 第二阶段
            event_pack.notify_post_handle_and_wait_pre_post_handle()
            update_packs = self._pre_post_handle(run_reqs, is_chuncked_mode=False)

            # 第三阶段
            event_pack.notify_forward_and_wait_post_handle()
            sync_event.synchronize()
            self._post_handle(
                run_reqs=run_reqs,
                next_token_ids=next_token_ids_cpu,
                next_token_logprobs=next_token_logprobs_cpu,
                run_reqs_update_packs=update_packs,
                extra_post_req_handle_func=self.extra_post_req_handle_func,
            )

            # 第四阶段
            event_pack.notify_pre_post_handle()
        else:
            event_pack.notify_post_handle_and_wait_pre_post_handle()
            event_pack.notify_forward_and_wait_post_handle()
            event_pack.notify_pre_post_handle()
        return

    def prefill_mtp(self, event_pack: OverlapEventPack, prefill_reqs: List[InferReq]):
        # main model prefill
        model_input, run_reqs, _ = padded_prepare_prefill_inputs(prefill_reqs, is_multimodal=self.is_multimodal)
        req_num = len(run_reqs)
        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            model_output: ModelOutput = self.model.forward(model_input)
            b_has_out_cpu = model_input.b_prefill_has_output_cpu[0:req_num]
            logits = model_output.logits[0:req_num, :]
            b_req_idx = model_input.b_req_idx[0:req_num]
            b_mtp_index = model_input.b_mtp_index[0:req_num]

            if req_num > 0:
                next_token_ids, next_token_ids_cpu, next_token_logprobs_cpu = self._sample_and_scatter_token(
                    logits=logits,
                    b_req_idx=b_req_idx,
                    b_mtp_index=b_mtp_index,
                    run_reqs=run_reqs,
                    is_prefill=True,
                    b_prefill_has_output_cpu=b_has_out_cpu,
                    mask_func=None,
                )

            # mtp kv fill
            draft_next_token_ids_gpu = torch.zeros((model_input.batch_size), dtype=torch.int64, device="cuda")
            if req_num > 0:
                draft_next_token_ids_gpu[0:req_num].copy_(next_token_ids)
            self._draft_prefill_forward(
                model_input=model_input,
                model_output=model_output,
                next_token_ids=draft_next_token_ids_gpu,
            )
            sync_event = torch.cuda.Event()
            sync_event.record()

        if req_num > 0:

            # 第二阶段
            event_pack.notify_post_handle_and_wait_pre_post_handle()
            update_packs = self._pre_post_handle(run_reqs, is_chuncked_mode=not self.disable_chunked_prefill)

            # 第三阶段
            event_pack.notify_forward_and_wait_post_handle()
            sync_event.synchronize()

            self._post_handle(
                run_reqs=run_reqs,
                next_token_ids=next_token_ids_cpu,
                next_token_logprobs=next_token_logprobs_cpu,
                run_reqs_update_packs=update_packs,
                extra_post_req_handle_func=self.extra_post_req_handle_func,
                nixl_prefill_chuncked_handle_func=self.nixl_prefill_chuncked_handle_func,
            )

            # 第四阶段
            event_pack.notify_pre_post_handle()
        else:
            event_pack.notify_post_handle_and_wait_pre_post_handle()
            event_pack.notify_forward_and_wait_post_handle()
            event_pack.notify_pre_post_handle()
        return

    def decode_mtp(self, event_pack: OverlapEventPack, decode_reqs: List[InferReq]):
        model_input, run_reqs, _ = padded_prepare_decode_inputs(decode_reqs)
        b_mtp_index_cpu = model_input.b_mtp_index
        req_num = len(run_reqs)

        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            model_output = self.model.forward(model_input)
            mtp_accept_len, b_req_mtp_start_loc, next_token_ids = None, None, None
            if req_num > 0:
                logits = model_output.logits[0:req_num, :]
                b_mtp_index_cpu = b_mtp_index_cpu[0:req_num]
                b_req_idx = model_input.b_req_idx[0:req_num]

                next_token_ids, next_token_logprobs = sample(logits, run_reqs, self.eos_id)
                next_token_ids_cpu, next_token_logprobs_cpu = self._async_copy_next_token_infos_to_pin_mem(
                    next_token_ids, next_token_logprobs
                )

                # verify the next_token_ids
                b_req_mtp_start_loc = [index for index, mtp_index in enumerate(b_mtp_index_cpu) if mtp_index == 0]
                b_req_mtp_start_loc = g_pin_mem_manager.gen_from_list(
                    key="b_req_mtp_start_loc",
                    data=b_req_mtp_start_loc,
                    dtype=torch.int32,
                ).cuda(non_blocking=True)

                mtp_accept_len, accepted_index = self._verify_mtp_v2(
                    new_next_token_ids=next_token_ids,
                    b_req_idx=b_req_idx,
                    b_req_mtp_start_loc=b_req_mtp_start_loc,
                )
                accepted_index_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor(
                    key="accepted_index",
                    gpu_tensor=accepted_index,
                )
                mtp_accept_len_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor(
                    key="mtp_accept_len",
                    gpu_tensor=mtp_accept_len,
                )

            verify_event = torch.cuda.Event()
            verify_event.record()

            eagle_mem_indexes_cpu = self._draft_decode_func(
                model_input=model_input,
                model_output=model_output,
                next_token_ids=next_token_ids,
                b_req_mtp_start_loc=b_req_mtp_start_loc,
                mtp_accept_len=mtp_accept_len,
                req_num=req_num,
            )
            if req_num > 0:
                g_infer_context.req_sampling_manager.update_reqs_out_token_counter_gpu(
                    b_req_idx=b_req_idx,
                    next_token_ids=next_token_ids,
                    mask=accepted_index == 1,
                )

            sync_event = torch.cuda.Event()
            sync_event.record()

        if req_num > 0:
            # 第二阶段
            event_pack.notify_post_handle_and_wait_pre_post_handle()
            verify_event.synchronize()
            verify_ok_reqs = [run_reqs[i] for i in range(len(run_reqs)) if accepted_index_cpu[i] == 1]
            update_packs = self._pre_post_handle(verify_ok_reqs, is_chuncked_mode=False)

            # 第三阶段
            event_pack.notify_forward_and_wait_post_handle()
            sync_event.synchronize()
            need_free_mem_indexes = model_input.mem_indexes_cpu[0:req_num][accepted_index_cpu == 0]
            if eagle_mem_indexes_cpu is not None:
                need_free_mem_indexes = torch.cat([need_free_mem_indexes, eagle_mem_indexes_cpu], dim=0)

            self._update_mtp_accept_ratio(decode_reqs=decode_reqs, mtp_accept_len_cpu=mtp_accept_len_cpu)
            select_mask = torch.tensor(accepted_index_cpu, dtype=torch.bool, device="cpu")
            self._post_handle(
                run_reqs=verify_ok_reqs,
                next_token_ids=next_token_ids_cpu[select_mask],
                next_token_logprobs=next_token_logprobs_cpu[select_mask],
                run_reqs_update_packs=update_packs,
                extra_post_req_handle_func=self.extra_post_req_handle_func,
            )
            if len(need_free_mem_indexes) > 0:
                g_infer_state_lock.acquire()
                g_infer_context.req_manager.mem_manager.free(need_free_mem_indexes)
                g_infer_state_lock.release()

            # 第四阶段
            event_pack.notify_pre_post_handle()
        else:
            event_pack.notify_post_handle_and_wait_pre_post_handle()
            event_pack.notify_forward_and_wait_post_handle()
            event_pack.notify_pre_post_handle()
        return

    def _draft_decode_vanilla(
        self,
        model_input: ModelInput,
        model_output: ModelOutput,
        next_token_ids: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        mtp_accept_len: torch.Tensor,
        req_num: int,
    ):
        all_next_token_ids = []
        # share some inference info with the main model
        draft_model_input = model_input
        draft_model_output = model_output
        draft_next_token_ids_gpu = torch.zeros((model_input.batch_size), dtype=torch.int64, device="cuda")
        if req_num > 0:
            draft_next_token_ids_gpu[:req_num].copy_(next_token_ids, non_blocking=True)

        all_next_token_ids.append(draft_next_token_ids_gpu)

        # process the draft model output
        for draft_model_idx in range(self.mtp_step):

            draft_model_input.input_ids = draft_next_token_ids_gpu
            draft_model_input.deepseekv3_mtp_draft_input_hiddens = draft_model_output.deepseekv3_mtp_main_output_hiddens
            # spec decode: MTP
            draft_model_output: ModelOutput = self.draft_models[draft_model_idx].forward(draft_model_input)
            draft_next_token_ids_gpu = self._gen_argmax_token_ids(draft_model_output)
            all_next_token_ids.append(draft_next_token_ids_gpu)

        if req_num > 0:
            all_next_token_ids = torch.stack(all_next_token_ids, dim=1)  # [batch_size, mtp_step + 1]
            all_next_token_ids = all_next_token_ids[0:req_num, :]
            mtp_scatter_next_token_ids(
                req_to_next_token_ids=self.model.req_manager.req_sampling_params_manager.req_to_next_token_ids,
                b_req_mtp_start_loc=b_req_mtp_start_loc,
                all_next_token_ids=all_next_token_ids,
                b_req_idx=model_input.b_req_idx[:req_num],
                mtp_accept_len=mtp_accept_len,
            )
        return None

    def _draft_decode_eagle(
        self,
        model_input: ModelInput,
        model_output: ModelOutput,
        next_token_ids: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        mtp_accept_len: torch.Tensor,
        req_num: int,
    ):
        all_next_token_ids = []
        # share some inference info with the main model
        draft_model_input = model_input
        draft_model_output = model_output
        all_next_token_ids.append(next_token_ids)
        draft_next_token_ids_gpu = torch.zeros((model_input.batch_size), dtype=torch.int64, device="cuda")
        if req_num > 0:
            draft_next_token_ids_gpu[:req_num].copy_(next_token_ids, non_blocking=True)

        real_req_num = req_num // (self.mtp_step + 1)
        padded_req_num = model_input.batch_size // (self.mtp_step + 1) - real_req_num
        eagle_mem_indexes_cpu = None

        g_infer_state_lock.acquire()
        if g_infer_context.radix_cache is not None:
            g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(real_req_num * self.mtp_step)
        eagle_mem_indexes_cpu = g_infer_context.req_manager.mem_manager.alloc(real_req_num * self.mtp_step)
        g_infer_state_lock.release()
        eagle_mem_indexes = eagle_mem_indexes_cpu.cuda(non_blocking=True)

        # process the draft model output
        for _step in range(self.mtp_step):

            draft_model_input.input_ids = draft_next_token_ids_gpu
            draft_model_input.deepseekv3_mtp_draft_input_hiddens = draft_model_output.deepseekv3_mtp_main_output_hiddens
            # spec decode: MTP
            draft_model_idx = _step % self.num_mtp_models
            draft_model_output: ModelOutput = self.draft_models[draft_model_idx].forward(draft_model_input)
            # update the meta info of the inference
            draft_model_input.b_seq_len += 1
            draft_model_input.max_len_in_batch += 1
            eagle_mem_indexes_i = eagle_mem_indexes[_step * real_req_num : (_step + 1) * real_req_num]
            eagle_mem_indexes_i = F.pad(
                input=eagle_mem_indexes_i,
                pad=(0, padded_req_num),
                mode="constant",
                value=g_infer_context.req_manager.mem_manager.HOLD_TOKEN_MEMINDEX,
            )
            draft_model_input.mem_indexes = torch.cat(
                [draft_model_input.mem_indexes.view(-1, self.mtp_step + 1)[:, 1:], eagle_mem_indexes_i.view(-1, 1)],
                dim=1,
            ).view(-1)
            draft_next_token_ids_gpu = self._gen_argmax_token_ids(draft_model_output)
            all_next_token_ids.append(draft_next_token_ids_gpu)

        if req_num > 0:
            all_next_token_ids = torch.stack(all_next_token_ids, dim=1)  # [batch_size, mtp_step + 1]
            all_next_token_ids = all_next_token_ids[0:req_num, :]
            mtp_scatter_next_token_ids(
                req_to_next_token_ids=self.model.req_manager.req_sampling_params_manager.req_to_next_token_ids,
                b_req_mtp_start_loc=b_req_mtp_start_loc,
                all_next_token_ids=all_next_token_ids,
                b_req_idx=model_input.b_req_idx[:req_num],
                mtp_accept_len=mtp_accept_len,
            )
        return eagle_mem_indexes_cpu

    def prefill_overlap_mtp(self, event_pack: OverlapEventPack, prefill_reqs: List[InferReq]):
        (
            model_input0,
            run_reqs0,
            _,
            model_input1,
            run_reqs1,
            _,
        ) = padded_overlap_prepare_prefill_inputs(prefill_reqs, is_multimodal=self.is_multimodal)
        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            model_output0, model_output1 = self.model.microbatch_overlap_prefill(model_input0, model_input1)
            logits0 = model_output0.logits
            logits1 = model_output1.logits
            req_num0, req_num1 = len(run_reqs0), len(run_reqs1)
            logits = torch.empty((req_num0 + req_num1, logits0.shape[1]), dtype=logits0.dtype, device=logits0.device)
            logits[0:req_num0, :].copy_(logits0[0:req_num0, :], non_blocking=True)
            logits[req_num0 : (req_num0 + req_num1), :].copy_(logits1[0:req_num1, :], non_blocking=True)

            run_reqs = run_reqs0 + run_reqs1
            b_has_out_cpu = (
                model_input0.b_prefill_has_output_cpu[0:req_num0] + model_input1.b_prefill_has_output_cpu[0:req_num1]
            )
            b_mtp_index = torch.cat((model_input0.b_mtp_index[0:req_num0], model_input1.b_mtp_index[0:req_num1]), dim=0)
            b_req_idx = torch.cat((model_input0.b_req_idx[0:req_num0], model_input1.b_req_idx[0:req_num1]), dim=0)

            if (req_num0 + req_num1) > 0:
                next_token_ids, next_token_ids_cpu, next_token_logprobs_cpu = self._sample_and_scatter_token(
                    logits=logits,
                    run_reqs=run_reqs,
                    b_req_idx=b_req_idx,
                    b_mtp_index=b_mtp_index,
                    is_prefill=True,
                    b_prefill_has_output_cpu=b_has_out_cpu,
                )

            # spec prefill: MTP
            draft_model_input0, draft_model_input1 = model_input0, model_input1
            draft_next_token_ids_gpu0 = torch.zeros((model_input0.batch_size), dtype=torch.int64, device="cuda")
            if req_num0 > 0:
                draft_next_token_ids_gpu0[0:req_num0].copy_(next_token_ids[0:req_num0], non_blocking=True)

            draft_next_token_ids_gpu1 = torch.zeros((model_input1.batch_size), dtype=torch.int64, device="cuda")
            if req_num1 > 0:
                draft_next_token_ids_gpu1[0:req_num1].copy_(
                    next_token_ids[req_num0 : (req_num0 + req_num1)], non_blocking=True
                )

            draft_model_output0, draft_model_output1 = model_output0, model_output1

            for draft_model_idx in range(self.num_mtp_models):

                draft_model_input0 = prepare_mtp_prefill_inputs(
                    model_input=draft_model_input0,
                    b_next_token_ids=draft_next_token_ids_gpu0,
                    deepseekv3_mtp_draft_input_hiddens=draft_model_output0.deepseekv3_mtp_main_output_hiddens,
                )

                draft_model_input1 = prepare_mtp_prefill_inputs(
                    model_input=draft_model_input1,
                    b_next_token_ids=draft_next_token_ids_gpu1,
                    deepseekv3_mtp_draft_input_hiddens=draft_model_output1.deepseekv3_mtp_main_output_hiddens,
                )

                draft_model_output0, draft_model_output1 = self.draft_models[
                    draft_model_idx
                ].microbatch_overlap_prefill(draft_model_input0, draft_model_input1)
                draft_next_token_ids_gpu0 = self._gen_argmax_token_ids(draft_model_output0)
                draft_next_token_ids_gpu1 = self._gen_argmax_token_ids(draft_model_output1)

            sync_event = torch.cuda.Event()
            sync_event.record()

        if req_num0 + req_num1 > 0:
            event_pack.notify_post_handle_and_wait_pre_post_handle()
            update_packs = self._pre_post_handle(run_reqs, is_chuncked_mode=not self.disable_chunked_prefill)

            event_pack.notify_forward_and_wait_post_handle()
            sync_event.synchronize()

            self._post_handle(
                run_reqs=run_reqs,
                next_token_ids=next_token_ids_cpu,
                next_token_logprobs=next_token_logprobs_cpu,
                run_reqs_update_packs=update_packs,
                extra_post_req_handle_func=self.extra_post_req_handle_func,
                nixl_prefill_chuncked_handle_func=self.nixl_prefill_chuncked_handle_func,
            )
            event_pack.notify_pre_post_handle()
        else:
            event_pack.notify_post_handle_and_wait_pre_post_handle()
            event_pack.notify_forward_and_wait_post_handle()
            event_pack.notify_pre_post_handle()
        return

    def decode_overlap_mtp(self, event_pack: OverlapEventPack, decode_reqs: List[InferReq]):
        (
            model_input0,
            run_reqs0,
            _,
            model_input1,
            run_reqs1,
            _,
        ) = padded_overlap_prepare_decode_inputs(decode_reqs)
        req_num0, req_num1 = len(run_reqs0), len(run_reqs1)
        all_next_token_ids = []
        b_mtp_index_cpu0 = model_input0.b_mtp_index
        b_mtp_index_cpu1 = model_input1.b_mtp_index
        with torch.cuda.stream(g_infer_context.get_overlap_stream()):

            model_output0, model_output1 = self.model.microbatch_overlap_decode(model_input0, model_input1)
            logits0 = model_output0.logits
            logits1 = model_output1.logits
            run_reqs = run_reqs0 + run_reqs1
            b_req_idx, mtp_accept_len, b_req_mtp_start_loc, next_token_ids = None, None, None, None
            if (req_num0 + req_num1) > 0:
                logits = torch.empty(
                    (req_num0 + req_num1, logits0.shape[1]), dtype=logits0.dtype, device=logits0.device
                )
                logits[0:req_num0, :].copy_(logits0[0:req_num0, :], non_blocking=True)
                logits[req_num0 : (req_num0 + req_num1), :].copy_(logits1[0:req_num1, :], non_blocking=True)
                next_token_ids, next_token_logprobs = sample(logits, run_reqs, self.eos_id)
                next_token_ids_cpu, next_token_logprobs_cpu = self._async_copy_next_token_infos_to_pin_mem(
                    next_token_ids, next_token_logprobs
                )

                b_req_idx = torch.cat((model_input0.b_req_idx[0:req_num0], model_input1.b_req_idx[0:req_num1]), dim=0)
                b_mtp_index_cpu = torch.cat((b_mtp_index_cpu0[0:req_num0], b_mtp_index_cpu1[0:req_num1]), dim=0)
                b_req_mtp_start_loc = [index for index, mtp_index in enumerate(b_mtp_index_cpu) if mtp_index == 0]
                b_req_mtp_start_loc = g_pin_mem_manager.gen_from_list(
                    key="b_req_mtp_start_loc",
                    data=b_req_mtp_start_loc,
                    dtype=torch.int32,
                ).cuda(non_blocking=True)

                mtp_accept_len, accepted_index = self._verify_mtp_v2(
                    new_next_token_ids=next_token_ids,
                    b_req_idx=b_req_idx,
                    b_req_mtp_start_loc=b_req_mtp_start_loc,
                )
                accepted_index_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor(
                    key="accepted_index",
                    gpu_tensor=accepted_index,
                )
                mtp_accept_len_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor(
                    key="mtp_accept_len",
                    gpu_tensor=mtp_accept_len,
                )
                all_next_token_ids.append(next_token_ids)

            verify_event = torch.cuda.Event()
            verify_event.record()

            eagle_mem_indexes_cpu = self._draft_decode_overlap_func(
                model_input0=model_input0,
                model_input1=model_input1,
                model_output0=model_output0,
                model_output1=model_output1,
                b_req_idx=b_req_idx,
                next_token_ids=next_token_ids,
                mtp_accept_len=mtp_accept_len,
                b_req_mtp_start_loc=b_req_mtp_start_loc,
                req_num0=req_num0,
                req_num1=req_num1,
            )

            if (req_num0 + req_num1) > 0:
                g_infer_context.req_sampling_manager.update_reqs_out_token_counter_gpu(
                    b_req_idx=b_req_idx,
                    next_token_ids=next_token_ids,
                    mask=accepted_index == 1,
                )
            sync_event = torch.cuda.Event()
            sync_event.record()

        if req_num0 + req_num1 > 0:
            event_pack.notify_post_handle_and_wait_pre_post_handle()
            verify_event.synchronize()
            verify_ok_reqs = [run_reqs[i] for i in range(len(run_reqs)) if accepted_index_cpu[i] == 1]
            update_packs = self._pre_post_handle(verify_ok_reqs, is_chuncked_mode=False)

            event_pack.notify_forward_and_wait_post_handle()
            sync_event.synchronize()
            mem_indexes_cpu = torch.cat(
                (model_input0.mem_indexes_cpu[0:req_num0], model_input1.mem_indexes_cpu[0:req_num1]), dim=0
            )
            need_free_mem_indexes = mem_indexes_cpu[accepted_index_cpu == 0]
            if eagle_mem_indexes_cpu is not None:
                need_free_mem_indexes = torch.cat((need_free_mem_indexes, eagle_mem_indexes_cpu), dim=0)

            self._update_mtp_accept_ratio(decode_reqs=decode_reqs, mtp_accept_len_cpu=mtp_accept_len_cpu)
            select_mask = torch.tensor(accepted_index_cpu, dtype=torch.bool, device="cpu")
            self._post_handle(
                run_reqs=verify_ok_reqs,
                next_token_ids=next_token_ids_cpu[select_mask],
                next_token_logprobs=next_token_logprobs_cpu[select_mask],
                run_reqs_update_packs=update_packs,
                extra_post_req_handle_func=self.extra_post_req_handle_func,
            )
            if len(need_free_mem_indexes) > 0:
                g_infer_state_lock.acquire()
                g_infer_context.req_manager.mem_manager.free(need_free_mem_indexes)
                g_infer_state_lock.release()
            event_pack.notify_pre_post_handle()
        else:
            event_pack.notify_post_handle_and_wait_pre_post_handle()
            event_pack.notify_forward_and_wait_post_handle()
            event_pack.notify_pre_post_handle()
        return

    def _draft_prefill_forward(self, model_input: ModelInput, model_output: ModelOutput, next_token_ids: torch.Tensor):
        # spec prefill: MTP, 这个地方只是为了填充draft model的 kv， 并不会使用生成的token_id。
        draft_model_input = model_input
        draft_model_output = model_output
        draft_next_token_ids_gpu = next_token_ids
        for draft_model_idx in range(self.num_mtp_models):
            draft_model_input = prepare_mtp_prefill_inputs(
                model_input=draft_model_input,
                b_next_token_ids=draft_next_token_ids_gpu,
                deepseekv3_mtp_draft_input_hiddens=draft_model_output.deepseekv3_mtp_main_output_hiddens,
            )
            draft_model_output = self.draft_models[draft_model_idx].forward(draft_model_input)
            draft_next_token_ids_gpu = self._gen_argmax_token_ids(draft_model_output)
        return

    def _draft_decode_vanilla_overlap(
        self,
        model_input0: ModelInput,
        model_input1: ModelInput,
        model_output0: ModelOutput,
        model_output1: ModelOutput,
        b_req_idx: torch.Tensor,
        next_token_ids: torch.Tensor = None,
        mtp_accept_len: torch.Tensor = None,
        b_req_mtp_start_loc: torch.Tensor = None,
        req_num0: int = 0,
        req_num1: int = 0,
    ):
        all_next_token_ids = []
        all_next_token_ids.append(next_token_ids)
        # share some inference info with the main model
        draft_model_input0, draft_model_input1 = model_input0, model_input1
        draft_model_output0, draft_model_output1 = model_output0, model_output1

        draft_next_token_ids_gpu0 = torch.zeros((model_input0.batch_size), dtype=torch.int64, device="cuda")
        draft_next_token_ids_gpu1 = torch.zeros((model_input1.batch_size), dtype=torch.int64, device="cuda")
        if req_num0 > 0:
            draft_next_token_ids_gpu0[0:req_num0].copy_(next_token_ids[0:req_num0], non_blocking=True)
        if req_num1 > 1:
            draft_next_token_ids_gpu1[0:req_num1].copy_(
                next_token_ids[req_num0 : (req_num0 + req_num1)], non_blocking=True
            )

        # process the draft model output
        for draft_model_idx in range(self.mtp_step):

            draft_model_input0.input_ids = draft_next_token_ids_gpu0
            draft_model_input0.deepseekv3_mtp_draft_input_hiddens = (
                draft_model_output0.deepseekv3_mtp_main_output_hiddens
            )
            draft_model_input1.input_ids = draft_next_token_ids_gpu1
            draft_model_input1.deepseekv3_mtp_draft_input_hiddens = (
                draft_model_output1.deepseekv3_mtp_main_output_hiddens
            )

            draft_model_output0, draft_model_output1 = self.draft_models[draft_model_idx].microbatch_overlap_decode(
                draft_model_input0, draft_model_input1
            )

            draft_next_token_ids_gpu0 = self._gen_argmax_token_ids(draft_model_output0)
            draft_next_token_ids_gpu1 = self._gen_argmax_token_ids(draft_model_output1)
            draft_next_token_ids = torch.cat(
                (draft_next_token_ids_gpu0[0:req_num0], draft_next_token_ids_gpu1[0:req_num1]), dim=0
            )
            all_next_token_ids.append(draft_next_token_ids)

        if req_num0 + req_num1 > 0:
            all_next_token_ids = torch.stack(all_next_token_ids, dim=1)
            mtp_scatter_next_token_ids(
                req_to_next_token_ids=self.model.req_manager.req_sampling_params_manager.req_to_next_token_ids,
                b_req_mtp_start_loc=b_req_mtp_start_loc,
                all_next_token_ids=all_next_token_ids,
                b_req_idx=b_req_idx,
                mtp_accept_len=mtp_accept_len,
            )
        return None

    def _draft_decode_eagle_overlap(
        self,
        model_input0: ModelInput,
        model_input1: ModelInput,
        model_output0: ModelOutput,
        model_output1: ModelOutput,
        b_req_idx: torch.Tensor,
        next_token_ids: torch.Tensor = None,
        mtp_accept_len: torch.Tensor = None,
        b_req_mtp_start_loc: torch.Tensor = None,
        req_num0: int = 0,
        req_num1: int = 0,
    ):
        all_next_token_ids = []
        all_next_token_ids.append(next_token_ids)
        # share some inference info with the main model
        draft_model_input0, draft_model_input1 = model_input0, model_input1
        draft_model_output0, draft_model_output1 = model_output0, model_output1

        draft_next_token_ids_gpu0 = torch.zeros((model_input0.batch_size), dtype=torch.int64, device="cuda")
        draft_next_token_ids_gpu1 = torch.zeros((model_input1.batch_size), dtype=torch.int64, device="cuda")
        if req_num0 > 0:
            draft_next_token_ids_gpu0[0:req_num0].copy_(next_token_ids[0:req_num0], non_blocking=True)
        if req_num1 > 1:
            draft_next_token_ids_gpu1[0:req_num1].copy_(
                next_token_ids[req_num0 : (req_num0 + req_num1)], non_blocking=True
            )
        real_req_num0 = req_num0 // (self.mtp_step + 1)
        real_req_num1 = req_num1 // (self.mtp_step + 1)
        real_req_num = real_req_num0 + real_req_num1
        padded_req_num0 = model_input0.batch_size // (self.mtp_step + 1) - real_req_num0
        padded_req_num1 = model_input1.batch_size // (self.mtp_step + 1) - real_req_num1
        g_infer_state_lock.acquire()
        if g_infer_context.radix_cache is not None:
            g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(real_req_num * self.mtp_step)
        eagle_mem_indexes_cpu = g_infer_context.req_manager.mem_manager.alloc(real_req_num * self.mtp_step)
        g_infer_state_lock.release()
        eagle_mem_indexes = eagle_mem_indexes_cpu.cuda(non_blocking=True)
        eagle_mem_indexes0 = eagle_mem_indexes[0 : real_req_num0 * self.mtp_step]
        eagle_mem_indexes1 = eagle_mem_indexes[real_req_num0 * self.mtp_step : real_req_num * self.mtp_step]

        # process the draft model output
        for _step in range(self.mtp_step):

            draft_model_input0.input_ids = draft_next_token_ids_gpu0
            draft_model_input0.deepseekv3_mtp_draft_input_hiddens = (
                draft_model_output0.deepseekv3_mtp_main_output_hiddens
            )
            draft_model_input1.input_ids = draft_next_token_ids_gpu1
            draft_model_input1.deepseekv3_mtp_draft_input_hiddens = (
                draft_model_output1.deepseekv3_mtp_main_output_hiddens
            )

            draft_model_idx = _step % self.num_mtp_models
            draft_model_output0, draft_model_output1 = self.draft_models[draft_model_idx].microbatch_overlap_decode(
                draft_model_input0, draft_model_input1
            )

            draft_model_input0.b_seq_len += 1
            draft_model_input0.max_len_in_batch += 1
            eagle_mem_indexes_i = eagle_mem_indexes0[_step * real_req_num0 : (_step + 1) * real_req_num0]
            eagle_mem_indexes_i = F.pad(
                input=eagle_mem_indexes_i,
                pad=(0, padded_req_num0),
                mode="constant",
                value=g_infer_context.req_manager.mem_manager.HOLD_TOKEN_MEMINDEX,
            )
            draft_model_input0.mem_indexes = torch.cat(
                [draft_model_input0.mem_indexes.view(-1, self.mtp_step + 1)[:, 1:], eagle_mem_indexes_i.view(-1, 1)],
                dim=1,
            ).view(-1)

            draft_model_input1.b_seq_len += 1
            draft_model_input1.max_len_in_batch += 1
            eagle_mem_indexes_i = eagle_mem_indexes1[_step * real_req_num1 : (_step + 1) * real_req_num1]
            eagle_mem_indexes_i = F.pad(
                input=eagle_mem_indexes_i,
                pad=(0, padded_req_num1),
                mode="constant",
                value=g_infer_context.req_manager.mem_manager.HOLD_TOKEN_MEMINDEX,
            )
            draft_model_input1.mem_indexes = torch.cat(
                [draft_model_input1.mem_indexes.view(-1, self.mtp_step + 1)[:, 1:], eagle_mem_indexes_i.view(-1, 1)],
                dim=1,
            ).view(-1)

            draft_next_token_ids_gpu0 = self._gen_argmax_token_ids(draft_model_output0)
            draft_next_token_ids_gpu1 = self._gen_argmax_token_ids(draft_model_output1)
            draft_next_token_ids = torch.cat(
                (draft_next_token_ids_gpu0[0:req_num0], draft_next_token_ids_gpu1[0:req_num1]), dim=0
            )
            all_next_token_ids.append(draft_next_token_ids)

        if req_num0 + req_num1 > 0:
            all_next_token_ids = torch.stack(all_next_token_ids, dim=1)
            mtp_scatter_next_token_ids(
                req_to_next_token_ids=self.model.req_manager.req_sampling_params_manager.req_to_next_token_ids,
                b_req_mtp_start_loc=b_req_mtp_start_loc,
                all_next_token_ids=all_next_token_ids,
                b_req_idx=b_req_idx,
                mtp_accept_len=mtp_accept_len,
            )
        return eagle_mem_indexes_cpu
