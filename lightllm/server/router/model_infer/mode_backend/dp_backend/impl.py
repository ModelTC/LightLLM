import torch
import time
from typing import List, Tuple
from lightllm.common.basemodel.triton_kernel.mtp_utils import gen_b_req_mtp_start_loc
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.common.basemodel.batch_objs import ModelOutput, ModelInput
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
from lightllm.server.router.model_infer.mode_backend.pre import (
    prepare_prefill_inputs,
    prepare_decode_inputs,
    overlap_prepare_prefill_inputs,
    overlap_prepare_decode_inputs,
)
from lightllm.server.router.model_infer.mode_backend.overlap_events import OverlapEventPack
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager
from lightllm.server.router.model_infer.mtp_speculative.engine import SpecEngine
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_engine import DPOverlapSpecEngine
from lightllm.server.router.model_infer.mtp_speculative import utils as mtp_utils
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import MtpMemIndexesToFree
from .control_state import DPControlState


class DPChunkedPrefillBackend(ModeBackend):
    def __init__(self) -> None:
        super().__init__()

        # 用于控制每一步是执行prefill 和 decode 还是跳过
        self.control_state_machine = DPControlState(backend=self)

        # 在 mtp 模式下切换绑定的prefill 和 decode 函数
        spec_mode = get_env_start_args().mtp_mode
        if spec_mode is not None:
            if spec_mode in ("dspark", "dflash"):
                raise NotImplementedError("DP backend does not support DFlash/DSpark parallel block drafting yet.")
            if self.enable_prefill_microbatch_overlap:
                self.prefill = self.prefill_overlap_mtp
            else:
                self.prefill = self.prefill_mtp
            if self.enable_decode_microbatch_overlap:
                self.decode = self.decode_overlap_mtp
            else:
                self.decode = self.decode_mtp
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

    def init_spec_engine(self):
        engine_kwargs = dict(
            backend=self,
            spec_mode=self.args.mtp_mode,
            enable_dynmaic_mtp=self.args.mtp_dynamic_verify,
        )
        # 非 overlap DP 与普通后端复用同一个 SpecEngine。
        self.spec_engine = SpecEngine(
            backend=self,
            spec_mode=self.args.mtp_mode,
            enable_dynmaic_mtp=self.args.mtp_dynamic_verify,
        )

        self.dp_overlap_spec_engine = DPOverlapSpecEngine(
            **engine_kwargs,
            common_engine=self.spec_engine,
        )
        self.prefill_draft_engine = (
            self.dp_overlap_spec_engine if self.enable_prefill_microbatch_overlap else self.spec_engine
        )
        self.decode_draft_engine = (
            self.dp_overlap_spec_engine if self.enable_decode_microbatch_overlap else self.spec_engine
        )
        return

    def _init_reqs(self, reqs: List[Tuple]):
        if not self.args.enable_dp_prompt_cache_fetch:
            return super()._init_reqs(reqs)

        dp_rank_in_node = self.dp_rank_in_node
        current_dp_reqs = [req for req in reqs if req[3] == dp_rank_in_node]
        other_dp_reqs = [req for req in reqs if req[3] != dp_rank_in_node]

        infer_reqs = g_infer_context.add_reqs(reqs, init_prefix_cache=True)
        req_dp_ranks = [req[3] for req in reqs]
        self.dp_kv_shared_module.fill_reqs_info(reqs=infer_reqs)
        trans_taskes = self.dp_kv_shared_module.build_shared_kv_trans_tasks(reqs=infer_reqs, req_dp_ranks=req_dp_ranks)
        self.dp_kv_shared_module.kv_trans(trans_tasks=trans_taskes)

        # other_dp_reqs 只是为本 DP 做完 prefix cache / KV 拉取后的临时本地对象，
        # 真正推理仍在其归属 DP 上进行。这里仅清理本 DP 的 InferReq 与 KV 引用，
        # 不能写 shm_infer_released / final token metadata，否则会误把归属 DP
        # 上尚未结束的请求标记为已完成。所以设置 modify_shm_finish_state 为 False。
        g_infer_context._filter(
            finished_request_ids=[req[0] for req in other_dp_reqs],
            modify_shm_finish_state=False,
        )

        req_ids = [e[0] for e in current_dp_reqs]

        if self.args.enable_cpu_cache:
            self._load_cpu_cache_to_reqs(req_ids=req_ids)

        return req_ids

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
                    # 进行一次流同步，保证 _try_read_new_reqs 中的一些算子操作，必然已经完成。
                    # 防止后续的推理流程读取到显存中可能存在错误的数据。
                    g_infer_context.get_overlap_stream().wait_stream(torch.cuda.current_stream())
                    self.prefill(
                        event_pack=event_pack,
                        prefill_reqs=prefill_reqs,
                    )
                    continue
                elif run_way.is_decode():
                    # 进行一次流同步，保证 _try_read_new_reqs 中的一些算子操作，必然已经完成。
                    # 防止后续的推理流程读取到显存中可能存在错误的数据。
                    g_infer_context.get_overlap_stream().wait_stream(torch.cuda.current_stream())
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
        model_input, run_reqs = prepare_prefill_inputs(prefill_reqs, is_chuncked_mode=not self.disable_chunked_prefill)
        run_reqs_num = len(run_reqs)
        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            model_output = self.model.forward(model_input)
            self._capture_prompt_logprobs_if_needed(model_input, run_reqs, model_output.prompt_logics)
            if run_reqs_num > 0:
                (
                    _,
                    next_token_ids_cpu,
                    next_token_logprobs_cpu,
                    next_token_ranks_cpu,
                ) = self._sample_and_scatter_token(
                    logits=model_output.logits,
                    b_req_idx=model_input.b_req_idx,
                    b_mtp_index=model_input.b_mtp_index,
                    run_reqs=run_reqs,
                    is_prefill=True,
                    b_prefill_has_output_cpu=model_input.b_prefill_has_output_cpu,
                    mask_func=None,
                )
                g_infer_context.copy_linear_att_state_to_cache_buffer(
                    b_req_idx=model_input.b_req_idx,
                    reqs=run_reqs,
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
                next_token_ranks=next_token_ranks_cpu,
                run_reqs_update_packs=update_packs,
                extra_post_req_handle_func=self.extra_post_req_handle_func,
                pd_prefill_chunked_handle_func=self.pd_prefill_chunked_handle_func,
            )
            # 第四阶段
            event_pack.notify_pre_post_handle()
        else:
            event_pack.notify_post_handle_and_wait_pre_post_handle()
            event_pack.notify_forward_and_wait_post_handle()
            event_pack.notify_pre_post_handle()
        return

    def decode_normal(self, event_pack: OverlapEventPack, decode_reqs: List[InferReq]):
        model_input, run_reqs = prepare_decode_inputs(req_objs=decode_reqs)
        model_input: ModelInput = model_input
        run_reqs_num = len(run_reqs)
        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            model_output = self.model.forward(model_input)
            if run_reqs_num > 0:
                (
                    _,
                    next_token_ids_cpu,
                    next_token_logprobs_cpu,
                    next_token_ranks_cpu,
                ) = self._sample_and_scatter_token(
                    logits=model_output.logits,
                    b_req_idx=model_input.b_req_idx,
                    b_mtp_index=model_input.b_mtp_index,
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
                next_token_ranks=next_token_ranks_cpu,
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
            model_input1,
            run_reqs1,
        ) = overlap_prepare_prefill_inputs(prefill_reqs)

        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            model_output0, model_output1 = self.model.microbatch_overlap_prefill(model_input0, model_input1)
            self._capture_prompt_logprobs_if_needed(model_input0, run_reqs0, model_output0.prompt_logics)
            self._capture_prompt_logprobs_if_needed(model_input1, run_reqs1, model_output1.prompt_logics)
            logits0 = model_output0.logits
            logits1 = model_output1.logits

            req_num0, req_num1 = len(run_reqs0), len(run_reqs1)
            logits = torch.empty((req_num0 + req_num1, logits0.shape[1]), dtype=logits0.dtype, device=logits0.device)
            logits[0:req_num0, :].copy_(logits0, non_blocking=True)
            logits[req_num0 : req_num0 + req_num1, :].copy_(logits1, non_blocking=True)

            run_reqs = run_reqs0 + run_reqs1
            b_has_out_cpu = model_input0.b_prefill_has_output_cpu + model_input1.b_prefill_has_output_cpu
            b_mtp_index = torch.cat((model_input0.b_mtp_index, model_input1.b_mtp_index), dim=0)
            b_req_idx = torch.cat((model_input0.b_req_idx, model_input1.b_req_idx), dim=0)

            if req_num0 + req_num1 > 0:
                (
                    _,
                    next_token_ids_cpu,
                    next_token_logprobs_cpu,
                    next_token_ranks_cpu,
                ) = self._sample_and_scatter_token(
                    logits=logits,
                    b_req_idx=b_req_idx,
                    b_mtp_index=b_mtp_index,
                    run_reqs=run_reqs,
                    is_prefill=True,
                    b_prefill_has_output_cpu=b_has_out_cpu,
                    mask_func=None,
                )

                if g_infer_context.is_linear_att_mixed_model:
                    g_infer_context.copy_linear_att_state_to_cache_buffer(b_req_idx=b_req_idx, reqs=run_reqs)

                sync_event = torch.cuda.Event()
                sync_event.record()

        if req_num0 + req_num1 > 0:
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
                next_token_ranks=next_token_ranks_cpu,
                run_reqs_update_packs=update_packs,
                extra_post_req_handle_func=self.extra_post_req_handle_func,
                pd_prefill_chunked_handle_func=self.pd_prefill_chunked_handle_func,
            )
            # 第四阶段
            event_pack.notify_pre_post_handle()
        else:
            event_pack.notify_post_handle_and_wait_pre_post_handle()
            event_pack.notify_forward_and_wait_post_handle()
            event_pack.notify_pre_post_handle()
        return

    def decode_overlap(self, event_pack: OverlapEventPack, decode_reqs: List[InferReq]):
        model_input0, run_reqs0, _, model_input1, run_reqs1, _ = overlap_prepare_decode_inputs(req_objs=decode_reqs)
        run_reqs = run_reqs0 + run_reqs1
        req_num0, req_num1 = len(run_reqs0), len(run_reqs1)

        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            model_output0, model_output1 = self.model.microbatch_overlap_decode(model_input0, model_input1)
            if req_num0 + req_num1 > 0:
                logits = torch.cat((model_output0.logits, model_output1.logits), dim=0)
                b_req_idx = torch.cat((model_input0.b_req_idx, model_input1.b_req_idx), dim=0)
                b_mtp_index = torch.cat((model_input0.b_mtp_index, model_input1.b_mtp_index), dim=0)
                (
                    _,
                    next_token_ids_cpu,
                    next_token_logprobs_cpu,
                    next_token_ranks_cpu,
                ) = self._sample_and_scatter_token(
                    logits=logits,
                    b_req_idx=b_req_idx,
                    b_mtp_index=b_mtp_index,
                    run_reqs=run_reqs,
                    is_prefill=False,
                    mask_func=None,
                )
                sync_event = torch.cuda.Event()
                sync_event.record()

        if req_num0 + req_num1 > 0:
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
                next_token_ranks=next_token_ranks_cpu,
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
        model_input, run_reqs = prepare_prefill_inputs(
            prefill_reqs,
            is_chuncked_mode=not self.disable_chunked_prefill,
        )
        req_num = len(run_reqs)
        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            model_output: ModelOutput = self.model.forward(model_input)
            b_has_out_cpu = model_input.b_prefill_has_output_cpu
            self._capture_prompt_logprobs_if_needed(model_input, run_reqs, model_output.prompt_logics)
            b_req_idx = model_input.b_req_idx
            b_mtp_index = model_input.b_mtp_index

            if req_num > 0:
                (
                    next_token_ids,
                    next_token_ids_cpu,
                    next_token_logprobs_cpu,
                    next_token_ranks_cpu,
                ) = self._sample_and_scatter_token(
                    logits=model_output.logits,
                    b_req_idx=b_req_idx,
                    b_mtp_index=b_mtp_index,
                    run_reqs=run_reqs,
                    is_prefill=True,
                    b_prefill_has_output_cpu=b_has_out_cpu,
                    mask_func=None,
                )
            else:
                next_token_ids = torch.empty((0,), dtype=torch.int64, device=model_input.b_req_idx.device)

            # BaseModel 已负责空 batch 的内部 padding，这里直接把真实 target
            # 输出交给与非 DP 路径相同的 SpecEngine。
            self.spec_engine.fill_draft_model_kv_state(
                target_model_input=model_input,
                target_model_output=model_output,
                target_next_token_ids=next_token_ids,
            )
            if req_num > 0:
                g_infer_context.copy_linear_att_state_to_cache_buffer(b_req_idx=b_req_idx, reqs=run_reqs)

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
                next_token_ranks=next_token_ranks_cpu,
                run_reqs_update_packs=update_packs,
                extra_post_req_handle_func=self.extra_post_req_handle_func,
                pd_prefill_chunked_handle_func=self.pd_prefill_chunked_handle_func,
            )

            # 第四阶段
            event_pack.notify_pre_post_handle()
        else:
            event_pack.notify_post_handle_and_wait_pre_post_handle()
            event_pack.notify_forward_and_wait_post_handle()
            event_pack.notify_pre_post_handle()
        return

    def decode_mtp(self, event_pack: OverlapEventPack, decode_reqs: List[InferReq]):
        """复用普通 SpecEngine 执行 DP speculative draft-and-verify。"""

        model_input, run_reqs = prepare_decode_inputs(req_objs=decode_reqs)
        spec_engine = self.spec_engine
        req_num = len(decode_reqs)

        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            spec_plan = spec_engine.plan_decode(
                model_input=model_input,
                decode_reqs=decode_reqs,
            )
            model_input, async_selected_row_mask_cpu = spec_engine.prepare_decode_model_input(
                model_input=model_input,
                req_num=req_num,
                plan=spec_plan,
            )
            model_output = self.model.forward(model_input)

            if async_selected_row_mask_cpu is not None:
                async_selected_row_mask_cpu.wait()
                selected_rows = async_selected_row_mask_cpu.tensor.tolist()
                run_reqs = [req for req, selected in zip(run_reqs, selected_rows) if selected]

            if req_num > 0:
                next_token_ids, next_token_logprobs = sample(
                    model_output.logits,
                    run_reqs,
                    self.eos_id,
                )
                next_token_ranks = self._get_next_token_ranks(model_output.logits, next_token_ids)

                b_req_mtp_start_loc = gen_b_req_mtp_start_loc(
                    b_mtp_index=model_input.b_mtp_index,
                    num_reqs=req_num,
                )

                mtp_accept_len, accepted_index = mtp_utils.verify_mtp_tokens(
                    backend=self,
                    next_token_ids=next_token_ids,
                    b_req_idx=model_input.b_req_idx,
                    b_req_mtp_start_loc=b_req_mtp_start_loc,
                    b_mtp_index=model_input.b_mtp_index,
                )
                accepted_index_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor(
                    key="accepted_index",
                    gpu_tensor=accepted_index,
                )
                mtp_accept_len_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor(
                    key="mtp_accept_len",
                    gpu_tensor=mtp_accept_len,
                )
                g_infer_context.req_sampling_manager.update_reqs_out_token_counter_gpu(
                    b_req_idx=model_input.b_req_idx,
                    next_token_ids=next_token_ids,
                    mask=accepted_index == 1,
                )
            else:
                next_token_ids = torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=model_input.b_req_idx.device,
                )
                b_req_mtp_start_loc = torch.empty(
                    (0,),
                    dtype=torch.int32,
                    device=model_input.b_req_idx.device,
                )
                mtp_accept_len = torch.empty_like(b_req_mtp_start_loc)

            verify_event = torch.cuda.Event()
            verify_event.record()

            proposal = spec_engine.propose_next(
                target_model_input=model_input,
                target_model_output=model_output,
                target_next_token_ids=next_token_ids,
                b_req_mtp_start_loc=b_req_mtp_start_loc,
                draft_step=spec_plan.draft_step,
                accept_len=mtp_accept_len,
            )
            if req_num > 0:
                mtp_utils.scatter_mtp_next_tokens(
                    backend=self,
                    proposal=proposal,
                    target_next_token_ids=next_token_ids,
                    b_req_mtp_start_loc=b_req_mtp_start_loc,
                    b_req_idx=model_input.b_req_idx,
                    mtp_accept_len=mtp_accept_len,
                )
                (
                    next_token_ids_cpu,
                    next_token_logprobs_cpu,
                    next_token_ranks_cpu,
                ) = self._async_copy_next_token_infos_to_pin_mem(
                    next_token_ids=next_token_ids,
                    next_token_logprobs=next_token_logprobs,
                    next_token_ranks=next_token_ranks,
                )

            sync_event = torch.cuda.Event()
            sync_event.record()

        if req_num > 0:
            # 第二阶段
            event_pack.notify_post_handle_and_wait_pre_post_handle()
            if spec_plan.skip_verify_sync:
                verify_ok_reqs = run_reqs
            else:
                verify_event.synchronize()
                verify_ok_reqs = [req for req, accepted in zip(run_reqs, accepted_index_cpu.tolist()) if accepted]

            update_packs = self._pre_post_handle(verify_ok_reqs, is_chuncked_mode=False)

            # 第三阶段
            event_pack.notify_forward_and_wait_post_handle()
            sync_event.synchronize()

            spec_engine.update_planner_statics(
                plan=spec_plan,
                proposal=proposal,
                req_num=req_num,
                accept_lengths_cpu=mtp_accept_len_cpu,
            )
            mtp_utils.record_request_mtp_metrics(
                backend=self,
                decode_reqs=decode_reqs,
                accept_lengths_cpu=mtp_accept_len_cpu,
                verify_run_reqs=run_reqs,
            )

            proposal.extra_mem_indexes_cpu.append(
                MtpMemIndexesToFree(
                    mem_indexes_cpu=model_input.mem_indexes_cpu,
                    free_mask_cpu=accepted_index_cpu == 0,
                ),
            )

            select_mask = accepted_index_cpu.to(dtype=torch.bool)
            self._post_handle(
                run_reqs=verify_ok_reqs,
                next_token_ids=next_token_ids_cpu[select_mask],
                next_token_logprobs=next_token_logprobs_cpu[select_mask],
                next_token_ranks=next_token_ranks_cpu[select_mask],
                run_reqs_update_packs=update_packs,
                extra_post_req_handle_func=self.extra_post_req_handle_func,
            )
            mtp_utils.free_mem_indexes(
                backend=self,
                extra_mem_indexes_cpu=proposal.extra_mem_indexes_cpu,
            )

            # 第四阶段
            event_pack.notify_pre_post_handle()
        else:
            event_pack.notify_post_handle_and_wait_pre_post_handle()
            event_pack.notify_forward_and_wait_post_handle()
            sync_event.synchronize()
            mtp_utils.free_mem_indexes(
                backend=self,
                extra_mem_indexes_cpu=proposal.extra_mem_indexes_cpu,
            )
            event_pack.notify_pre_post_handle()
        return

    def prefill_overlap_mtp(self, event_pack: OverlapEventPack, prefill_reqs: List[InferReq]):
        (
            model_input0,
            run_reqs0,
            model_input1,
            run_reqs1,
        ) = overlap_prepare_prefill_inputs(prefill_reqs)
        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            model_output0, model_output1 = self.model.microbatch_overlap_prefill(model_input0, model_input1)
            self._capture_prompt_logprobs_if_needed(model_input0, run_reqs0, model_output0.prompt_logics)
            self._capture_prompt_logprobs_if_needed(model_input1, run_reqs1, model_output1.prompt_logics)
            logits0 = model_output0.logits
            logits1 = model_output1.logits
            req_num0, req_num1 = len(run_reqs0), len(run_reqs1)
            req_num = req_num0 + req_num1
            logits = torch.empty(
                (req_num0 + req_num1, logits0.shape[1]),
                dtype=logits0.dtype,
                device=logits0.device,
            )
            logits[0:req_num0, :].copy_(logits0, non_blocking=True)
            logits[req_num0 : (req_num0 + req_num1), :].copy_(logits1, non_blocking=True)

            run_reqs = run_reqs0 + run_reqs1
            b_has_out_cpu = model_input0.b_prefill_has_output_cpu + model_input1.b_prefill_has_output_cpu
            b_mtp_index = torch.cat((model_input0.b_mtp_index, model_input1.b_mtp_index), dim=0)
            b_req_idx = torch.cat((model_input0.b_req_idx, model_input1.b_req_idx), dim=0)

            if req_num > 0:
                (
                    next_token_ids,
                    next_token_ids_cpu,
                    next_token_logprobs_cpu,
                    next_token_ranks_cpu,
                ) = self._sample_and_scatter_token(
                    logits=logits,
                    run_reqs=run_reqs,
                    b_req_idx=b_req_idx,
                    b_mtp_index=b_mtp_index,
                    is_prefill=True,
                    b_prefill_has_output_cpu=b_has_out_cpu,
                )
            else:
                next_token_ids = torch.empty((0,), dtype=torch.int64, device=logits.device)

            target_next_token_ids_gpu0 = next_token_ids[:req_num0]
            target_next_token_ids_gpu1 = next_token_ids[req_num0:]

            self.prefill_draft_engine.fill_draft_model_kv_state_overlap(
                target_model_input0=model_input0,
                target_model_output0=model_output0,
                target_next_token_ids0=target_next_token_ids_gpu0,
                target_model_input1=model_input1,
                target_model_output1=model_output1,
                target_next_token_ids1=target_next_token_ids_gpu1,
            )

            if req_num > 0 and g_infer_context.is_linear_att_mixed_model:
                g_infer_context.copy_linear_att_state_to_cache_buffer(b_req_idx=b_req_idx, reqs=run_reqs)

            sync_event = torch.cuda.Event()
            sync_event.record()

        if req_num > 0:
            event_pack.notify_post_handle_and_wait_pre_post_handle()
            update_packs = self._pre_post_handle(run_reqs, is_chuncked_mode=not self.disable_chunked_prefill)

            event_pack.notify_forward_and_wait_post_handle()
            sync_event.synchronize()

            self._post_handle(
                run_reqs=run_reqs,
                next_token_ids=next_token_ids_cpu,
                next_token_logprobs=next_token_logprobs_cpu,
                next_token_ranks=next_token_ranks_cpu,
                run_reqs_update_packs=update_packs,
                extra_post_req_handle_func=self.extra_post_req_handle_func,
                pd_prefill_chunked_handle_func=self.pd_prefill_chunked_handle_func,
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
            decode_reqs0,
            model_input1,
            run_reqs1,
            decode_reqs1,
        ) = overlap_prepare_decode_inputs(req_objs=decode_reqs)
        real_request_num0 = len(decode_reqs0)
        real_request_num1 = len(decode_reqs1)
        req_num = real_request_num0 + real_request_num1
        spec_engine = self.decode_draft_engine
        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            spec_plan = spec_engine.plan_decode(
                model_input0=model_input0,
                model_input1=model_input1,
                decode_reqs=decode_reqs,
            )
            (
                model_input0,
                selected_row_mask_cpu0,
                model_input1,
                selected_row_mask_cpu1,
            ) = spec_engine.prepare_decode_model_inputs(
                model_input0=model_input0,
                req_num0=real_request_num0,
                model_input1=model_input1,
                req_num1=real_request_num1,
                plan=spec_plan,
            )
            model_output0, model_output1 = self.model.microbatch_overlap_decode(model_input0, model_input1)

            if selected_row_mask_cpu0 is not None:
                selected_row_mask_cpu0.wait()
                selected_rows0 = selected_row_mask_cpu0.tensor.tolist()
                run_reqs0 = [req for req, selected in zip(run_reqs0, selected_rows0) if selected]
            if selected_row_mask_cpu1 is not None:
                selected_row_mask_cpu1.wait()
                selected_rows1 = selected_row_mask_cpu1.tensor.tolist()
                run_reqs1 = [req for req, selected in zip(run_reqs1, selected_rows1) if selected]

            verify_row_num0 = model_input0.batch_size
            verify_row_num1 = model_input1.batch_size
            verify_row_num = verify_row_num0 + verify_row_num1
            logits0 = model_output0.logits
            logits1 = model_output1.logits
            run_reqs = run_reqs0 + run_reqs1
            if req_num > 0:
                assert len(run_reqs) == verify_row_num
                logits = torch.empty(
                    (verify_row_num, logits0.shape[1]),
                    dtype=logits0.dtype,
                    device=logits0.device,
                )
                logits[:verify_row_num0, :].copy_(logits0, non_blocking=True)
                logits[verify_row_num0:, :].copy_(logits1, non_blocking=True)
                next_token_ids, next_token_logprobs = sample(logits, run_reqs, self.eos_id)
                next_token_ranks = self._get_next_token_ranks(logits, next_token_ids)
                (
                    next_token_ids_cpu,
                    next_token_logprobs_cpu,
                    next_token_ranks_cpu,
                ) = self._async_copy_next_token_infos_to_pin_mem(next_token_ids, next_token_logprobs, next_token_ranks)

                b_req_idx = torch.cat((model_input0.b_req_idx, model_input1.b_req_idx), dim=0)
                b_mtp_index = torch.cat(
                    (model_input0.b_mtp_index, model_input1.b_mtp_index),
                    dim=0,
                )
                b_req_mtp_start_loc = gen_b_req_mtp_start_loc(
                    b_mtp_index=b_mtp_index,
                    num_reqs=req_num,
                )
                mtp_accept_len, accepted_index = mtp_utils.verify_mtp_tokens(
                    backend=self,
                    next_token_ids=next_token_ids,
                    b_req_idx=b_req_idx,
                    b_req_mtp_start_loc=b_req_mtp_start_loc,
                    b_mtp_index=b_mtp_index,
                )
                mtp_accept_len0 = mtp_accept_len[:real_request_num0]
                mtp_accept_len1 = mtp_accept_len[real_request_num0:]
                accepted_index_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor(
                    key="accepted_index",
                    gpu_tensor=accepted_index,
                )
                mtp_accept_len_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor(
                    key="mtp_accept_len",
                    gpu_tensor=mtp_accept_len,
                )
                accepted_index_cpu0 = accepted_index_cpu[:verify_row_num0]
                accepted_index_cpu1 = accepted_index_cpu[verify_row_num0:]
                mtp_accept_len_cpu0 = mtp_accept_len_cpu[:real_request_num0]
                mtp_accept_len_cpu1 = mtp_accept_len_cpu[real_request_num0:]
            else:
                b_req_idx = torch.empty((0,), dtype=torch.int32, device=model_input0.b_req_idx.device)
                mtp_accept_len = torch.empty((0,), dtype=torch.int32, device=model_input0.b_req_idx.device)
                mtp_accept_len0 = mtp_accept_len
                mtp_accept_len1 = mtp_accept_len
                b_req_mtp_start_loc = torch.empty((0,), dtype=torch.int32, device=model_input0.b_req_idx.device)
                next_token_ids = torch.empty((0,), dtype=torch.int64, device=model_input0.b_req_idx.device)
            verify_event = torch.cuda.Event()
            verify_event.record()

            target_next_token_ids0 = next_token_ids[:verify_row_num0]
            target_next_token_ids1 = next_token_ids[verify_row_num0:]
            proposal = self.decode_draft_engine.propose_next_overlap(
                target_model_input0=model_input0,
                target_model_output0=model_output0,
                target_next_token_ids0=target_next_token_ids0,
                accept_len0=mtp_accept_len0,
                target_model_input1=model_input1,
                target_model_output1=model_output1,
                target_next_token_ids1=target_next_token_ids1,
                accept_len1=mtp_accept_len1,
                draft_step=spec_plan.draft_step,
            )
            if req_num > 0:
                mtp_utils.scatter_mtp_next_tokens(
                    backend=self,
                    proposal=proposal,
                    target_next_token_ids=next_token_ids,
                    b_req_mtp_start_loc=b_req_mtp_start_loc,
                    b_req_idx=b_req_idx,
                    mtp_accept_len=mtp_accept_len,
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
            event_pack.notify_post_handle_and_wait_pre_post_handle()
            verify_event.synchronize()
            mtp_utils.record_request_mtp_metrics(
                backend=self,
                decode_reqs=decode_reqs0,
                accept_lengths_cpu=mtp_accept_len_cpu0,
                verify_run_reqs=run_reqs0,
            )
            mtp_utils.record_request_mtp_metrics(
                backend=self,
                decode_reqs=decode_reqs1,
                accept_lengths_cpu=mtp_accept_len_cpu1,
                verify_run_reqs=run_reqs1,
            )
            verify_ok_reqs0 = [req for req, accepted in zip(run_reqs0, accepted_index_cpu0.tolist()) if accepted]
            verify_ok_reqs1 = [req for req, accepted in zip(run_reqs1, accepted_index_cpu1.tolist()) if accepted]
            verify_ok_reqs = verify_ok_reqs0 + verify_ok_reqs1
            update_packs = self._pre_post_handle(verify_ok_reqs, is_chuncked_mode=False)

            event_pack.notify_forward_and_wait_post_handle()
            sync_event.synchronize()
            spec_engine.update_planner_statics(
                plan=spec_plan,
                proposal=proposal,
                req_num=req_num,
                accept_lengths_cpu=mtp_accept_len_cpu,
            )
            proposal.extra_mem_indexes_cpu.extend(
                (
                    MtpMemIndexesToFree(
                        mem_indexes_cpu=model_input0.mem_indexes_cpu,
                        free_mask_cpu=accepted_index_cpu0 == 0,
                    ),
                    MtpMemIndexesToFree(
                        mem_indexes_cpu=model_input1.mem_indexes_cpu,
                        free_mask_cpu=accepted_index_cpu1 == 0,
                    ),
                )
            )

            select_mask = accepted_index_cpu.to(dtype=torch.bool)
            self._post_handle(
                run_reqs=verify_ok_reqs,
                next_token_ids=next_token_ids_cpu[select_mask],
                next_token_logprobs=next_token_logprobs_cpu[select_mask],
                next_token_ranks=next_token_ranks_cpu[select_mask],
                run_reqs_update_packs=update_packs,
                extra_post_req_handle_func=self.extra_post_req_handle_func,
            )
            mtp_utils.free_mem_indexes(
                backend=self,
                extra_mem_indexes_cpu=proposal.extra_mem_indexes_cpu,
            )
            event_pack.notify_pre_post_handle()
        else:
            event_pack.notify_post_handle_and_wait_pre_post_handle()
            event_pack.notify_forward_and_wait_post_handle()
            sync_event.synchronize()
            mtp_utils.free_mem_indexes(
                backend=self,
                extra_mem_indexes_cpu=proposal.extra_mem_indexes_cpu,
            )
            event_pack.notify_pre_post_handle()
        return
