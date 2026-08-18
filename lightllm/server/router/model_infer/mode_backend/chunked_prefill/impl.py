import torch
import time
import torch.distributed as dist
from typing import List
from lightllm.common.basemodel.triton_kernel.mtp_utils import gen_b_req_mtp_start_loc
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.server.router.model_infer.mode_backend.overlap_events import OverlapEventPack
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.server.router.model_infer.mode_backend.pre import (
    prepare_prefill_inputs,
    prepare_decode_inputs,
)
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager
from lightllm.utils.log_utils import init_logger
from lightllm.utils.dist_utils import get_current_device_id
from .control_state import ControlState
from lightllm.utils.dist_utils import create_new_group_for_current_dp
from lightllm.utils.envs_utils import get_env_start_args

logger = init_logger(__name__)


class ChunkedPrefillBackend(ModeBackend):
    def __init__(self) -> None:
        super().__init__()

        # 用于控制每一步是执行prefill 和 decode 还是跳过
        self.control_state_machine = ControlState()
        self.enable_dynmaic_mtp = False

        # 在 mtp 模式下切换绑定的prefill 和 decode 函数
        if get_env_start_args().mtp_mode is not None:
            self.prefill = self.prefill_mtp
            self.decode = self.decode_mtp
            self.enable_dynmaic_mtp = get_env_start_args().mtp_dynamic_verify
        else:
            self.prefill = self.prefill_normal
            self.decode = self.decode_normal

        self.classed_req_strict_prefill = False
        return

    def init_custom(self):
        super().init_custom()
        if self.enable_dynmaic_mtp:
            self.spec_gloo_group = create_new_group_for_current_dp("gloo")
            logger.info(f"spec_gloo_group ranks {dist.get_rank(self.spec_gloo_group)}")

    def infer_loop(self):
        torch.cuda.set_device(get_current_device_id())
        try:
            while True:
                event_pack = self.overlap_event_manager.get_overlap_event_pack()
                # 关闭overlap 模式
                if not self.support_overlap:
                    event_pack._close_overlap()

                event_pack.wait_to_forward()

                self._try_read_new_reqs()

                prefill_reqs, decode_reqs = self._get_classed_reqs(
                    no_decode=self.classed_req_no_decode,
                    strict_prefill=self.classed_req_strict_prefill,
                    recover_paused=self.control_state_machine.try_recover_paused_reqs(),
                )

                run_way = self.control_state_machine.select_run_way(prefill_reqs=prefill_reqs, decode_reqs=decode_reqs)

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
        # 第一阶段: 模型推理
        model_input, run_reqs = prepare_prefill_inputs(prefill_reqs, is_chuncked_mode=not self.disable_chunked_prefill)
        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            model_output = self.model.forward(model_input)
            self._capture_prompt_logprobs_if_needed(model_input, run_reqs, model_output.prompt_logics)
            (_, next_token_ids_cpu, next_token_logprobs_cpu, next_token_ranks_cpu,) = self._sample_and_scatter_token(
                logits=model_output.logits,
                b_req_idx=model_input.b_req_idx,
                b_mtp_index=model_input.b_mtp_index,
                run_reqs=run_reqs,
                is_prefill=True,
                b_prefill_has_output_cpu=model_input.b_prefill_has_output_cpu,
                mask_func=self.prefill_mask_func,
            )
            g_infer_context.copy_linear_att_state_to_cache_buffer(
                b_req_idx=model_input.b_req_idx,
                reqs=run_reqs,
            )
            sync_event = torch.cuda.Event()
            sync_event.record()

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
        return

    def decode_normal(
        self,
        event_pack: OverlapEventPack,
        decode_reqs: List[InferReq],
    ):
        model_input, run_reqs = prepare_decode_inputs(decode_reqs)
        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            model_output = self.model.forward(model_input)
            (_, next_token_ids_cpu, next_token_logprobs_cpu, next_token_ranks_cpu,) = self._sample_and_scatter_token(
                logits=model_output.logits,
                b_req_idx=model_input.b_req_idx,
                b_mtp_index=model_input.b_mtp_index,
                run_reqs=run_reqs,
                is_prefill=False,
                mask_func=self.decode_mask_func,
            )
            sync_event = torch.cuda.Event()
            sync_event.record()

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
        return

    def prefill_mtp(
        self,
        event_pack: OverlapEventPack,
        prefill_reqs: List[InferReq],
    ):
        model_input, run_reqs = prepare_prefill_inputs(prefill_reqs, is_chuncked_mode=not self.disable_chunked_prefill)
        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            model_output = self.model.forward(model_input)
            self._capture_prompt_logprobs_if_needed(model_input, run_reqs, model_output.prompt_logics)
            (
                next_token_ids,
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
                mask_func=self.prefill_mask_func,
            )
            # mtp kv fill
            spec_engine = self.spec_engine
            spec_engine.build_draft_state_from_prefill(
                target_model_input=model_input,
                target_model_output=model_output,
                next_token_ids=next_token_ids,
            )
            g_infer_context.copy_linear_att_state_to_cache_buffer(
                b_req_idx=model_input.b_req_idx,
                reqs=run_reqs,
            )
            sync_event = torch.cuda.Event()
            sync_event.record()

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
        return

    def decode_mtp(
        self,
        event_pack: OverlapEventPack,
        decode_reqs: List[InferReq],
    ):
        """Run the speculative draft-and-verify decode flow."""
        model_input, run_reqs = prepare_decode_inputs(decode_reqs)
        spec_engine = self.spec_engine
        req_num = len(decode_reqs)

        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            spec_plan = spec_engine.plan_decode(model_input=model_input, decode_reqs=decode_reqs)

            model_input, async_selected_row_mask_cpu = spec_engine.prepare_decode_model_input(
                model_input=model_input,
                req_num=req_num,
                plan=spec_plan,
            )

            model_output = self.model.forward(model_input)
            if async_selected_row_mask_cpu is not None:
                async_selected_row_mask_cpu.wait()
                run_reqs = spec_plan.filter_reqs(
                    reqs=run_reqs,
                    selected_row_mask_cpu=async_selected_row_mask_cpu.tensor,
                )
            next_token_ids, next_token_logprobs = sample(
                model_output.logits,
                run_reqs,
                self.eos_id,
            )
            next_token_ranks = self._get_next_token_ranks(model_output.logits, next_token_ids)

            b_req_mtp_start_loc = gen_b_req_mtp_start_loc(model_input.b_mtp_index, num_reqs=req_num)
            mtp_accept_len, accepted_index = spec_engine.verify_tokens(
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

            verify_event = torch.cuda.Event()
            verify_event.record()

            proposal = spec_engine.propose_next(
                main_model_input=model_input,
                main_model_output=model_output,
                next_token_ids=next_token_ids,
                b_req_mtp_start_loc=b_req_mtp_start_loc,
                draft_step=spec_plan.draft_step,
                accept_len=mtp_accept_len,
            )
            spec_engine.scatter_next_tokens(
                b_req_mtp_start_loc=b_req_mtp_start_loc,
                all_next_token_ids=proposal.token_ids,
                b_req_idx=model_input.b_req_idx,
                mtp_accept_len=mtp_accept_len,
                schedule_scores=proposal.schedule_scores,
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

            g_infer_context.req_sampling_manager.update_reqs_out_token_counter_gpu(
                b_req_idx=model_input.b_req_idx,
                next_token_ids=next_token_ids,
                mask=accepted_index == 1,
            )

            sync_event = torch.cuda.Event()
            sync_event.record()

        # 第二阶段
        event_pack.notify_post_handle_and_wait_pre_post_handle()

        run_reqs, verify_ok_reqs = spec_engine.resolve_decode_reqs(
            plan=spec_plan,
            verify_event=verify_event,
            run_reqs=run_reqs,
            decode_reqs=decode_reqs,
            accepted_index_cpu=accepted_index_cpu,
        )

        update_packs = self._pre_post_handle(verify_ok_reqs, is_chuncked_mode=False)

        # 第三阶段
        event_pack.notify_forward_and_wait_post_handle()
        sync_event.synchronize()

        spec_engine.update_planner_feedback(
            plan=spec_plan,
            proposal=proposal,
            req_num=req_num,
            accept_lengths_cpu=mtp_accept_len_cpu,
        )

        spec_engine.record_request_spec_metrics(
            decode_reqs=decode_reqs,
            accept_lengths_cpu=mtp_accept_len_cpu,
            verified_row_reqs=run_reqs if self.enable_dynmaic_mtp else None,
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

        spec_engine.free_unused_decode_mem(
            model_input=model_input,
            selected_row_mask_cpu=(
                async_selected_row_mask_cpu.tensor if async_selected_row_mask_cpu is not None else None
            ),
            accepted_index_cpu=accepted_index_cpu,
            extra_mem_indexes_cpu=proposal.extra_mem_indexes_cpu,
        )

        # 第四阶段
        event_pack.notify_pre_post_handle()
        return
