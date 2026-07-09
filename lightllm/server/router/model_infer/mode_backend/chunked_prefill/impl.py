import torch
import time
import torch.distributed as dist
from typing import List
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.server.router.model_infer.mode_backend.overlap_events import OverlapEventPack
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.server.router.model_infer.mode_backend.pre import (
    prepare_prefill_inputs,
    prepare_decode_inputs,
)
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.utils.log_utils import init_logger
from lightllm.utils.dist_utils import get_current_device_id
from .control_state import ControlState
from lightllm.utils.dist_utils import create_new_group_for_current_dp

logger = init_logger(__name__)


class ChunkedPrefillBackend(ModeBackend):
    def __init__(self) -> None:
        super().__init__()

        # 用于控制每一步是执行prefill 和 decode 还是跳过
        self.control_state_machine = ControlState()
        self.enable_dynamic_mtp = False

        # 在 mtp 模式下切换绑定的prefill 和 decode 函数
        if self.spec_config.enabled:
            self.prefill = self.prefill_mtp
            self.decode = self.decode_mtp
            self.enable_dynamic_mtp = self.spec_config.dynamic_verify
        else:
            self.prefill = self.prefill_normal
            self.decode = self.decode_normal

        self.classed_req_strict_prefill = False
        return

    def init_custom(self):
        super().init_custom()
        if self.enable_dynamic_mtp:
            self.mtp_gloo_group = create_new_group_for_current_dp("gloo")
            logger.info(f"mtp_gloo_group ranks {dist.get_rank(self.mtp_gloo_group)}")
        return

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
            _, next_token_ids_cpu, next_token_logprobs_cpu = self._sample_and_scatter_token(
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
            _, next_token_ids_cpu, next_token_logprobs_cpu = self._sample_and_scatter_token(
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
            next_token_ids, next_token_ids_cpu, next_token_logprobs_cpu = self._sample_and_scatter_token(
                logits=model_output.logits,
                b_req_idx=model_input.b_req_idx,
                b_mtp_index=model_input.b_mtp_index,
                run_reqs=run_reqs,
                is_prefill=True,
                b_prefill_has_output_cpu=model_input.b_prefill_has_output_cpu,
                mask_func=self.prefill_mask_func,
            )
            # mtp kv fill
            spec_runtime = self.spec_adapter
            spec_runtime.build_initial_draft_state(
                model_input=model_input,
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
        """
        MTP解码的通用流程，整合eagle和vanilla的共同逻辑
        """
        model_input, run_reqs = prepare_decode_inputs(decode_reqs)
        spec_runtime = self.spec_adapter

        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            spec_plan = spec_runtime.plan_decode(model_input=model_input, req_num=len(decode_reqs))
            model_input, selected_run_reqs = spec_runtime.prepare_decode_model_input(
                model_input=model_input,
                req_num=len(decode_reqs),
                plan=spec_plan,
            )
            selected_run_reqs_cpu = spec_runtime.async_copy_selected_run_reqs(selected_run_reqs)

            model_output = self.model.forward(model_input)
            next_token_ids, next_token_logprobs = sample(
                model_output.logits,
                run_reqs,
                self.eos_id,
                dynamic_batch_size=spec_plan.dynamic_batch_size,
                selected_run_reqs=selected_run_reqs,
            )

            spec_decode_state = spec_runtime.run_decode_speculative_forward(
                model_input=model_input,
                model_output=model_output,
                run_reqs=run_reqs,
                req_num=len(decode_reqs),
                plan=spec_plan,
                selected_run_reqs_cpu=selected_run_reqs_cpu,
                next_token_ids=next_token_ids,
                next_token_logprobs=next_token_logprobs,
                copy_next_token_infos=self._async_copy_next_token_infos_to_pin_mem,
            )

        # 第二阶段
        event_pack.notify_post_handle_and_wait_pre_post_handle()

        run_reqs, verify_ok_reqs = spec_runtime.resolve_decode_pre_post_reqs(
            state=spec_decode_state,
            decode_reqs=decode_reqs,
        )
        self._update_mtp_verify_token_num(
            decode_reqs=decode_reqs,
            dynamic_mtp_run_reqs=run_reqs if self.enable_dynamic_mtp else None,
        )
        update_packs = self._pre_post_handle(verify_ok_reqs, is_chuncked_mode=False)

        # 第三阶段
        event_pack.notify_forward_and_wait_post_handle()
        spec_post_state = spec_runtime.finish_decode_post(
            state=spec_decode_state,
            req_num=len(decode_reqs),
            run_reqs=run_reqs,
        )

        self._update_mtp_accept_ratio(
            decode_reqs=decode_reqs,
            mtp_accept_len_cpu=spec_post_state.mtp_accept_len_cpu,
        )
        self._post_handle(
            run_reqs=verify_ok_reqs,
            next_token_ids=spec_post_state.next_token_ids,
            next_token_logprobs=spec_post_state.next_token_logprobs,
            run_reqs_update_packs=update_packs,
            extra_post_req_handle_func=self.extra_post_req_handle_func,
        )

        if len(spec_post_state.need_free_mem_indexes) > 0:
            g_infer_context.req_manager.mem_manager.free(spec_post_state.need_free_mem_indexes)

        # 第四阶段
        event_pack.notify_pre_post_handle()
        return
