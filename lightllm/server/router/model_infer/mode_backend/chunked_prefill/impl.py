import torch
import time
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
from lightllm.server.router.model_infer.mtp_speculative.engine import SpecEngine
from lightllm.server.router.model_infer.mtp_speculative import utils as mtp_utils
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import MtpMemIndexesToFree
from lightllm.utils.log_utils import init_logger
from lightllm.utils.dist_utils import get_current_device_id
from .control_state import ControlState
from lightllm.utils.envs_utils import get_env_start_args

logger = init_logger(__name__)


class ChunkedPrefillBackend(ModeBackend):
    def __init__(self) -> None:
        super().__init__()

        # 用于控制每一步是执行prefill 和 decode 还是跳过
        self.control_state_machine = ControlState()

        # 在 mtp 模式下切换绑定的prefill 和 decode 函数
        if get_env_start_args().mtp_mode is not None:
            self.prefill = self.prefill_mtp
            self.decode = self.decode_mtp
        else:
            self.prefill = self.prefill_normal
            self.decode = self.decode_normal

        self.classed_req_strict_prefill = False
        return

    def init_spec_engine(self):
        self.spec_engine = SpecEngine(
            backend=self,
            spec_mode=self.args.mtp_mode,
            enable_dynmaic_mtp=self.args.mtp_dynamic_verify,
        )
        return

    @torch.inference_mode(mode=True)
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
        if self.prefill_mask_func is not None:
            model_input.use_vocab_parallel_topk = False
        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            model_output = self.model.forward(model_input)
            self._capture_prompt_logprobs_if_needed(model_input, run_reqs, model_output.prompt_logics)
            (_, next_token_ids_cpu, next_token_logprobs_cpu, next_token_ranks_cpu,) = self._sample_and_scatter_token(
                model_output=model_output,
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
        if self.decode_mask_func is not None:
            model_input.use_vocab_parallel_topk = False
        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            model_output = self.model.forward(model_input)
            (_, next_token_ids_cpu, next_token_logprobs_cpu, next_token_ranks_cpu,) = self._sample_and_scatter_token(
                model_output=model_output,
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
        if self.prefill_mask_func is not None:
            model_input.use_vocab_parallel_topk = False
        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            model_output = self.model.forward(model_input)
            self._capture_prompt_logprobs_if_needed(model_input, run_reqs, model_output.prompt_logics)
            (
                next_token_ids,
                next_token_ids_cpu,
                next_token_logprobs_cpu,
                next_token_ranks_cpu,
            ) = self._sample_and_scatter_token(
                model_output=model_output,
                b_req_idx=model_input.b_req_idx,
                b_mtp_index=model_input.b_mtp_index,
                run_reqs=run_reqs,
                is_prefill=True,
                b_prefill_has_output_cpu=model_input.b_prefill_has_output_cpu,
                mask_func=self.prefill_mask_func,
            )
            # mtp kv fill
            spec_engine = self.spec_engine
            spec_engine.fill_draft_model_kv_state(
                target_model_input=model_input,
                target_model_output=model_output,
                target_next_token_ids=next_token_ids,
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
            # 动态 MTP verify 可能只从原始物理 batch 中选择部分行参与 target forward。
            # 等待异步回传的行选择掩码后，按相同掩码过滤 run_reqs，使请求列表的
            # 长度和顺序与压缩后的 model_output.logits 保持一一对应，供后续采样使用。
            if async_selected_row_mask_cpu is not None:
                async_selected_row_mask_cpu.wait()
                selected_rows = async_selected_row_mask_cpu.tensor.tolist()
                run_reqs = [req for req, selected in zip(run_reqs, selected_rows) if selected]
            next_token_ids, next_token_logprobs = sample(
                model_output,
                run_reqs,
                self.eos_id,
            )
            next_token_ranks = self._get_next_token_ranks(model_output, next_token_ids)

            b_req_mtp_start_loc = gen_b_req_mtp_start_loc(model_input.b_mtp_index, num_reqs=req_num)
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

            verify_event = torch.cuda.Event()
            verify_event.record()

            g_infer_context.req_sampling_manager.update_reqs_out_token_counter_gpu(
                b_req_idx=model_input.b_req_idx,
                next_token_ids=next_token_ids,
                mask=accepted_index == 1,
            )

            proposal = spec_engine.propose_next(
                target_model_input=model_input,
                target_model_output=model_output,
                target_next_token_ids=next_token_ids,
                b_req_mtp_start_loc=b_req_mtp_start_loc,
                draft_step=spec_plan.draft_step,
                accept_len=mtp_accept_len,
            )
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

        # 第二阶段
        event_pack.notify_post_handle_and_wait_pre_post_handle()

        # 当 pre_draft_step == 0 时，上一轮没有生成 draft token，本轮每个请求
        # 只有一个由 target model 产生且必然提交的 token，不存在需要根据
        # accepted_index_cpu 剔除的 draft 行。因此这里可以直接使用 run_reqs，
        # 无需等待 verify_event，避免一次不必要的 GPU/CPU 同步。
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

        select_mask = accepted_index_cpu.to(dtype=torch.bool)
        self._post_handle(
            run_reqs=verify_ok_reqs,
            next_token_ids=next_token_ids_cpu[select_mask],
            next_token_logprobs=next_token_logprobs_cpu[select_mask],
            next_token_ranks=next_token_ranks_cpu[select_mask],
            run_reqs_update_packs=update_packs,
            extra_post_req_handle_func=self.extra_post_req_handle_func,
        )

        proposal.extra_mem_indexes_cpu.append(
            MtpMemIndexesToFree(
                mem_indexes_cpu=model_input.mem_indexes_cpu,
                free_mask_cpu=accepted_index_cpu == 0,
            ),
        )
        mtp_utils.free_mem_indexes(
            backend=self,
            extra_mem_indexes_cpu=proposal.extra_mem_indexes_cpu,
        )

        # 第四阶段
        event_pack.notify_pre_post_handle()
        return
