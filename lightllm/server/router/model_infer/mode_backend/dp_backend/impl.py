import torch
import numpy as np
from typing import List, Tuple
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.common.basemodel.batch_objs import ModelOutput
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
from lightllm.server.router.model_infer.mode_backend.pre import padded_prepare_prefill_inputs
from lightllm.server.router.model_infer.mode_backend.pre import padded_overlap_prepare_prefill_inputs
from lightllm.server.router.model_infer.mode_backend.pre import padded_prepare_decode_inputs
from lightllm.server.router.model_infer.mode_backend.pre import padded_overlap_prepare_decode_inputs
from lightllm.server.router.model_infer.mode_backend.overlap_events import OverlapEventPack
from lightllm.common.basemodel.infer_lock import g_infer_state_lock
from lightllm.server.router.model_infer.mode_backend.mtp_pre_process import (
    prepare_mtp_prefill_inputs,
)
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.envs_utils import get_env_start_args
from .control_state import DPControlState


class DPChunkedPrefillBackend(ModeBackend):
    def __init__(self) -> None:
        super().__init__()

        # 用于控制每一步是执行prefill 和 decode 还是跳过
        self.control_state_machine = DPControlState()

        # 在 mtp 模式下切换绑定的prefill 和 decode 函数
        if get_env_start_args().mtp_mode:
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
        return

    def infer_loop(self):
        torch.cuda.set_device(get_current_device_id())
        try:
            while True:
                event_pack = self.overlap_event_manager.get_overlap_event_pack()
                event_pack.wait_to_forward()

                self._try_read_new_reqs()

                prefill_reqs, decode_reqs = self._get_classed_reqs(
                    recover_paused=self.control_state_machine.try_recover_paused_reqs()
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
                    continue

        except BaseException as e:
            self.logger.exception(str(e))
            raise e

    def prefill_normal(
        self,
        event_pack: OverlapEventPack,
        prefill_reqs: List[InferReq],
    ):
        model_input, run_reqs, padded_req_num = padded_prepare_prefill_inputs(
            prefill_reqs, is_multimodal=self.is_multimodal
        )
        model_output: ModelOutput = self.model.forward(model_input)
        logits = model_output.logits
        if len(run_reqs) != 0:
            logits = logits[0 : len(run_reqs), :]
            next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
            next_token_logprobs = torch.log(next_token_probs)
            sync_event = torch.cuda.Event()
            sync_event.record()

            # 第二阶段
            event_pack.notify_post_handle_and_wait_pre_post_handle()
            update_packs = self._pre_post_handle(run_reqs, is_chuncked_mode=not self.disable_chunked_prefill)

            # 第三阶段
            event_pack.notify_forward_and_wait_post_handle()
            sync_event.synchronize()
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = next_token_logprobs.detach().cpu().numpy()
            self._post_handle(
                run_reqs=run_reqs,
                next_token_ids=next_token_ids,
                next_token_logprobs=next_token_logprobs,
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

    def decode_normal(self, event_pack: OverlapEventPack, decode_reqs: List[InferReq]):
        model_input, run_reqs, padded_req_num = padded_prepare_decode_inputs(
            decode_reqs, is_multimodal=self.is_multimodal
        )
        model_output: ModelOutput = self.model.forward(model_input)
        logits = model_output.logits

        if len(run_reqs) != 0:
            logits = logits[0 : len(run_reqs), :]
            next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
            next_token_logprobs = torch.log(next_token_probs)
            sync_event = torch.cuda.Event()
            sync_event.record()

            # 第二阶段
            event_pack.notify_post_handle_and_wait_pre_post_handle()
            update_packs = self._pre_post_handle(run_reqs, is_chuncked_mode=False)

            # 第三阶段
            event_pack.notify_forward_and_wait_post_handle()
            sync_event.synchronize()
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = next_token_logprobs.detach().cpu().numpy()
            self._post_handle(
                run_reqs=run_reqs,
                next_token_ids=next_token_ids,
                next_token_logprobs=next_token_logprobs,
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
            micro_input0,
            run_reqs0,
            padded_req_num0,
            micro_input1,
            run_reqs1,
            padded_req_num1,
        ) = padded_overlap_prepare_prefill_inputs(prefill_reqs, is_multimodal=self.is_multimodal)
        model_output0, model_output1 = self.model.microbatch_overlap_prefill(micro_input0, micro_input1)
        logits0 = model_output0.logits
        logits1 = model_output1.logits

        req_num0, req_num1 = len(run_reqs0), len(run_reqs1)
        all_logits = torch.empty((req_num0 + req_num1, logits0.shape[1]), dtype=logits0.dtype, device=logits0.device)

        all_logits[0:req_num0, :].copy_(logits0[0:req_num0, :], non_blocking=True)
        all_logits[req_num0 : (req_num0 + req_num1), :].copy_(logits1[0:req_num1, :], non_blocking=True)

        run_reqs = run_reqs0 + run_reqs1
        if run_reqs:
            logits = all_logits[0 : len(run_reqs), :]
            next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
            next_token_logprobs = torch.log(next_token_probs)
            sync_event = torch.cuda.Event()
            sync_event.record()

            # 第二阶段
            event_pack.notify_post_handle_and_wait_pre_post_handle()
            update_packs = self._pre_post_handle(run_reqs, is_chuncked_mode=not self.node_broadcast_tensor)

            # 第三阶段
            event_pack.notify_forward_and_wait_post_handle()
            sync_event.synchronize()
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = next_token_logprobs.detach().cpu().numpy()
            self._post_handle(
                run_reqs=run_reqs,
                next_token_ids=next_token_ids,
                next_token_logprobs=next_token_logprobs,
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

    def decode_overlap(self, event_pack: OverlapEventPack, decode_reqs: List[InferReq]):
        (
            micro_input0,
            run_reqs0,
            padded_req_num0,
            micro_input1,
            run_reqs1,
            padded_req_num1,
        ) = padded_overlap_prepare_decode_inputs(decode_reqs, is_multimodal=self.is_multimodal)
        model_output0, model_output1 = self.model.microbatch_overlap_decode(micro_input0, micro_input1)
        logits0 = model_output0.logits
        logits1 = model_output1.logits

        req_num0, req_num1 = len(run_reqs0), len(run_reqs1)
        all_logits = torch.empty((req_num0 + req_num1, logits0.shape[1]), dtype=logits0.dtype, device=logits0.device)

        all_logits[0:req_num0, :].copy_(logits0[0:req_num0, :], non_blocking=True)
        all_logits[req_num0 : (req_num0 + req_num1), :].copy_(logits1[0:req_num1, :], non_blocking=True)

        run_reqs = run_reqs0 + run_reqs1
        if run_reqs:
            logits = all_logits[0 : len(run_reqs), :]
            next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
            next_token_logprobs = torch.log(next_token_probs)
            sync_event = torch.cuda.Event()
            sync_event.record()

            # 第二阶段
            event_pack.notify_post_handle_and_wait_pre_post_handle()
            update_packs = self._pre_post_handle(run_reqs, is_chuncked_mode=False)

            # 第三阶段
            event_pack.notify_forward_and_wait_post_handle()
            sync_event.synchronize()
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = next_token_logprobs.detach().cpu().numpy()
            self._post_handle(
                run_reqs=run_reqs,
                next_token_ids=next_token_ids,
                next_token_logprobs=next_token_logprobs,
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
        model_input, run_reqs, padded_req_num = padded_prepare_prefill_inputs(
            prefill_reqs, is_multimodal=self.is_multimodal
        )
        model_output: ModelOutput = self.model.forward(model_input)

        next_token_ids_cpu = []

        if len(run_reqs) != 0:
            next_token_ids_gpu, next_token_probs = sample(model_output.logits[: len(run_reqs)], run_reqs, self.eos_id)
            next_token_ids_cpu = next_token_ids_gpu.detach().cpu().numpy()
            next_token_logprobs_cpu = torch.log(next_token_probs).detach().cpu().numpy()

            update_packs = self._pre_post_handle(run_reqs, is_chuncked_mode=not self.disable_chunked_prefill)
            self._post_handle(
                run_reqs=run_reqs,
                next_token_ids=next_token_ids_cpu,
                next_token_logprobs=next_token_logprobs_cpu,
                run_reqs_update_packs=update_packs,
                extra_post_req_handle_func=self.extra_post_req_handle_func,
            )

        # fill mtp draft model prefill kv
        # 因为存在padding的请求，需要将padding的请求一并考虑同时进行推理。
        draft_model_input = model_input
        draft_next_token_ids_gpu = torch.zeros((model_input.batch_size), dtype=torch.int64, device="cuda")
        if len(run_reqs) != 0:
            draft_next_token_ids_gpu[0 : len(run_reqs)].copy_(next_token_ids_gpu)

        draft_model_output = model_output

        for draft_model_idx in range(self.mtp_step):
            draft_model_input = prepare_mtp_prefill_inputs(
                model_input=draft_model_input,
                b_next_token_ids=draft_next_token_ids_gpu,
                deepseekv3_mtp_draft_input_hiddens=draft_model_output.deepseekv3_mtp_main_output_hiddens,
            )

            draft_model_output = self.draft_models[draft_model_idx].forward(draft_model_input)
            draft_next_token_ids_gpu, _ = self._gen_argmax_token_ids(draft_model_output)
        return

    def decode_mtp(self, event_pack: OverlapEventPack, decode_reqs: List[InferReq]):
        model_input, run_reqs, padded_req_num = padded_prepare_decode_inputs(
            decode_reqs, is_multimodal=self.is_multimodal
        )
        # main model decode
        model_output = self.model.forward(model_input)

        need_free_mem_indexes = []
        verify_ok_req_last_indexes = []
        if len(run_reqs) != 0:
            next_token_ids_gpu, next_token_probs = sample(model_output.logits[: len(run_reqs)], run_reqs, self.eos_id)
            next_token_ids_cpu = next_token_ids_gpu.detach().cpu().numpy()
            next_token_logprobs_cpu = torch.log(next_token_probs).detach().cpu().numpy()

            # verify
            mem_indexes_cpu = model_input.mem_indexes[0 : len(run_reqs)].cpu().numpy()
            verify_ok_reqs, verify_ok_req_indexes, verify_ok_req_last_indexes, need_free_mem_indexes = self._verify_mtp(
                run_reqs, next_token_ids_cpu, mem_indexes_cpu
            )

            update_packs = self._pre_post_handle(verify_ok_reqs, is_chuncked_mode=False)
            self._post_handle(
                run_reqs=verify_ok_reqs,
                next_token_ids=next_token_ids_cpu[verify_ok_req_indexes],
                next_token_logprobs=next_token_logprobs_cpu[verify_ok_req_indexes],
                run_reqs_update_packs=update_packs,
                extra_post_req_handle_func=self.extra_post_req_handle_func,
            )

        # fill draft model kv and gen next mtp token ids.
        draft_model_input = model_input
        draft_model_output = model_output
        draft_next_token_ids_gpu = torch.zeros((model_input.batch_size), dtype=torch.int64, device="cuda")
        if len(run_reqs) != 0:
            draft_next_token_ids_gpu[0 : len(run_reqs)].copy_(next_token_ids_gpu)

        # process the draft model output
        for draft_model_idx in range(self.mtp_step):

            draft_model_input.input_ids = draft_next_token_ids_gpu
            draft_model_input.deepseekv3_mtp_draft_input_hiddens = draft_model_output.deepseekv3_mtp_main_output_hiddens
            # spec decode: MTP
            draft_model_output: ModelOutput = self.draft_models[draft_model_idx].forward(draft_model_input)
            draft_next_token_ids_gpu, draft_next_token_ids_cpu = self._gen_argmax_token_ids(draft_model_output)

            if verify_ok_req_last_indexes:
                unique_reqs = [run_reqs[index] for index in verify_ok_req_last_indexes]
                self._update_reqs_mtp_gen_token_ids(
                    reqs=unique_reqs, mtp_draft_next_token_ids=draft_next_token_ids_cpu[verify_ok_req_last_indexes]
                )

        if need_free_mem_indexes:
            g_infer_state_lock.acquire()
            g_infer_context.req_manager.mem_manager.free(need_free_mem_indexes)
            g_infer_state_lock.release()
        return

    def prefill_overlap_mtp(self, event_pack: OverlapEventPack, prefill_reqs: List[InferReq]):
        (
            micro_input0,
            run_reqs0,
            padded_req_num0,
            micro_input1,
            run_reqs1,
            padded_req_num1,
        ) = padded_overlap_prepare_prefill_inputs(prefill_reqs, is_multimodal=self.is_multimodal)

        micro_output0, micro_output1 = self.model.microbatch_overlap_prefill(micro_input0, micro_input1)

        req_num0, req_num1 = len(run_reqs0), len(run_reqs1)
        run_reqs = run_reqs0 + run_reqs1
        next_token_ids_cpu = []
        if len(run_reqs) != 0:
            all_logits = torch.empty(
                (len(run_reqs), micro_output0.logits.shape[1]),
                dtype=micro_output0.logits.dtype,
                device=micro_output0.logits.device,
            )

            all_logits[0:req_num0, :].copy_(micro_output0.logits[0:req_num0, :], non_blocking=True)
            all_logits[req_num0 : (req_num0 + req_num1), :].copy_(
                micro_output1.logits[0:req_num1, :], non_blocking=True
            )

            next_token_ids_gpu, next_token_probs = sample(all_logits, run_reqs, self.eos_id)
            next_token_ids_cpu = next_token_ids_gpu.detach().cpu().numpy()
            next_token_logprobs_cpu = torch.log(next_token_probs).detach().cpu().numpy()

            update_packs = self._pre_post_handle(run_reqs, is_chuncked_mode=not self.disable_chunked_prefill)
            self._post_handle(
                run_reqs=run_reqs,
                next_token_ids=next_token_ids_cpu,
                next_token_logprobs=next_token_logprobs_cpu,
                run_reqs_update_packs=update_packs,
                extra_post_req_handle_func=self.extra_post_req_handle_func,
            )

        # spec prefill: MTP
        draft_micro_input0, draft_micro_input1 = micro_input0, micro_input1
        draft_next_token_ids_gpu0 = torch.zeros((micro_input0.batch_size), dtype=torch.int64, device="cuda")
        if req_num0 > 0:
            draft_next_token_ids_gpu0[0:req_num0].copy_(next_token_ids_gpu[0:req_num0])

        draft_next_token_ids_gpu1 = torch.zeros((micro_input1.batch_size), dtype=torch.int64, device="cuda")
        if req_num1 > 0:
            draft_next_token_ids_gpu1[0:req_num1].copy_(next_token_ids_gpu[req_num0 : (req_num0 + req_num1)])

        draft_micro_output0, draft_micro_output1 = micro_output0, micro_output1

        for draft_model_idx in range(self.mtp_step):

            draft_micro_input0 = prepare_mtp_prefill_inputs(
                model_input=draft_micro_input0,
                b_next_token_ids=draft_next_token_ids_gpu0,
                deepseekv3_mtp_draft_input_hiddens=draft_micro_output0.deepseekv3_mtp_main_output_hiddens,
            )

            draft_micro_input1 = prepare_mtp_prefill_inputs(
                model_input=draft_micro_input1,
                b_next_token_ids=draft_next_token_ids_gpu1,
                deepseekv3_mtp_draft_input_hiddens=draft_micro_output1.deepseekv3_mtp_main_output_hiddens,
            )

            draft_micro_output0, draft_micro_output1 = self.draft_models[draft_model_idx].microbatch_overlap_prefill(
                draft_micro_input0, draft_micro_input1
            )
            draft_next_token_ids_gpu0, _ = self._gen_argmax_token_ids(draft_micro_output0)
            draft_next_token_ids_gpu1, _ = self._gen_argmax_token_ids(draft_micro_output1)
        return

    def decode_overlap_mtp(self, event_pack: OverlapEventPack, decode_reqs: List[InferReq]):
        (
            micro_input0,
            run_reqs0,
            padded_req_num0,
            micro_input1,
            run_reqs1,
            padded_req_num1,
        ) = padded_overlap_prepare_decode_inputs(decode_reqs, is_multimodal=self.is_multimodal)

        micro_output0, micro_output1 = self.model.microbatch_overlap_decode(micro_input0, micro_input1)

        req_num0, req_num1 = len(run_reqs0), len(run_reqs1)
        run_reqs = run_reqs0 + run_reqs1
        need_free_mem_indexes = []
        verify_ok_req_last_indexes = []
        if len(run_reqs) != 0:
            all_logits = torch.empty(
                (req_num0 + req_num1, micro_output0.logits.shape[1]),
                dtype=micro_output0.logits.dtype,
                device=micro_output0.logits.device,
            )

            all_logits[0:req_num0, :].copy_(micro_output0.logits[0:req_num0, :], non_blocking=True)
            all_logits[req_num0 : (req_num0 + req_num1), :].copy_(
                micro_output1.logits[0:req_num1, :], non_blocking=True
            )

            next_token_ids_gpu, next_token_probs = sample(all_logits, run_reqs, self.eos_id)
            next_token_ids_cpu = next_token_ids_gpu.detach().cpu().numpy()
            next_token_logprobs_cpu = torch.log(next_token_probs).detach().cpu().numpy()
            micro_mem_indexes_cpu0 = micro_input0.mem_indexes[0:req_num0].cpu()
            micro_mem_indexes_cpu1 = micro_input1.mem_indexes[0:req_num1].cpu()
            mem_indexes_cpu = torch.cat((micro_mem_indexes_cpu0, micro_mem_indexes_cpu1), dim=0).numpy()

            # verify
            verify_ok_reqs, verify_ok_req_indexes, verify_ok_req_last_indexes, need_free_mem_indexes = self._verify_mtp(
                run_reqs, next_token_ids_cpu, mem_indexes_cpu
            )

            update_packs = self._pre_post_handle(verify_ok_reqs, is_chuncked_mode=False)
            self._post_handle(
                run_reqs=verify_ok_reqs,
                next_token_ids=next_token_ids_cpu[verify_ok_req_indexes],
                next_token_logprobs=next_token_logprobs_cpu[verify_ok_req_indexes],
                run_reqs_update_packs=update_packs,
                extra_post_req_handle_func=self.extra_post_req_handle_func,
            )

        # share some inference info with the main model
        draft_micro_input0, draft_micro_input1 = micro_input0, micro_input1

        draft_next_token_ids_gpu0 = torch.zeros((micro_input0.batch_size), dtype=torch.int64, device="cuda")
        draft_next_token_ids_gpu1 = torch.zeros((micro_input1.batch_size), dtype=torch.int64, device="cuda")
        if req_num0 > 0:
            draft_next_token_ids_gpu0[0:req_num0].copy_(next_token_ids_gpu[0:req_num0])
        if req_num1 > 1:
            draft_next_token_ids_gpu1[0:req_num1].copy_(next_token_ids_gpu[req_num0 : (req_num0 + req_num1)])
        draft_micro_output0, draft_micro_output1 = micro_output0, micro_output1

        # process the draft model output
        for draft_model_idx in range(self.mtp_step):

            draft_micro_input0.input_ids = draft_next_token_ids_gpu0
            draft_micro_input0.deepseekv3_mtp_draft_input_hiddens = (
                draft_micro_output0.deepseekv3_mtp_main_output_hiddens
            )
            draft_micro_input1.input_ids = draft_next_token_ids_gpu1
            draft_micro_input1.deepseekv3_mtp_draft_input_hiddens = (
                draft_micro_output1.deepseekv3_mtp_main_output_hiddens
            )

            draft_micro_output0, draft_micro_output1 = self.draft_models[draft_model_idx].microbatch_overlap_decode(
                draft_micro_input0, draft_micro_input1
            )

            draft_next_token_ids_gpu0, draft_next_token_ids_cpu0 = self._gen_argmax_token_ids(draft_micro_output0)
            draft_next_token_ids_gpu1, draft_next_token_ids_cpu1 = self._gen_argmax_token_ids(draft_micro_output1)

            if verify_ok_req_last_indexes:
                all_draft_next_token_ids_cpu = np.concatenate(
                    [draft_next_token_ids_cpu0[0:req_num0], draft_next_token_ids_cpu1[0:req_num1]], axis=0
                )
                unique_reqs = [run_reqs[index] for index in verify_ok_req_last_indexes]
                self._update_reqs_mtp_gen_token_ids(
                    reqs=unique_reqs, mtp_draft_next_token_ids=all_draft_next_token_ids_cpu[verify_ok_req_last_indexes]
                )

        if need_free_mem_indexes:
            g_infer_state_lock.acquire()
            g_infer_context.req_manager.mem_manager.free(need_free_mem_indexes)
            g_infer_state_lock.release()
        return
