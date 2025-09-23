import torch
import time
from typing import List, Optional, Callable, Dict, Any
from queue import Queue
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.server.router.model_infer.mode_backend.overlap_events import OverlapEventPack
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.server.router.model_infer.mode_backend.pre import (
    prepare_prefill_inputs,
    prepare_decode_inputs,
)
from lightllm.server.router.model_infer.mode_backend.mtp_pre_process import (
    prepare_mtp_prefill_inputs,
    prepare_eagle_decode_inputs,
)
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager
from lightllm.common.basemodel.infer_lock import g_infer_state_lock
from lightllm.common.basemodel.batch_objs import ModelOutput, ModelInput
from lightllm.common.basemodel.triton_kernel.gather_token_id import scatter_token
from lightllm.common.basemodel.triton_kernel.mtp_utils import (
    mtp_scatter_next_token_ids,
)
from lightllm.utils.log_utils import init_logger
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.envs_utils import get_env_start_args
from .control_state import ControlState

logger = init_logger(__name__)


class ChunkedPrefillBackend(ModeBackend):
    def __init__(self) -> None:
        super().__init__()

        # 用于控制每一步是执行prefill 和 decode 还是跳过
        self.control_state_machine = ControlState()

        # 在 mtp 模式下切换绑定的prefill 和 decode 函数
        if get_env_start_args().mtp_mode:
            self.prefill = self.prefill_mtp
            self.decode = self.decode_mtp
            self.is_mtp_eagle = get_env_start_args().mtp_mode == "deepseekv3_eagle"
            self.prefill_mtp_step = 1 if self.is_mtp_eagle else get_env_start_args().mtp_step
        else:
            self.prefill = self.prefill_normal
            self.decode = self.decode_normal
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
        # 第一阶段: 模型推理
        model_input, run_reqs = prepare_prefill_inputs(
            prefill_reqs, is_chuncked_mode=not self.disable_chunked_prefill, is_multimodal=self.is_multimodal
        )
        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            _, next_token_ids_cpu, next_token_logprobs_cpu, _ = self._main_model_forward(
                model_input, run_reqs, self.prefill_mask_func
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
            _, next_token_ids_cpu, next_token_logprobs_cpu, _ = self._main_model_forward(
                model_input, run_reqs, self.decode_mask_func
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
        model_input, run_reqs = prepare_prefill_inputs(
            prefill_reqs, is_chuncked_mode=not self.disable_chunked_prefill, is_multimodal=self.is_multimodal
        )
        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            next_token_ids, next_token_ids_cpu, next_token_logprobs_cpu, model_output = self._main_model_forward(
                model_input, run_reqs, self.prefill_mask_func
            )
            # mtp kv fill
            self._draft_prefill_forward(model_input, model_output, self.prefill_mtp_step, next_token_ids)
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
        )

        # 第四阶段
        event_pack.notify_pre_post_handle()
        return

    def decode_mtp(
        self,
        event_pack: OverlapEventPack,
        decode_reqs: List[InferReq],
    ):
        if self.is_mtp_eagle:
            draft_model_input, _, eagle_mem_indexes_cpu = prepare_eagle_decode_inputs(decode_reqs, self.mtp_step)
            self._decode_mtp_common(
                event_pack=event_pack,
                decode_reqs=decode_reqs,
                draft_decode_func=self._draft_decode_eagle,
                additional_mem_indexes_cpu=eagle_mem_indexes_cpu,
                draft_model_input=draft_model_input,
                eagle_mem_indexes_cpu=eagle_mem_indexes_cpu,
            )
        else:
            self._decode_mtp_common(
                event_pack=event_pack,
                decode_reqs=decode_reqs,
                draft_decode_func=self._draft_decode_vanilla,
            )
        return

    def _main_model_forward(
        self, model_input: ModelInput, run_reqs: List[InferReq], mask_func: Optional[Callable] = None
    ):
        model_output = self.model.forward(model_input)
        logits = model_output.logits

        if mask_func is not None:
            mask_func(run_reqs, logits)

        next_token_ids, next_token_logprobs = sample(logits, run_reqs, self.eos_id)
        b_has_out = None
        if model_input.is_prefill:
            b_has_out = g_pin_mem_manager.gen_from_list(
                key="b_has_out", data=model_input.b_prefill_has_output_cpu, dtype=torch.bool
            ).cuda(non_blocking=True)

        scatter_token(
            next_token_ids=next_token_ids,
            req_to_next_token_ids=self.model.req_manager.req_sampling_params_manager.req_to_next_token_ids,
            b_req_idx=model_input.b_req_idx,
            b_mtp_index=model_input.b_mtp_index,
            b_has_out=b_has_out,
        )
        g_infer_context.req_sampling_manager.update_reqs_out_token_counter_gpu(
            b_req_idx=model_input.b_req_idx,
            next_token_ids=next_token_ids,
            mask=b_has_out,
        )
        next_token_ids_cpu, next_token_logprobs_cpu = self._async_copy_next_token_infos_to_pin_mem(
            next_token_ids, next_token_logprobs
        )
        return next_token_ids, next_token_ids_cpu, next_token_logprobs_cpu, model_output

    def _draft_prefill_forward(
        self, model_input: ModelInput, model_output: ModelOutput, mtp_step: int, next_token_ids: torch.Tensor
    ):
        # spec prefill: MTP, 这个地方只是为了填充draft model的 kv， 并不会使用生成的token_id。
        draft_model_input = model_input
        draft_model_output = model_output
        draft_next_token_ids_gpu = next_token_ids
        for draft_model_idx in range(mtp_step):
            draft_model_input = prepare_mtp_prefill_inputs(
                model_input=draft_model_input,
                b_next_token_ids=draft_next_token_ids_gpu,
                deepseekv3_mtp_draft_input_hiddens=draft_model_output.deepseekv3_mtp_main_output_hiddens,
            )
            draft_model_output = self.draft_models[draft_model_idx].forward(draft_model_input)
            draft_next_token_ids_gpu = self._gen_argmax_token_ids(draft_model_output)
        return

    def _decode_mtp_common(
        self,
        event_pack: OverlapEventPack,
        decode_reqs: List[InferReq],
        draft_decode_func: Callable,
        additional_mem_indexes_cpu: Optional[torch.Tensor] = None,
        **draft_kwargs
    ):
        """
        MTP解码的通用流程，整合eagle和vanilla的共同逻辑
        """
        model_input, run_reqs = prepare_decode_inputs(decode_reqs)

        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            # draft verify with main model
            verify_info = self._draft_verify(model_input, run_reqs)

            # 调用具体的draft decode函数
            draft_decode_func(main_model_input=model_input, verify_info=verify_info, **draft_kwargs)

            g_infer_context.req_sampling_manager.update_reqs_out_token_counter_gpu(
                b_req_idx=model_input.b_req_idx,
                next_token_ids=verify_info["next_token_ids"],
                mask=verify_info["accepted_index"] == 1,
            )
            sync_event = torch.cuda.Event()
            sync_event.record()

        # 第二阶段
        event_pack.notify_post_handle_and_wait_pre_post_handle()
        verify_info["verify_event"].synchronize()
        verify_ok_reqs = [run_reqs[i] for i in range(len(run_reqs)) if verify_info["accepted_index_cpu"][i] == 1]
        update_packs = self._pre_post_handle(verify_ok_reqs, is_chuncked_mode=False)

        # 第三阶段
        event_pack.notify_forward_and_wait_post_handle()
        sync_event.synchronize()

        # 处理需要释放的内存索引
        need_free_mem_indexes = model_input.mem_indexes_cpu[verify_info["accepted_index_cpu"] == 0]
        if additional_mem_indexes_cpu is not None:
            need_free_mem_indexes = torch.cat([need_free_mem_indexes, additional_mem_indexes_cpu], dim=0)

        self._update_mtp_accept_ratio(decode_reqs=decode_reqs, mtp_accept_len_cpu=verify_info["mtp_accept_len_cpu"])
        select_mask = torch.tensor(verify_info["accepted_index_cpu"], dtype=torch.bool, device="cpu")
        self._post_handle(
            run_reqs=verify_ok_reqs,
            next_token_ids=verify_info["next_token_ids_cpu"][select_mask],
            next_token_logprobs=verify_info["next_token_logprobs_cpu"][select_mask],
            run_reqs_update_packs=update_packs,
            extra_post_req_handle_func=self.extra_post_req_handle_func,
        )

        if len(need_free_mem_indexes) > 0:
            g_infer_state_lock.acquire()
            g_infer_context.req_manager.mem_manager.free(need_free_mem_indexes)
            g_infer_state_lock.release()

        # 第四阶段
        event_pack.notify_pre_post_handle()
        return

    def _draft_verify(self, model_input: ModelInput, run_reqs: List[InferReq]):
        """
        main model forward and verify the next_token_ids
        """
        b_mtp_index_cpu = model_input.b_mtp_index
        model_output = self.model.forward(model_input)
        all_next_token_ids = []
        next_token_ids, next_token_logprobs = sample(model_output.logits, run_reqs, self.eos_id)
        all_next_token_ids.append(next_token_ids)
        # verify the next_token_ids
        b_req_mtp_start_loc = [index for index, mtp_index in enumerate(b_mtp_index_cpu) if mtp_index == 0]
        b_req_mtp_start_loc = g_pin_mem_manager.gen_from_list(
            key="b_req_mtp_start_loc",
            data=b_req_mtp_start_loc,
            dtype=torch.int32,
        ).cuda(non_blocking=True)

        mtp_accept_len, accepted_index = self._verify_mtp_v2(
            new_next_token_ids=next_token_ids,
            b_req_idx=model_input.b_req_idx,
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

        next_token_ids_cpu, next_token_logprobs_cpu = self._async_copy_next_token_infos_to_pin_mem(
            next_token_ids, next_token_logprobs
        )
        return {
            "verify_event": verify_event,
            "main_model_output": model_output,
            "b_req_mtp_start_loc": b_req_mtp_start_loc,
            "mtp_accept_len": mtp_accept_len,
            "accepted_index": accepted_index,
            "next_token_ids": next_token_ids,
            "accepted_index_cpu": accepted_index_cpu,
            "mtp_accept_len_cpu": mtp_accept_len_cpu,
            "next_token_ids_cpu": next_token_ids_cpu,
            "next_token_logprobs_cpu": next_token_logprobs_cpu,
        }

    def _draft_decode_vanilla(
        self,
        main_model_input: ModelInput,
        verify_info: Dict[str, Any],
    ):
        main_model_output = verify_info["main_model_output"]
        next_token_ids = verify_info["next_token_ids"]
        mtp_accept_len = verify_info["mtp_accept_len"]
        b_req_mtp_start_loc = verify_info["b_req_mtp_start_loc"]

        # share some inference info with the main model
        draft_model_input = main_model_input
        draft_model_output = main_model_output
        draft_next_token_ids = next_token_ids
        all_next_token_ids = []
        all_next_token_ids.append(next_token_ids)
        # process the draft model output
        for draft_model_idx in range(self.mtp_step):

            draft_model_input.input_ids = draft_next_token_ids
            draft_model_input.deepseekv3_mtp_draft_input_hiddens = draft_model_output.deepseekv3_mtp_main_output_hiddens
            # spec decode: MTP
            draft_model_output: ModelOutput = self.draft_models[draft_model_idx].forward(draft_model_input)
            draft_next_token_ids = self._gen_argmax_token_ids(draft_model_output)
            all_next_token_ids.append(draft_next_token_ids)

        all_next_token_ids = torch.stack(all_next_token_ids, dim=1)  # [batch_size, mtp_step + 1]

        mtp_scatter_next_token_ids(
            req_to_next_token_ids=self.model.req_manager.req_sampling_params_manager.req_to_next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            all_next_token_ids=all_next_token_ids,
            b_req_idx=main_model_input.b_req_idx,
            mtp_accept_len=mtp_accept_len,
        )
        return

    def _draft_decode_eagle(
        self,
        main_model_input: ModelInput,
        verify_info: Dict[str, Any],
        eagle_mem_indexes_cpu: torch.Tensor,
        draft_model_input: ModelInput,
    ):

        main_model_output = verify_info["main_model_output"]
        next_token_ids = verify_info["next_token_ids"]
        mtp_accept_len = verify_info["mtp_accept_len"]

        all_next_token_ids = []
        all_next_token_ids.append(next_token_ids[mtp_accept_len - 1])

        # 第一步draft，填充接受的token的kv。
        main_model_input.input_ids = next_token_ids
        main_model_input.deepseekv3_mtp_draft_input_hiddens = main_model_output.deepseekv3_mtp_main_output_hiddens
        draft_model_output = self.draft_models[0].forward(main_model_input)
        draft_next_token_ids = self._gen_argmax_token_ids(draft_model_output)
        all_next_token_ids.append(draft_next_token_ids[mtp_accept_len - 1])

        # 剩余的draft step。
        draft_model_input.input_ids = draft_next_token_ids[mtp_accept_len - 1]
        draft_model_input.b_seq_len = draft_model_input.b_seq_len.cuda(non_blocking=True)
        draft_model_input.b_seq_len += mtp_accept_len
        draft_model_input.max_len_in_batch += self.mtp_step * 2
        draft_model_input.deepseekv3_mtp_draft_input_hiddens = draft_model_output.deepseekv3_mtp_main_output_hiddens[
            mtp_accept_len - 1
        ]
        for _step in range(1, self.mtp_step):
            draft_batch_size = draft_model_input.batch_size
            draft_model_input.mem_indexes = None
            draft_model_input.mem_indexes_cpu = eagle_mem_indexes_cpu[
                (_step - 1) * draft_batch_size : (_step * draft_batch_size)
            ]
            draft_model_output = self.draft_models[0].forward(draft_model_input)
            draft_next_token_ids = self._gen_argmax_token_ids(draft_model_output)
            draft_model_input.input_ids = draft_next_token_ids
            draft_model_input.b_seq_len += 1
            draft_model_input.deepseekv3_mtp_draft_input_hiddens = draft_model_output.deepseekv3_mtp_main_output_hiddens
            all_next_token_ids.append(draft_next_token_ids)

        all_next_token_ids = torch.stack(all_next_token_ids, dim=-1)  # [batch_size, mtp_step + 1]
        self.model.req_manager.req_sampling_params_manager.req_to_next_token_ids[
            draft_model_input.b_req_idx, : self.mtp_step + 1
        ] = all_next_token_ids
        return
