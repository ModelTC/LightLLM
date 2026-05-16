import torch
import time
from typing import List, Optional, Callable, Dict, Any
from queue import Queue
from lightllm.server.router.model_infer.mode_backend.update_mem_index import update_eagle_mem_indexes_triton
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.server.router.model_infer.mode_backend.overlap_events import OverlapEventPack
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.server.router.model_infer.mode_backend.pre import (
    prepare_prefill_inputs,
    prepare_decode_inputs,
)
from lightllm.server.router.model_infer.mode_backend.mtp_pre_process import (
    prepare_mtp_prefill_inputs,
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
from lightllm.utils.envs_utils import get_env_start_args, enable_dynamic_mtp_verify
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
            self.is_mtp_eagle = get_env_start_args().mtp_mode in ["eagle_with_att", "eagle_no_att", "eagle3"]
            self.num_mtp_models = 1 if self.is_mtp_eagle else get_env_start_args().mtp_step
            self._draft_decode_func = self._draft_decode_eagle if self.is_mtp_eagle else self._draft_decode_vanilla
            self.enable_dynamic_mtp = enable_dynamic_mtp_verify()
        else:
            self.prefill = self.prefill_normal
            self.decode = self.decode_normal

        self.classed_req_strict_prefill = False
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
            nixl_prefill_chuncked_handle_func=self.nixl_prefill_chuncked_handle_func,
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
            self._draft_prefill_forward(
                model_input=model_input, model_output=model_output, next_token_ids=next_token_ids
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
            nixl_prefill_chuncked_handle_func=self.nixl_prefill_chuncked_handle_func,
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

        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            b_mtp_index_cpu = model_input.b_mtp_index
            model_output = self.model.forward(model_input)
            next_token_ids, next_token_logprobs = sample(model_output.logits, run_reqs, self.eos_id)
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

            verify_event = torch.cuda.Event()
            verify_event.record()

            if self.enable_dynamic_mtp:
                all_next_token_ids, additional_mem_indexes_cpu, draft_probs_list = self._draft_decode_func(
                    main_model_input=model_input,
                    main_model_output=model_output,
                    next_token_ids=next_token_ids,
                    b_req_mtp_start_loc=b_req_mtp_start_loc,
                )
            else:
                all_next_token_ids, additional_mem_indexes_cpu = self._draft_decode_func(
                    main_model_input=model_input,
                    main_model_output=model_output,
                    next_token_ids=next_token_ids,
                    b_req_mtp_start_loc=b_req_mtp_start_loc,
                )

            # dynamic_sizes_gpu 用于第二阶段更新 req 的 mtp_size
            if self.enable_dynamic_mtp:
                draft_probs_tensor = torch.cat(draft_probs_list, dim=-1).view(self.mtp_step, b_mtp_index_cpu.shape[0])
                dynamic_sizes_gpu = self._compute_dynamic_mtp_size_gpu_part(draft_probs_tensor=draft_probs_tensor)
                # 异步拷贝回 CPU Pin Memory
                dynamic_sizes_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor(
                    key="dynamic_mtp_sizes", gpu_tensor=dynamic_sizes_gpu
                )

                dynamic_mtp_event = torch.cuda.Event()
                dynamic_mtp_event.record()

            mtp_scatter_next_token_ids(
                req_to_next_token_ids=self.model.req_manager.req_sampling_params_manager.req_to_next_token_ids,
                b_req_mtp_start_loc=b_req_mtp_start_loc,
                all_next_token_ids=all_next_token_ids,
                b_req_idx=model_input.b_req_idx,
                mtp_accept_len=mtp_accept_len,
            )

            next_token_ids_cpu, next_token_logprobs_cpu = self._async_copy_next_token_infos_to_pin_mem(
                next_token_ids, next_token_logprobs
            )

            g_infer_context.req_sampling_manager.update_reqs_out_token_counter_gpu(
                b_req_idx=model_input.b_req_idx,
                next_token_ids=next_token_ids,
                mask=accepted_index == 1,
            )

            mtp_accept_len_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor(
                key="mtp_accept_len",
                gpu_tensor=mtp_accept_len,
            )

            sync_event = torch.cuda.Event()
            sync_event.record()

        # 第二阶段
        event_pack.notify_post_handle_and_wait_pre_post_handle()
        self._update_mtp_verify_token_num(decode_reqs=decode_reqs)

        verify_event.synchronize()
        accepted_index_cpu_numpy = accepted_index_cpu.numpy()
        verify_ok_reqs = [run_reqs[i] for i in range(len(run_reqs)) if accepted_index_cpu_numpy[i] == 1]
        if self.enable_dynamic_mtp:
            dynamic_mtp_event.synchronize()
            self._update_dynamic_mtp_size_cpu_part(
                run_reqs=run_reqs, dynamic_sizes_cpu=dynamic_sizes_cpu, accepted_index_cpu=accepted_index_cpu
            )
        update_packs = self._pre_post_handle(verify_ok_reqs, is_chuncked_mode=False)

        # 第三阶段
        event_pack.notify_forward_and_wait_post_handle()
        sync_event.synchronize()

        # 处理需要释放的内存索引
        need_free_mem_indexes = model_input.mem_indexes_cpu[accepted_index_cpu == 0]
        if additional_mem_indexes_cpu is not None:
            need_free_mem_indexes = torch.cat([need_free_mem_indexes, additional_mem_indexes_cpu], dim=0)

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
        return

    def _compute_dynamic_mtp_size_gpu_part(
        self,
        draft_probs_tensor: torch.Tensor,
    ) -> torch.Tensor:
        rand_vals = torch.rand_like(draft_probs_tensor)
        accepted_mask = draft_probs_tensor > rand_vals
        valid_steps = torch.cumprod(accepted_mask.to(torch.int32), dim=0)
        dynamic_mtp_sizes = valid_steps.sum(dim=0)
        return dynamic_mtp_sizes

    def _update_dynamic_mtp_size_cpu_part(
        self,
        run_reqs: List[InferReq],
        dynamic_sizes_cpu: torch.Tensor,
        accepted_index_cpu: torch.Tensor,
    ):
        assert len(run_reqs) == dynamic_sizes_cpu.shape[0] == accepted_index_cpu.shape[0]
        for req, new_size, accepted in zip(run_reqs, dynamic_sizes_cpu.numpy(), accepted_index_cpu.numpy()):
            if int(accepted) == 1:
                req.current_mtp_step = int(new_size)
                assert req.current_mtp_step <= req.mtp_step

    def _draft_prefill_forward(self, model_input: ModelInput, model_output: ModelOutput, next_token_ids: torch.Tensor):
        # spec prefill: MTP, 这个地方只是为了填充draft model的 kv， 并不会使用生成的token_id。
        draft_model_input = model_input
        draft_model_output = model_output
        draft_next_token_ids_gpu = next_token_ids
        for draft_model_idx in range(self.num_mtp_models):
            draft_model_input = prepare_mtp_prefill_inputs(
                model_input=draft_model_input,
                b_next_token_ids=draft_next_token_ids_gpu,
                mtp_draft_input_hiddens=draft_model_output.mtp_main_output_hiddens,
            )
            draft_model_output = self.draft_models[draft_model_idx].forward(draft_model_input)
            draft_next_token_ids_gpu = self._gen_argmax_token_ids(draft_model_output)
        return

    def _draft_decode_vanilla(
        self,
        main_model_input: ModelInput,
        main_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
    ):
        # share some inference info with the main model
        draft_model_input = main_model_input
        draft_model_output = main_model_output
        draft_next_token_ids = next_token_ids
        all_next_token_ids = []
        all_next_token_ids.append(next_token_ids)
        # process the draft model output
        for draft_model_idx in range(self.mtp_step):
            draft_model_input.input_ids = draft_next_token_ids
            draft_model_input.mtp_draft_input_hiddens = draft_model_output.mtp_main_output_hiddens
            # spec decode: MTP
            draft_model_output: ModelOutput = self.draft_models[draft_model_idx].forward(draft_model_input)
            draft_next_token_ids = self._gen_argmax_token_ids(draft_model_output)
            all_next_token_ids.append(draft_next_token_ids)

        all_next_token_ids = torch.stack(all_next_token_ids, dim=1)  # [batch_size, mtp_step + 1]

        return all_next_token_ids, None

    def _draft_decode_eagle(
        self,
        main_model_input: ModelInput,
        main_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
    ):

        num_reqs = b_req_mtp_start_loc.shape[0]

        g_infer_state_lock.acquire()
        if g_infer_context.radix_cache is not None:
            g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(num_reqs * self.mtp_step)
        eagle_mem_indexes_cpu = g_infer_context.req_manager.mem_manager.alloc(num_reqs * self.mtp_step)
        g_infer_state_lock.release()
        eagle_mem_indexes = eagle_mem_indexes_cpu.cuda(non_blocking=True)

        # share some inference info with the main model
        draft_model_input = main_model_input
        draft_model_output = main_model_output
        draft_next_token_ids = next_token_ids
        all_next_token_ids = []
        all_next_token_ids.append(next_token_ids)

        # 用于收集每个 step 的 probs
        draft_probs_list = [] if self.enable_dynamic_mtp else None

        # process the draft model output
        for _step in range(self.mtp_step):
            draft_model_input.input_ids = draft_next_token_ids
            draft_model_input.mtp_draft_input_hiddens = draft_model_output.mtp_main_output_hiddens
            # spec decode: MTP
            draft_model_idx = _step % self.num_mtp_models
            draft_model_output: ModelOutput = self.draft_models[draft_model_idx].forward(draft_model_input)

            # 收集 probs（如果需要）
            if self.enable_dynamic_mtp:
                draft_next_token_ids, draft_probs = self._gen_argmax_token_ids_and_prob(draft_model_output)
                draft_probs_list.append(draft_probs)
            else:
                draft_next_token_ids = self._gen_argmax_token_ids(draft_model_output)
            draft_model_input.b_seq_len += 1
            draft_model_input.max_kv_seq_len += 1
            eagle_mem_indexes_i = eagle_mem_indexes[_step * num_reqs : (_step + 1) * num_reqs]

            if self.enable_dynamic_mtp:
                draft_model_input.mem_indexes = update_eagle_mem_indexes_triton(
                    old_mem_indexes=draft_model_input.mem_indexes,
                    new_step_mem_indexes=eagle_mem_indexes_i,
                    b_req_mtp_start_loc=b_req_mtp_start_loc,
                )
            else:
                draft_model_input.mem_indexes = torch.cat(
                    [draft_model_input.mem_indexes.view(-1, self.mtp_step + 1)[:, 1:], eagle_mem_indexes_i.view(-1, 1)],
                    dim=1,
                ).view(-1)
            all_next_token_ids.append(draft_next_token_ids)

        all_next_token_ids = torch.stack(all_next_token_ids, dim=1)  # [batch_size, mtp_step + 1]

        if self.enable_dynamic_mtp:
            return all_next_token_ids, eagle_mem_indexes_cpu, draft_probs_list
        else:
            return all_next_token_ids, eagle_mem_indexes_cpu
