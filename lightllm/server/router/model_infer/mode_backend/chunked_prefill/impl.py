import os
import torch
import time
import copy
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
)
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager
from lightllm.common.basemodel.batch_objs import ModelOutput, ModelInput
from lightllm.common.basemodel.triton_kernel.gather_token_id import scatter_token
from lightllm.common.basemodel.triton_kernel.copy_kv_index_to_req import copy_kv_index_to_req
from lightllm.common.basemodel.triton_kernel.mtp_utils import (
    mtp_scatter_next_token_ids,
    scatter_mtp_accept_len,
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
            self.is_mtp_eagle = get_env_start_args().mtp_mode in ["eagle_with_att", "eagle_no_att"]
            self.num_mtp_models = 1 if self.is_mtp_eagle else get_env_start_args().mtp_step
            self._draft_decode_func = self._draft_decode_eagle if self.is_mtp_eagle else self._draft_decode_vanilla
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
            self._draft_prefill_forward(
                model_input=model_input, model_output=model_output, next_token_ids=next_token_ids
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

    def _init_mtp_fused_graph(self):
        self.mtp_fused_graph = None
        self._init_mtp_chain_scratch()
        if get_env_start_args().disable_cudagraph:
            return
        if not (self.is_mtp_eagle and self.num_mtp_models == 1):
            return
        if self.enable_decode_microbatch_overlap or self.args.dp > 1:
            return
        from lightllm.utils.envs_utils import enable_diverse_mode_gqa_decode_fast_kernel

        if enable_diverse_mode_gqa_decode_fast_kernel():
            return
        if os.getenv("LIGHTLLM_DISABLE_MTP_FUSED_GRAPH", "0") == "1":
            logger.info("mtp fused decode graph disabled by env")
            return
        from lightllm.common.basemodel.attention.flashinfer.fp import FlashInferAttBackend

        if isinstance(self.model.decode_att_backend, FlashInferAttBackend) or isinstance(
            self.draft_models[0].decode_att_backend, FlashInferAttBackend
        ):
            # flashinfer 的 decode init_state 每步构建 wrapper 并 plan (含 D2H 拷贝),
            # 无法被捕获进图内; 用 --llm_decode_att_backend fa3/triton 可启用 fused graph。
            logger.warning("mtp fused decode graph disabled: flashinfer decode att backend is not capture-safe")
            return
        from .mtp_fused_decode_graph import MTPFusedDecodeGraph

        self.mtp_fused_graph = MTPFusedDecodeGraph(backend=self)
        self.mtp_fused_graph.warmup()
        return

    def _init_mtp_chain_scratch(self):
        # chain step>=1 的 kv scratch 槽位: 每轮内容都是丢弃的, 启动时(池空)一次性分配、
        # 每轮复用, 避免运行期在 kv 高水位下按轮分配触发 radix 驱逐失败。
        self.mtp_chain_scratch = None
        if not (self.is_mtp_eagle and self.mtp_step > 1):
            return
        max_rows = get_env_start_args().running_max_req_size * (self.mtp_step + 1)
        scratch_num = max_rows * (self.mtp_step - 1)
        scratch_cpu = g_infer_context.req_manager.mem_manager.alloc(scratch_num)
        self.mtp_chain_scratch = scratch_cpu.cuda()
        logger.info(f"mtp chain scratch kv slots reserved: {scratch_num}")
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

        if self.mtp_fused_graph is not None and self.mtp_fused_graph.can_run(
            decode_reqs=decode_reqs,
            max_kv_seq_len=model_input.max_kv_seq_len,
            batch_size=model_input.batch_size,
        ):
            self._decode_mtp_fused(
                event_pack=event_pack,
                model_input=model_input,
                run_reqs=run_reqs,
                decode_reqs=decode_reqs,
            )
            return

        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            model_output = self.model.forward(model_input)
            next_token_ids, next_token_logprobs = sample(model_output.logits, run_reqs, self.eos_id)
            # verify the next_token_ids
            n_real = model_input.batch_size // (self.mtp_step + 1)
            b_req_mtp_start_loc = torch.arange(n_real, dtype=torch.int32, device="cuda") * (self.mtp_step + 1)

            mtp_accept_len, accepted_index = self._verify_mtp_v2(
                new_next_token_ids=next_token_ids,
                b_req_idx=model_input.b_req_idx,
                b_req_mtp_start_loc=b_req_mtp_start_loc,
            )
            scatter_mtp_accept_len(
                self.model.req_manager.req_to_accept_len, b_req_mtp_start_loc, model_input.b_req_idx, mtp_accept_len
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

            # 调用具体的draft decode函数
            additional_mem_indexes_cpu = self._draft_decode_func(
                main_model_input=model_input,
                main_model_output=model_output,
                next_token_ids=next_token_ids,
                mtp_accept_len=mtp_accept_len,
                b_req_mtp_start_loc=b_req_mtp_start_loc,
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
        verify_event.synchronize()
        verify_ok_reqs = [run_reqs[i] for i in range(len(run_reqs)) if accepted_index_cpu[i] == 1]
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
            g_infer_context.req_manager.mem_manager.free(need_free_mem_indexes)

        # 第四阶段
        event_pack.notify_pre_post_handle()
        return

    def _decode_mtp_fused(
        self,
        event_pack: OverlapEventPack,
        model_input: ModelInput,
        run_reqs: List[InferReq],
        decode_reqs: List[InferReq],
    ):
        """
        整个 mtp decode step (verify fwd + 采样 + verify + draft chain) 单 cuda graph 的执行路径。
        与 decode_mtp 的阶段/事件协议保持一致。
        """
        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            fused_out = self.mtp_fused_graph.replay_verify(model_input=model_input, run_reqs=run_reqs)
            accepted_index_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor(
                key="accepted_index",
                gpu_tensor=fused_out.accepted_index,
            )
            mtp_accept_len_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor(
                key="mtp_accept_len",
                gpu_tensor=fused_out.mtp_accept_len,
            )
            next_token_ids_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor(
                key="next_token_ids",
                gpu_tensor=fused_out.next_token_ids,
            )
            next_token_logprobs_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor(
                key="next_token_logprobs",
                gpu_tensor=fused_out.next_token_logprobs,
            )
            verify_event = torch.cuda.Event()
            verify_event.record()

            self.mtp_fused_graph.replay_draft()
            sync_event = torch.cuda.Event()
            sync_event.record()

        # 第二阶段
        event_pack.notify_post_handle_and_wait_pre_post_handle()
        verify_event.synchronize()
        verify_ok_reqs = [run_reqs[i] for i in range(len(run_reqs)) if accepted_index_cpu[i] == 1]
        update_packs = self._pre_post_handle(verify_ok_reqs, is_chuncked_mode=False)

        # 第三阶段
        event_pack.notify_forward_and_wait_post_handle()
        sync_event.synchronize()

        need_free_mem_indexes = model_input.mem_indexes_cpu[accepted_index_cpu == 0]

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
            g_infer_context.req_manager.mem_manager.free(need_free_mem_indexes)

        # 第四阶段
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
        mtp_accept_len: torch.Tensor,
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

        mtp_scatter_next_token_ids(
            req_to_next_token_ids=self.model.req_manager.req_sampling_params_manager.req_to_next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            all_next_token_ids=all_next_token_ids,
            b_req_idx=main_model_input.b_req_idx,
            mtp_accept_len=mtp_accept_len,
        )
        return None

    def _draft_decode_eagle(
        self,
        main_model_input: ModelInput,
        main_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
        mtp_accept_len: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
    ):
        batch_size = main_model_input.batch_size
        # chain step>=1 只写静态 scratch 槽位, 不覆盖 verify 槽位 (那里保存着 step0 写入的
        # (采样 token, 主模型 hidden) 正确 kv, 是下一轮 draft 的前缀上下文)。
        verify_mem_indexes = main_model_input.mem_indexes
        verify_b_seq_len = main_model_input.b_seq_len.clone()

        # share some inference info with the main model
        draft_model_input = main_model_input
        draft_model_output = main_model_output
        draft_next_token_ids = next_token_ids
        all_next_token_ids = []
        all_next_token_ids.append(next_token_ids)
        # process the draft model output
        for _step in range(self.mtp_step):

            draft_model_input.input_ids = draft_next_token_ids
            draft_model_input.mtp_draft_input_hiddens = draft_model_output.mtp_main_output_hiddens
            # spec decode: MTP
            draft_model_idx = _step % self.num_mtp_models
            draft_model_output: ModelOutput = self.draft_models[draft_model_idx].forward(draft_model_input)
            draft_next_token_ids = self._gen_argmax_token_ids(draft_model_output)
            draft_model_input.b_seq_len += 1
            draft_model_input.max_kv_seq_len += 1
            if _step + 1 < self.mtp_step:
                draft_model_input.mem_indexes = self.mtp_chain_scratch[_step * batch_size : (_step + 1) * batch_size]
            all_next_token_ids.append(draft_next_token_ids)

        if self.mtp_step > 1:
            # 恢复 verify 位置 -> verify 槽位的映射 (step0 的正确 kv), 消除 chain 写入的污染。
            copy_kv_index_to_req(
                self.model.req_manager.req_to_token_indexs,
                main_model_input.b_req_idx,
                verify_b_seq_len,
                verify_mem_indexes,
            )
        draft_model_input.mem_indexes = verify_mem_indexes
        draft_model_input.b_seq_len.copy_(verify_b_seq_len)

        all_next_token_ids = torch.stack(all_next_token_ids, dim=1)  # [batch_size, mtp_step + 1]

        mtp_scatter_next_token_ids(
            req_to_next_token_ids=self.model.req_manager.req_sampling_params_manager.req_to_next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            all_next_token_ids=all_next_token_ids,
            b_req_idx=main_model_input.b_req_idx,
            mtp_accept_len=mtp_accept_len,
        )
        return None
