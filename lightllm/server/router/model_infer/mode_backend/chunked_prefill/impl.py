import torch
import time
from typing import List, Optional, Callable, Dict, Any
from queue import Queue
from transformers import AutoTokenizer

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
from lightllm.utils.log_utils import init_logger, print_rank0
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.envs_utils import get_env_start_args, enable_dynamic_mtp_verify
from .control_state import ControlState

logger = init_logger(__name__)

tokenizer = AutoTokenizer.from_pretrained("/mtc/models/qwen3-8b", trust_remote_code=True)


def compute_dynamic_mtp_size(
    probs: torch.Tensor,
    reqs: List[InferReq],
    max_mtp_size: int,
    draft_probs: Optional[List[torch.Tensor]] = None,
) -> List[int]:
    """
    根据 prob 分布计算每个请求的动态 MTP 验证长度

    Args:
        probs: [batch_size, vocab_size] - 主模型每个 token 的概率分布
        reqs: 当前请求列表
        max_mtp_size: 最大 MTP 长度（来自 args.mtp_step）
        draft_probs: 可选，draft 模型每个 step 的 probs 列表，每个元素形状为 [batch_size, vocab_size]

    Returns:
        List[int]: 每个请求的动态 MTP 验证长度
    """
    # TODO: 实现更复杂的动态 MTP 长度计算逻辑
    # 可以考虑以下因素：
    # 1. 主模型 probs 的熵 - 熵越低表示置信度越高，可以增加验证长度
    # 2. draft 模型和主模型的一致性 - 一致性越高，验证通过率可能越高
    # 3. draft 模型 probs 的熵 - 评估 draft 模型的置信度
    #
    # 当前实现：随机返回一个 [1, max_mtp_size] 之间的值作为占位符
    batch_size = probs.shape[0]
    dynamic_mtp_sizes = []
    for i in range(batch_size):
        # 随机生成一个 [1, max_mtp_size] 之间的值
        dynamic_mtp_size = torch.randint(1, max_mtp_size + 1, (1,)).item()
        dynamic_mtp_sizes.append(dynamic_mtp_size)
    return dynamic_mtp_sizes


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
            print_rank0("mtp_accept_len:", mtp_accept_len)
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

            # 判断是否启用动态 MTP 验证
            enable_dynamic_mtp = enable_dynamic_mtp_verify() and self.mtp_step > 0

            # 调用具体的 draft decode 函数
            # 返回格式：(additional_mem_indexes_cpu, draft_probs_list)
            additional_mem_indexes_cpu, draft_probs_list = self._draft_decode_func(
                main_model_input=model_input,
                main_model_output=model_output,
                next_token_ids=next_token_ids,
                mtp_accept_len=mtp_accept_len,
                b_req_mtp_start_loc=b_req_mtp_start_loc,
                enable_dynamic_mtp=enable_dynamic_mtp,
            )

            # 启用动态 MTP 验证时，根据主模型和 draft 模型的 probs 计算下一轮的动态验证长度
            if enable_dynamic_mtp:
                self._update_dynamic_mtp_size(
                    main_model_logits=model_output.logits,
                    run_reqs=run_reqs,
                    b_mtp_index_cpu=b_mtp_index_cpu,
                    decode_reqs=decode_reqs,
                    draft_probs_list=draft_probs_list,
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
            g_infer_state_lock.acquire()
            g_infer_context.req_manager.mem_manager.free(need_free_mem_indexes)
            g_infer_state_lock.release()

        # 第四阶段
        event_pack.notify_pre_post_handle()
        return

    def _update_dynamic_mtp_size(
        self,
        main_model_logits: torch.Tensor,
        run_reqs: List[InferReq],
        b_mtp_index_cpu: List[int],
        decode_reqs: List[InferReq],
        draft_probs_list: List[torch.Tensor],
    ):
        """
        根据主模型和 draft 模型的 probs 计算每个请求下一轮的动态 MTP 验证长度

        Args:
            main_model_logits: 主模型 logits [batch_size, vocab_size]
            run_reqs: 扩展后的请求列表（包含主 token + draft token）
            b_mtp_index_cpu: 每个请求的 mtp_index
            decode_reqs: 原始请求列表
            draft_probs_list: 每个 draft step 的 probs 列表，每个元素形状为 [batch_size, vocab_size]
        """
        # 收集主模型和 draft 模型的 probs
        # main_probs: 每个请求主 token 的 probs (mtp_index == 0)
        # all_draft_probs: 每个请求每个 draft step 的 probs
        main_probs = []
        all_draft_probs = []  # [step][req_idx]

        for i, mtp_index in enumerate(b_mtp_index_cpu):
            probs = torch.softmax(main_model_logits[i], dim=-1)
            if mtp_index == 0:
                main_probs.append(probs)
            else:
                # 确保 all_draft_probs 列表足够长
                while len(all_draft_probs) < mtp_index:
                    all_draft_probs.append([])
                all_draft_probs[mtp_index - 1].append(probs)

        # 计算动态 MTP 大小
        if len(main_probs) > 0:
            main_probs = torch.stack(main_probs, dim=0)  # [num_reqs, vocab_size]

            # 将 draft probs 也 stack 起来
            draft_probs_stacked = []
            for step_probs in all_draft_probs:
                if len(step_probs) > 0:
                    draft_probs_stacked.append(torch.stack(step_probs, dim=0))

            # 使用 placeholder 逻辑计算动态 MTP 大小
            # TODO: 实现更复杂的逻辑，结合 main_probs 和 draft_probs_stacked
            dynamic_mtp_sizes = compute_dynamic_mtp_size(
                probs=main_probs,
                reqs=decode_reqs,
                max_mtp_size=self.mtp_step,
                draft_probs=draft_probs_stacked,
            )

            # 更新每个请求的 mtp_size
            for req, mtp_size in zip(decode_reqs, dynamic_mtp_sizes):
                req.mtp_size = mtp_size
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
        enable_dynamic_mtp: bool = False,
    ):
        """
        Vanilla MTP draft decode.

        Args:
            enable_dynamic_mtp: 如果为 True，则收集每个 step 的 probs 用于动态 MTP 长度计算
        Returns:
            additional_mem_indexes_cpu: None
            draft_probs_list: 如果 enable_dynamic_mtp 为 True，返回每个 step 的 probs 列表
        """
        # share some inference info with the main model
        draft_model_input = main_model_input
        draft_model_output = main_model_output
        draft_next_token_ids = next_token_ids
        all_next_token_ids = []
        all_next_token_ids.append(next_token_ids)

        # 用于收集每个 step 的 probs
        draft_probs_list = [] if enable_dynamic_mtp else None

        # process the draft model output
        for draft_model_idx in range(self.mtp_step):
            draft_model_input.input_ids = draft_next_token_ids
            draft_model_input.mtp_draft_input_hiddens = draft_model_output.mtp_main_output_hiddens
            # spec decode: MTP
            draft_model_output: ModelOutput = self.draft_models[draft_model_idx].forward(draft_model_input)

            # 收集 probs（如果需要）
            if enable_dynamic_mtp:
                draft_logits = draft_model_output.logits
                draft_probs = torch.softmax(draft_logits, dim=-1)
                draft_probs_list.append(draft_probs)

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
        return None, draft_probs_list

    def _draft_decode_eagle(
        self,
        main_model_input: ModelInput,
        main_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
        mtp_accept_len: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        enable_dynamic_mtp: bool = False,
    ):
        """
        Eagle MTP draft decode.

        Args:
            enable_dynamic_mtp: 如果为 True，则收集每个 step 的 probs 用于动态 MTP 长度计算
        Returns:
            eagle_mem_indexes_cpu: 额外的内存索引
            draft_probs_list: 如果 enable_dynamic_mtp 为 True，返回每个 step 的 probs 列表
        """
        if not enable_dynamic_mtp:
            # 如果是静态MTP验证模式，那么组的大小很容易确定
            batch_size = main_model_input.batch_size
            num_reqs = batch_size // (self.mtp_step + 1)
        else:
            # 如果不是，那么我们需要从main_model_input.b_mtp_index_cpu中计算出每个请求的组大小
            b_mtp_index_cpu = main_model_input.b_mtp_index.cpu().tolist()
            batch_size = main_model_input.batch_size
            # num_reqs是指主模型的请求数量，也就是mtp_index为0的请求数量
            num_reqs = b_mtp_index_cpu.count(0)
            # 额外添加一项存储每个组的大小
            mtp_group_sizes = []
            current_group_size = 0
            for mtp_index in b_mtp_index_cpu:
                if mtp_index == 0:
                    if current_group_size > 0:
                        mtp_group_sizes.append(current_group_size)
                    current_group_size = 1
                else:
                    current_group_size += 1
            if current_group_size > 0:
                mtp_group_sizes.append(current_group_size)
            print_rank0(f"mtp_group_sizes: {mtp_group_sizes}")
        
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
        draft_probs_list = [] if enable_dynamic_mtp else None

        # process the draft model output
        for _step in range(self.mtp_step):
            draft_model_input.input_ids = draft_next_token_ids
            draft_model_input.mtp_draft_input_hiddens = draft_model_output.mtp_main_output_hiddens
            # spec decode: MTP
            draft_model_idx = _step % self.num_mtp_models
            draft_model_output: ModelOutput = self.draft_models[draft_model_idx].forward(draft_model_input)

            # 收集 probs（如果需要）
            if enable_dynamic_mtp:
                draft_logits = draft_model_output.logits
                draft_probs = torch.softmax(draft_logits, dim=-1)
                draft_probs_list.append(draft_probs)

            draft_next_token_ids = self._gen_argmax_token_ids(draft_model_output)
            draft_model_input.b_seq_len += 1
            draft_model_input.max_kv_seq_len += 1
            eagle_mem_indexes_i = eagle_mem_indexes[_step * num_reqs : (_step + 1) * num_reqs]

            # draft_model_input.mem_indexes = torch.cat(
            #     [draft_model_input.mem_indexes.view(-1, self.mtp_step + 1)[:, 1:], eagle_mem_indexes_i.view(-1, 1)],
            #     dim=1,
            # ).view(-1)
            # 1. 根据当前的 group_sizes 将原来的索引拆分成多个组
            # 这里的 group_sizes 应该对应之前未处理前的每一组的大小
            chunks = torch.split(draft_model_input.mem_indexes, mtp_group_sizes)
            # 2. 对每一个 chunk 进行处理：去掉第一个元素 ([:, 1:])，并加上对应的 eagle_mem_indexes_i 元素
            # 假设 eagle_mem_indexes_i 的形状是 (num_groups,)
            new_chunks = []
            for i, chunk in enumerate(chunks):
                # chunk[1:] 模拟了原来的 [:, 1:] 操作
                # eagle_mem_indexes_i[i:i+1] 确保拿出来的是一个长度为 1 的张量用于拼接
                updated_chunk = torch.cat([chunk[1:], eagle_mem_indexes_i[i:i+1]], dim=0)
                new_chunks.append(updated_chunk)
            # 3. 重新合并回一维张量
            draft_model_input.mem_indexes = torch.cat(new_chunks, dim=0)
            
            all_next_token_ids.append(draft_next_token_ids)

        all_next_token_ids = torch.stack(all_next_token_ids, dim=1)  # [batch_size, mtp_step + 1]

        mtp_scatter_next_token_ids(
            req_to_next_token_ids=self.model.req_manager.req_sampling_params_manager.req_to_next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            all_next_token_ids=all_next_token_ids,
            b_req_idx=main_model_input.b_req_idx,
            mtp_accept_len=mtp_accept_len,
        )
        return eagle_mem_indexes_cpu, draft_probs_list
