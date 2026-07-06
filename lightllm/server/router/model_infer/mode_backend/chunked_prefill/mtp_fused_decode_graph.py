import random
import bisect
import torch
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

from lightllm.common.basemodel.batch_objs import ModelInput
from lightllm.common.basemodel.triton_kernel.copy_kv_index_to_req import copy_kv_index_to_req
from lightllm.common.basemodel.triton_kernel.gather_token_id import gather_token
from lightllm.common.basemodel.triton_kernel.mtp_utils import (
    mtp_verify,
    mtp_scatter_next_token_ids,
    scatter_mtp_accept_len,
)
from lightllm.server.router.model_infer.infer_batch import InferReq, g_infer_context
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_env_start_args

if TYPE_CHECKING:
    from .impl import ChunkedPrefillBackend

logger = init_logger(__name__)


@dataclass
class FusedStepOutput:
    next_token_ids: torch.Tensor
    mtp_accept_len: torch.Tensor
    accepted_index: torch.Tensor
    eagle_mem_indexes_cpu: torch.Tensor


@dataclass
class _GraphBundle:
    verify_graph: torch.cuda.CUDAGraph
    draft_graph: torch.cuda.CUDAGraph
    mtp_accept_len: torch.Tensor
    accepted_index: torch.Tensor
    main_hiddens: torch.Tensor


class MTPFusedDecodeGraph:
    """
    将一个 mtp eagle decode step 捕获为两个 cuda graph:

    - verify graph: input gather + 主模型 forward + (惩罚无关的) 采样 + mtp verify
      + accept_len / out-token-counter 更新。它的输出 (next_token_ids / accept 信息)
      是 cpu 后处理需要读取的全部内容。
    - draft graph: mtp_step 次 draft forward + argmax 链 + 候选 token scatter。
      不产生 cpu 需要的输出, 因此可以在 verify 结果回传后异步在 gpu 上运行,
      掩盖下一个 step 的 cpu 准备时间 (与旧路径的 verify_event/sync_event 结构一致)。

    对比逐 forward 捕获的旧路径, 一个 step 只有两次 graph launch, 消除了 forward 之间
    的 eager glue (inferstate 构建、gen_decode_params、page_table 拷贝、copy_for_cuda_graph
    输入拷贝、argmax、mem_indexes 旋转) 与多次 cudaGraphLaunch 的 cpu 开销。

    采样使用 flashinfer 的 top_k_top_p_sampling_from_logits, philox seed/offset 保存在
    显存 tensor 中并在图内自增, 保证 replay 之间随机性正确推进。温度/top_k/top_p 为逐行
    静态输入 buffer, 每个 step 刷新, 支持 batch 内异构采样参数。

    含惩罚项、logprobs、自定义随机种子、logit_bias 等请求走原有慢速路径 (can_run 判定)。
    """

    def __init__(self, backend: "ChunkedPrefillBackend"):
        self.backend = backend
        self.model = backend.model
        self.draft_model = backend.draft_models[0]
        self.mtp_step = backend.mtp_step
        self.mtp_size = self.mtp_step + 1
        self.args = get_env_start_args()

        model = self.model
        self.graph_max_batch_size = model.graph_max_batch_size
        self.graph_max_len_in_batch = model.graph_max_len_in_batch
        self.cuda_graph_batch_sizes = model.graph.cuda_graph_batch_sizes
        self.mempool = model.graph.mempool
        self.vocab_size = model.vocab_size

        B = self.graph_max_batch_size
        n_max = B // self.mtp_size
        self.req_manager = model.req_manager
        self.sampling_manager = model.req_manager.req_sampling_params_manager

        # step 级静态输入 buffer, 每个 step 由 staging 刷新, 图内核直接读取其地址。
        self.b_req_idx = torch.zeros(B, dtype=torch.int32, device="cuda")
        self.b_seq_len = torch.zeros(B, dtype=torch.int32, device="cuda")
        self.b_mtp_index = torch.arange(self.mtp_size, dtype=torch.int32, device="cuda").repeat(n_max)
        self.mem_indexes = torch.zeros(B, dtype=torch.int32, device="cuda")
        self.eagle_mem_indexes = torch.zeros(n_max * self.mtp_step, dtype=torch.int32, device="cuda")
        self.input_ids = torch.zeros(B, dtype=torch.int64, device="cuda")
        self.b_position_delta = torch.zeros(B, dtype=torch.int64, device="cuda")
        self._position_delta_dirty = False

        self.temperature = torch.ones(B, dtype=torch.float32, device="cuda")
        self.top_k = torch.ones(B, dtype=torch.int32, device="cuda")
        self.top_p = torch.ones(B, dtype=torch.float32, device="cuda")
        self.philox_seed = torch.tensor([random.getrandbits(63)], dtype=torch.int64, device="cuda")
        self.philox_offset = torch.tensor([0], dtype=torch.int64, device="cuda")

        # 对应的 pinned staging buffer, 每个 step 一次性 H2D。
        pin = dict(dtype=torch.int32, device="cpu", pin_memory=True)
        self.b_req_idx_pin = torch.zeros(B, **pin)
        self.b_seq_len_pin = torch.zeros(B, **pin)
        self.mem_indexes_pin = torch.zeros(B, **pin)
        self.eagle_mem_indexes_pin = torch.zeros(n_max * self.mtp_step, **pin)
        self.temperature_pin = torch.ones(B, dtype=torch.float32, device="cpu", pin_memory=True)
        self.top_k_pin = torch.ones(B, dtype=torch.int32, device="cpu", pin_memory=True)
        self.top_p_pin = torch.ones(B, dtype=torch.float32, device="cpu", pin_memory=True)
        self.b_position_delta_pin = torch.zeros(B, dtype=torch.int64, device="cpu", pin_memory=True)

        # 静态输出 buffer (采样 token 拷贝到此供 D2H / verify / draft chain 使用)。
        self.out_next_token_ids = torch.zeros(B, dtype=torch.int64, device="cuda")

        self.graphs = {}

        self.hold_req_idx = self.req_manager.HOLD_REQUEST_ID
        self.hold_mem_index = self.model.mem_manager.HOLD_TOKEN_MEMINDEX

        from flashinfer.sampling import top_k_top_p_sampling_from_logits

        self._fi_sampling = top_k_top_p_sampling_from_logits
        return

    def _find_graph_batch_size(self, batch_size: int) -> Optional[int]:
        index = bisect.bisect_left(self.cuda_graph_batch_sizes, batch_size)
        if index < len(self.cuda_graph_batch_sizes):
            return self.cuda_graph_batch_sizes[index]
        return None

    # ---------------- eligibility ----------------

    def can_run(self, decode_reqs: List[InferReq], max_kv_seq_len: int, batch_size: int) -> bool:
        if self._find_graph_batch_size(batch_size) is None:
            return False
        if max_kv_seq_len + self.mtp_step > self.graph_max_len_in_batch:
            return False
        if self.backend.decode_mask_func is not None:
            return False
        for req in decode_reqs:
            if req.mtp_step != self.mtp_step:
                return False
            shm_param = req.sampling_param.shm_param
            if shm_param.return_logprobs:
                return False
            if req.generator is not None:
                return False
            if len(req.sampling_param.invalid_token_ids) != 0:
                return False
            if req.need_out_token_id_statistics:
                # presence/frequency/repetition penalty 需要惩罚核
                return False
            if shm_param.exponential_decay_length_penalty.to_tuple()[1] != 1.0:
                return False
            if shm_param.min_new_tokens > 1:
                out_len = req.get_cur_total_len() - req.shm_req.input_len
                if out_len < shm_param.min_new_tokens - 1:
                    return False
        return True

    # ---------------- capture ----------------

    @torch.no_grad()
    def warmup(self):
        logger.info("Begin capture mtp fused decode cudagraph.")
        for batch_size in self.cuda_graph_batch_sizes[::-1]:
            self._stage_warmup_inputs(batch_size)
            # 先 eager 跑一遍触发 triton / flashinfer 的 JIT 与 autotune, 再捕获。
            torch.cuda.synchronize()
            verify_ctx = self._run_verify_body(batch_size)
            self._run_draft_body(batch_size, verify_ctx)
            torch.cuda.synchronize()

            verify_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(verify_graph, pool=self.mempool):
                verify_ctx = self._run_verify_body(batch_size)
            mtp_accept_len, accepted_index, main_hiddens = verify_ctx

            draft_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(draft_graph, pool=self.mempool):
                self._run_draft_body(batch_size, verify_ctx)

            self.graphs[batch_size] = _GraphBundle(
                verify_graph=verify_graph,
                draft_graph=draft_graph,
                mtp_accept_len=mtp_accept_len,
                accepted_index=accepted_index,
                main_hiddens=main_hiddens,
            )
            verify_graph.replay()
            draft_graph.replay()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        logger.info(f"Capture mtp fused decode cudagraph success, sizes: {self.cuda_graph_batch_sizes}")
        return

    def _stage_warmup_inputs(self, batch_size: int):
        n_real = batch_size // self.mtp_size
        self.b_req_idx[:batch_size].fill_(self.hold_req_idx)
        seq_pattern = torch.arange(2, self.mtp_size + 2, dtype=torch.int32, device="cuda")
        self.b_seq_len[:batch_size].copy_(seq_pattern.repeat(n_real))
        self.mem_indexes[:batch_size].fill_(self.hold_mem_index)
        self.eagle_mem_indexes[: n_real * self.mtp_step].fill_(self.hold_mem_index)
        self.temperature[:batch_size].fill_(1.0)
        self.top_k[:batch_size].fill_(1)
        self.top_p[:batch_size].fill_(1.0)
        self.b_position_delta[:batch_size].zero_()
        return

    def _build_model_input(self, batch_size: int) -> ModelInput:
        return ModelInput(
            batch_size=batch_size,
            total_token_num=self.graph_max_len_in_batch * batch_size,
            max_q_seq_len=1,
            max_kv_seq_len=self.graph_max_len_in_batch,
            input_ids=self.input_ids[:batch_size],
            mem_indexes=self.mem_indexes[:batch_size],
            b_req_idx=self.b_req_idx[:batch_size],
            b_seq_len=self.b_seq_len[:batch_size],
            b_mtp_index=self.b_mtp_index[:batch_size],
            is_prefill=False,
            multimodal_params=[{"images": [], "audios": []} for _ in range(batch_size)],
        )

    def _forward_in_body(self, model, model_input: ModelInput):
        infer_state = model._create_inferstate(model_input)
        infer_state.b_position_delta_gpu = self.b_position_delta[: model_input.batch_size]
        copy_kv_index_to_req(
            model.req_manager.req_to_token_indexs,
            infer_state.b_req_idx,
            infer_state.b_seq_len,
            infer_state.mem_index,
        )
        infer_state.init_some_extra_state(model)
        infer_state.init_att_state()
        return model._token_forward(infer_state)

    def _run_verify_body(self, batch_size: int):
        """主模型 forward + 采样 + mtp verify + accept/counter 更新, 输出 cpu 需要的信息。"""
        model = self.model
        n_real = batch_size // self.mtp_size
        b_req_idx = self.b_req_idx[:batch_size]

        req_to_next_token_ids = self.sampling_manager.req_to_next_token_ids
        gather_token(
            req_to_next_token_ids,
            b_req_idx,
            self.b_mtp_index[:batch_size],
            out=self.input_ids[:batch_size],
        )

        model_input = self._build_model_input(batch_size)
        model_output = self._forward_in_body(model, model_input)

        logits = model_output.logits
        logits.div_(self.temperature[:batch_size].view(-1, 1))
        sampled_ids = self._fi_sampling(
            logits,
            self.top_k[:batch_size],
            self.top_p[:batch_size],
            filter_apply_order="joint",
            deterministic=True,
            seed=self.philox_seed,
            offset=self.philox_offset,
            check_nan=False,
        )
        self.philox_offset += 4 * ((batch_size * self.vocab_size + 3) // 4)
        next_token_ids = self.out_next_token_ids[:batch_size]
        next_token_ids.copy_(sampled_ids)

        b_req_mtp_start_loc = torch.arange(n_real, dtype=torch.int32, device="cuda") * self.mtp_size
        mtp_accept_len, accepted_index = mtp_verify(
            req_to_next_token_ids=req_to_next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            new_next_token_ids=next_token_ids,
            b_req_idx=b_req_idx,
        )
        scatter_mtp_accept_len(
            self.req_manager.req_to_accept_len,
            b_req_mtp_start_loc,
            b_req_idx,
            mtp_accept_len,
        )
        self.sampling_manager.update_reqs_out_token_counter_gpu(
            b_req_idx=b_req_idx,
            next_token_ids=next_token_ids,
            mask=accepted_index == 1,
        )
        return mtp_accept_len, accepted_index, model_output.mtp_main_output_hiddens

    def _run_draft_body(self, batch_size: int, verify_ctx):
        """draft chain, 与 _draft_decode_eagle 相同的滑动窗口语义。"""
        mtp_accept_len, _, main_hiddens = verify_ctx
        mtp_size = self.mtp_size
        n_real = batch_size // mtp_size
        b_req_idx = self.b_req_idx[:batch_size]
        b_seq_len = self.b_seq_len[:batch_size]
        req_to_next_token_ids = self.sampling_manager.req_to_next_token_ids

        draft_model_input = self._build_model_input(batch_size)
        draft_mem_indexes = self.mem_indexes[:batch_size]
        draft_next_token_ids = self.out_next_token_ids[:batch_size]
        draft_hiddens = main_hiddens
        all_next_token_ids = [draft_next_token_ids]
        for _step in range(self.mtp_step):
            draft_model_input.input_ids = draft_next_token_ids
            draft_model_input.mem_indexes = draft_mem_indexes
            draft_model_input.mtp_draft_input_hiddens = draft_hiddens
            draft_model_output = self._forward_in_body(self.draft_model, draft_model_input)
            draft_next_token_ids = torch.argmax(draft_model_output.logits, dim=-1)
            draft_hiddens = draft_model_output.mtp_main_output_hiddens
            all_next_token_ids.append(draft_next_token_ids)

            b_seq_len += 1
            eagle_mem_indexes_i = self.eagle_mem_indexes[_step * n_real : (_step + 1) * n_real]
            draft_mem_indexes = torch.cat(
                [draft_mem_indexes.view(-1, mtp_size)[:, 1:], eagle_mem_indexes_i.view(-1, 1)],
                dim=1,
            ).view(-1)

        all_next_token_ids = torch.stack(all_next_token_ids, dim=1)
        b_req_mtp_start_loc = torch.arange(n_real, dtype=torch.int32, device="cuda") * mtp_size
        mtp_scatter_next_token_ids(
            req_to_next_token_ids=req_to_next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            all_next_token_ids=all_next_token_ids,
            b_req_idx=b_req_idx,
            mtp_accept_len=mtp_accept_len,
        )
        return

    # ---------------- replay ----------------

    def replay_verify(self, model_input: ModelInput, run_reqs: List[InferReq]) -> FusedStepOutput:
        """
        以 prepare_decode_inputs 的产物为输入, 完成 padding + staging + verify graph replay。
        必须在目标 cuda stream 上下文中调用; draft graph 由 replay_draft 单独发射。
        """
        real_batch_size = model_input.batch_size
        batch_size = self._find_graph_batch_size(real_batch_size)
        self._replay_batch_size = batch_size
        n_real = batch_size // self.mtp_size
        real_n = real_batch_size // self.mtp_size
        mtp_step = self.mtp_step

        # eagle 额外 kv 槽位分配 (真实请求) + HOLD padding
        if g_infer_context.radix_cache is not None:
            g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(real_n * mtp_step)
        eagle_mem_indexes_cpu = g_infer_context.req_manager.mem_manager.alloc(real_n * mtp_step)

        # staging: 填 pinned buffer, 单次 H2D 到静态输入
        self.b_req_idx_pin[:real_batch_size].copy_(model_input.b_req_idx)
        self.b_seq_len_pin[:real_batch_size].copy_(model_input.b_seq_len)
        self.mem_indexes_pin[:real_batch_size].copy_(model_input.mem_indexes_cpu)
        self.eagle_mem_indexes_pin[: real_n * mtp_step].copy_(eagle_mem_indexes_cpu)
        if batch_size != real_batch_size:
            self.b_req_idx_pin[real_batch_size:batch_size].fill_(self.hold_req_idx)
            pad_n = n_real - real_n
            pad_seq = torch.arange(2, self.mtp_size + 2, dtype=torch.int32).repeat(pad_n)
            self.b_seq_len_pin[real_batch_size:batch_size].copy_(pad_seq)
            self.mem_indexes_pin[real_batch_size:batch_size].fill_(self.hold_mem_index)
            self.eagle_mem_indexes_pin[real_n * mtp_step : n_real * mtp_step].fill_(self.hold_mem_index)

        has_delta = False
        for i, req in enumerate(run_reqs):
            shm_param = req.sampling_param.shm_param
            self.temperature_pin[i] = shm_param.temperature
            top_k = shm_param.top_k
            self.top_k_pin[i] = self.vocab_size if top_k <= 0 else top_k
            self.top_p_pin[i] = shm_param.top_p
            delta = 0
            for image in req.multimodal_params["images"]:
                delta += image["grid_thwd"][3]
            if delta != 0:
                has_delta = True
            self.b_position_delta_pin[i] = delta
        if batch_size != real_batch_size:
            self.temperature_pin[real_batch_size:batch_size].fill_(1.0)
            self.top_k_pin[real_batch_size:batch_size].fill_(1)
            self.top_p_pin[real_batch_size:batch_size].fill_(1.0)
            self.b_position_delta_pin[real_batch_size:batch_size].zero_()

        self.b_req_idx[:batch_size].copy_(self.b_req_idx_pin[:batch_size], non_blocking=True)
        self.b_seq_len[:batch_size].copy_(self.b_seq_len_pin[:batch_size], non_blocking=True)
        self.mem_indexes[:batch_size].copy_(self.mem_indexes_pin[:batch_size], non_blocking=True)
        self.eagle_mem_indexes[: n_real * mtp_step].copy_(
            self.eagle_mem_indexes_pin[: n_real * mtp_step], non_blocking=True
        )
        self.temperature[:batch_size].copy_(self.temperature_pin[:batch_size], non_blocking=True)
        self.top_k[:batch_size].copy_(self.top_k_pin[:batch_size], non_blocking=True)
        self.top_p[:batch_size].copy_(self.top_p_pin[:batch_size], non_blocking=True)
        if has_delta or self._position_delta_dirty:
            self.b_position_delta[:batch_size].copy_(self.b_position_delta_pin[:batch_size], non_blocking=True)
        self._position_delta_dirty = has_delta

        bundle: _GraphBundle = self.graphs[batch_size]
        bundle.verify_graph.replay()

        return FusedStepOutput(
            next_token_ids=self.out_next_token_ids[:real_batch_size],
            mtp_accept_len=bundle.mtp_accept_len[:real_n],
            accepted_index=bundle.accepted_index[:real_batch_size],
            eagle_mem_indexes_cpu=eagle_mem_indexes_cpu,
        )

    def replay_draft(self):
        bundle: _GraphBundle = self.graphs[self._replay_batch_size]
        bundle.draft_graph.replay()
        return
