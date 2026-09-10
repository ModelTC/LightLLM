import copy
from types import SimpleNamespace

import torch
from lightllm.common.basemodel import InferStateInfo
from lightllm.common.basemodel.triton_kernel.copy_kv_index_to_req import copy_kv_index_to_req
from lightllm.common.infer_utils import init_req_to_token_indexes
from lightllm.models.gemma4.triton_kernel.build_b_image_token_end import build_b_image_token_end


class Gemma4InferStateInfo(InferStateInfo):
    def __init__(self):
        super().__init__()
        # Gemma-4 uses two RoPE frequency tables (one per layer type):
        # * sliding_attention layers: theta=10000, full rotation over head_dim=256
        # * full_attention layers:    theta=1_000_000, partial rotation (first 25% of head_dim=512)
        self.position_cos_sliding = None
        self.position_sin_sliding = None
        self.position_cos_full = None
        self.position_sin_full = None
        # b_image_token_end 用于标记每个 token 在 att 计算时，可以看到的对应的最大长度位置，用于
        # 对于文本token 和 image token 是区别对待的，
        # 文本token 对应的位置一定是 0， image token 对应的位置，是该token能看到的最远image token位置。
        # 相当于 image token 部分是双向 att，text token 还是 causal att。
        # 对应一个请求 token list 为 [t, t, i, i, t] 的一个token序列，
        # 则对应的 b_image_token_end 为 [0, 0, 4, 4, 0],
        # image token 可以看到自己当前这个token以及后面的 image token。
        self.b_image_token_end = None
        self.has_image_tokens = False
        self.sliding_window_mem_index = None
        self.sliding_window_mem_index_cpu = None
        self.sliding_requests = None

    def init_some_extra_state(self, model):
        super().init_some_extra_state(model)
        position_ids = self.position_ids
        self.position_cos_sliding = torch.index_select(model._cos_cached_sliding, 0, position_ids).view(
            position_ids.shape[0], -1
        )
        self.position_sin_sliding = torch.index_select(model._sin_cached_sliding, 0, position_ids).view(
            position_ids.shape[0], -1
        )
        self.position_cos_full = torch.index_select(model._cos_cached_full, 0, position_ids).view(
            position_ids.shape[0], -1
        )
        self.position_sin_full = torch.index_select(model._sin_cached_full, 0, position_ids).view(
            position_ids.shape[0], -1
        )
        if self.is_prefill:
            self._build_b_image_token_end()
        sliding_mem_manager = self.req_manager.sliding_mem_manager
        index_chunks = []
        token_num = 0
        for req_idx, _, q_len in self.sliding_requests:
            if req_idx != self.req_manager.HOLD_REQUEST_ID:
                index_chunks.extend(self.req_manager.alloc_sliding_window_indexes(req_idx, q_len))
            else:
                index_chunks.append(
                    torch.full((q_len,), sliding_mem_manager.HOLD_TOKEN_MEMINDEX, dtype=torch.int32, device="cpu")
                )
            token_num += q_len
        padding_token_num = self.input_ids.shape[0] - token_num
        if padding_token_num > 0:
            index_chunks.append(
                torch.full(
                    (padding_token_num,), sliding_mem_manager.HOLD_TOKEN_MEMINDEX, dtype=torch.int32, device="cpu"
                )
            )
        # Combine request-window and allocator views into owned pinned storage for asynchronous H2D.
        self.sliding_window_mem_index_cpu = torch.empty(
            (self.input_ids.shape[0],), dtype=torch.int32, device="cpu", pin_memory=True
        )
        if index_chunks:
            torch.cat(index_chunks, out=self.sliding_window_mem_index_cpu)
        self.sliding_window_mem_index = self.sliding_window_mem_index_cpu.cuda(non_blocking=True)
        if self.is_prefill:
            init_req_to_token_indexes(
                self.req_manager.req_to_sliding_window,
                self.b_req_idx,
                self.b_seq_len,
                self.b_ready_cache_len,
                self.b_q_start_loc,
                self.sliding_window_mem_index,
                self.max_q_seq_len,
            )
        else:
            copy_kv_index_to_req(
                self.req_manager.req_to_sliding_window,
                self.b_req_idx,
                self.b_seq_len,
                self.sliding_window_mem_index,
            )
        return

    def init_att_state(self):
        # Share batch tensors, but bind sliding attention to its own token table.
        sliding_state = copy.copy(self)
        sliding_state.req_manager = SimpleNamespace(req_to_token_indexs=self.req_manager.req_to_sliding_window)
        att_state = self.prefill_att_state1 if self.is_prefill else self.decode_att_state1
        att_state.infer_state = sliding_state
        super().init_att_state()

    def finish_forward(self):
        # Keep the latest W token slots after every KV-sharing reader has finished.
        start = 0
        for req_idx, seq_len, q_len in self.sliding_requests:
            if req_idx != self.req_manager.HOLD_REQUEST_ID:
                self.req_manager.update_sliding_window(
                    req_idx, seq_len, self.sliding_window_mem_index_cpu[start : start + q_len]
                )
            start += q_len

    def _build_b_image_token_end(self):
        device = self.position_ids.device
        self.b_image_token_end = torch.zeros(self.position_ids.shape[0], dtype=torch.int32, device=device)

        if not self.multimodal_params:
            return

        b_image_start_idx = []
        b_image_len = []
        b_image_nums = []
        b_image_start_num = []
        image_start_num = 0
        for params in self.multimodal_params:
            b_image_start_num.append(image_start_num)
            images = params.get("images", [])
            b_image_nums.append(len(images))
            for img in images:
                b_image_start_idx.append(img["start_idx"])
                b_image_len.append(img["token_num"])
                image_start_num += 1

        if image_start_num == 0:
            return

        self.has_image_tokens = True
        build_b_image_token_end(
            b_image_start_idx=torch.tensor(b_image_start_idx, dtype=torch.int32).cuda(non_blocking=True),
            b_image_len=torch.tensor(b_image_len, dtype=torch.int32).cuda(non_blocking=True),
            b_image_nums=torch.tensor(b_image_nums, dtype=torch.int32).cuda(non_blocking=True),
            b_image_start_num=torch.tensor(b_image_start_num, dtype=torch.int32).cuda(non_blocking=True),
            b_q_start_loc=self.b_q_start_loc,
            b_ready_cache_len=self.b_ready_cache_len,
            b_q_seq_len=self.b_q_seq_len,
            b_image_token_end=self.b_image_token_end,
        )
