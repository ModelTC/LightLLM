import torch
from lightllm.common.basemodel import InferStateInfo
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
            self.max_seq_len = self.max_kv_seq_len
            self._build_b_image_token_end()
        return

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
