import torch
from .impl import ChunkedPrefillBackend
from typing import List
from lightllm.server.router.model_infer.mode_backend.continues_batch.impl import ContinuesBatchBackend
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq
from lightllm.server.tokenizer import get_tokenizer
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class TokenHealingBackend(ChunkedPrefillBackend):
    def __init__(self) -> None:
        super().__init__()
        self.prefill_mask_func = self._prefill_mask_callback
        self.decode_mask_func = self._decode_mask_callback
        self.extra_post_req_handle_func = self._update_tokenhealing_req_prefix_str

    def init_custom(self):
        """
        初始化tokenizer 词表相关的的操作
        """
        self.tokenizer = get_tokenizer(
            self.args.model_dir, self.args.tokenizer_mode, trust_remote_code=self.args.trust_remote_code
        )
        vob_dict = self.tokenizer.get_vocab()
        self.token_to_token_id = vob_dict
        if len(vob_dict) == self.model.vocab_size:
            logger.warning(f"tokenizer error: {len(vob_dict)} != {self.model.vocab_size}")
        self.max_token_str_len = max([len(key) for key in vob_dict.keys()])
        self.logger.info(f"max vob token str len: {self.max_token_str_len}")
        self.pad_token_str = "\U0010FFFF" * self.max_token_str_len
        from sortedcontainers import SortedList

        self.token_id_to_token = {token_id: token for token, token_id in vob_dict.items()}
        self.sorted_tokens = SortedList(
            [(token_str, token_id) for token_str, token_id in vob_dict.items()], key=lambda x: x[0]
        )
        self.token_indexes = torch.tensor([e[1] for e in self.sorted_tokens], dtype=torch.int64, device="cuda")
        return

    def _decode_mask_callback(self, run_reqs: List[InferReq], logits: torch.Tensor):
        self._init_prefix_infos(run_reqs=run_reqs)

        all_no_prefix = all([len(e.prefix_str) == 0 for e in run_reqs])
        if not all_no_prefix:
            mask = torch.ones_like(logits, dtype=torch.bool)
            for i, run_obj in enumerate(run_reqs):
                self._mask_decode_not_prefix_token(i, run_obj, mask)

            logits[mask] = -1000000.0
        return

    def _prefill_mask_callback(self, run_reqs: List[InferReq], logits: torch.Tensor):
        # 对于不能满足前缀匹配的logic位置，将其logics设置为一个较大负值，将其概率掩盖为 0
        self._init_prefix_infos(run_reqs=run_reqs)
        mask = torch.ones_like(logits, dtype=torch.bool)
        for i, run_obj in enumerate(run_reqs):
            self._mask_not_prefix_token(i, run_obj, mask)
        logits[mask] = -1000000.0
        return

    def _update_tokenhealing_req_prefix_str(self, req_obj: InferReq, next_token_id, next_token_logprob):
        next_token = self.token_id_to_token[int(next_token_id)]

        if len(req_obj.prefix_str) != 0:
            if req_obj.prefix_str.startswith(next_token):
                req_obj.prefix_str = req_obj.prefix_str[len(next_token) :]
            elif next_token.startswith(req_obj.prefix_str):
                req_obj.prefix_str = ""
            else:
                assert False, "dead path"
        return

    def _mask_not_prefix_token(self, i, run_obj: InferReq, mask):
        if len(run_obj.prefix_str) != 0:
            ok_token_id_list = []
            for index in range(1, len(run_obj.prefix_str) + 1):
                sub_str = run_obj.prefix_str[0:index]
                if sub_str in self.token_to_token_id:
                    ok_token_id_list.append(self.token_to_token_id[sub_str])

            mask[i, ok_token_id_list] = False

            start_token_str = run_obj.prefix_str
            end_token_str = start_token_str + self.pad_token_str
            start_index = self.sorted_tokens.bisect_left((start_token_str, None))
            end_index = self.sorted_tokens.bisect_right((end_token_str, None))
            mask[i, self.token_indexes[start_index:end_index]] = False
        else:
            mask[i, :] = False
        return

    def _mask_decode_not_prefix_token(self, i, run_obj: InferReq, mask):
        if len(run_obj.prefix_str) != 0:
            start_token_str = run_obj.prefix_str
            end_token_str = start_token_str + self.pad_token_str
            start_index = self.sorted_tokens.bisect_left((start_token_str, None))
            end_index = self.sorted_tokens.bisect_right((end_token_str, None))
            mask[i, self.token_indexes[start_index:end_index]] = False

            if (end_index - start_index) <= 0:
                ok_token_id_list = []
                for index in range(1, len(run_obj.prefix_str) + 1):
                    sub_str = run_obj.prefix_str[0:index]
                    if sub_str in self.token_to_token_id:
                        ok_token_id_list.append(self.token_to_token_id[sub_str])

                mask[i, ok_token_id_list] = False
        else:
            mask[i, :] = False
        return

    def _init_prefix_infos(self, run_reqs: List[InferReq]):
        for i, run_obj in enumerate(run_reqs):
            if not hasattr(run_obj, "prefix_str"):
                run_obj: InferReq = run_obj
                prefix_token_str = "".join([self.token_id_to_token[e] for e in run_obj.prefix_token_ids])
                run_obj.prefix_str = prefix_token_str
        return
