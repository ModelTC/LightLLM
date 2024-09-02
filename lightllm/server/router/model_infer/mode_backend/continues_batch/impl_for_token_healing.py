import torch
from .impl import ContinuesBatchBackend
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.server.router.model_infer.infer_batch import InferBatch, InferReq, InferSamplingParams
from .pre_process import prepare_prefill_inputs, prepare_decode_inputs
from .post_process import sample
from lightllm.server.tokenizer import get_tokenizer


class TokenHealingBackend(ContinuesBatchBackend):
    def __init__(self) -> None:
        super().__init__()

    def init_custom(self):
        """
        初始化tokenizer 词表相关的的操作
        """
        self.tokenizer = get_tokenizer(
            self.args.model_dir, self.args.tokenizer_mode, trust_remote_code=self.args.trust_remote_code
        )
        vob_dict = self.tokenizer.get_vocab()
        self.token_to_token_id = vob_dict
        assert len(vob_dict) == self.model.vocab_size
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

    @calculate_time(show=False, min_cost_ms=300)
    def prefill_batch(self, batch_id):
        # 在 token_healing 的模式下，暂时不能启用 dynamic prompt cache
        assert self.radix_cache is None
        output_dict = {}
        batch: InferBatch = self.cache.pop(batch_id)
        kwargs, run_reqs = prepare_prefill_inputs(batch, self.radix_cache, self.model.mem_manager)

        logics = self.model.forward(**kwargs)

        # 对于不能满足前缀匹配的logic位置，将其logics设置为一个较大负值，将其概率掩盖为 0
        mask = torch.ones_like(logics, dtype=torch.bool)
        for i, run_obj in enumerate(run_reqs):
            assert not hasattr(run_obj, "prefix_str")

            prefix_token_str = "".join([self.token_id_to_token[e] for e in run_obj.prefix_token_ids])
            run_obj.prefix_str = prefix_token_str
            self._mask_not_prefix_token(i, run_obj, mask)

        logics[mask] = -1000000.0

        # 有prefix
        self._topk_repair(run_reqs)
        next_token_ids, next_token_probs = sample(logics, run_reqs, self.eos_id)
        self._topk_recover(run_reqs)

        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        for req_obj, next_token_id, next_token_logprob in zip(run_reqs, next_token_ids, next_token_logprobs):

            req_obj.cur_kv_len = len(req_obj.input_token_ids)
            req_obj.input_token_ids.append(next_token_id)
            req_obj.out_token_id_count[next_token_id] += 1
            req_obj.update_finish_status(self.eos_id)

            self._handle_req_ans(req_obj, next_token_id, next_token_logprob, output_dict)

        self.cache[batch.batch_id] = batch
        return output_dict

    @calculate_time(show=True, min_cost_ms=200)
    def decode_batch(self, batch_id):
        # 当前token headling 不支持 prompt cache
        assert self.radix_cache is None
        output_dict = {}
        batch: InferBatch = self.cache.pop(batch_id)
        kwargs, run_reqs = prepare_decode_inputs(batch, self.radix_cache)

        logits = self.model.forward(**kwargs)

        # 对 logits 添加 prefix 限制
        all_no_prefix = all([len(e.prefix_str) == 0 for e in run_reqs])
        if not all_no_prefix:
            mask = torch.ones_like(logits, dtype=torch.bool)
            for i, run_obj in enumerate(run_reqs):
                self._mask_decode_not_prefix_token(i, run_obj, mask)

            logits[mask] = -1000000.0
        # self._topk_repair(run_reqs)
        next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
        # self._topk_recover(run_reqs)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        for req_obj, next_token_id, next_token_logprob in zip(run_reqs, next_token_ids, next_token_logprobs):
            # prefill and decode is same
            req_obj: InferReq = req_obj
            req_obj.cur_kv_len = len(req_obj.input_token_ids)
            req_obj.input_token_ids.append(next_token_id)
            req_obj.out_token_id_count[next_token_id] += 1
            req_obj.update_finish_status(self.eos_id)

            self._handle_req_ans(req_obj, next_token_id, next_token_logprob, output_dict)

        self.cache[batch.batch_id] = batch
        return output_dict

    def _handle_req_ans(self, req_obj, next_token_id, next_token_logprob, output_dict):
        if len(req_obj.prefix_str) != 0:
            next_token = self.token_id_to_token[int(next_token_id)]
            if req_obj.prefix_str.startswith(next_token):
                output_dict[req_obj.r_id] = (
                    req_obj.req_status,
                    req_obj.cur_kv_len,
                    req_obj.get_output_len(),
                    [],
                    req_obj.finish_status.value,
                    None,
                )
                req_obj.prefix_str = req_obj.prefix_str[len(next_token) :]
            elif next_token.startswith(req_obj.prefix_str):
                metadata = {
                    "id": int(next_token_id),
                    "logprob": float(next_token_logprob),
                    "prefix_str": req_obj.prefix_str,
                }
                output_dict[req_obj.r_id] = (
                    req_obj.req_status,
                    req_obj.cur_kv_len,
                    req_obj.get_output_len(),
                    [(int(next_token_id), metadata)],
                    req_obj.finish_status.value,
                    None,
                )
                req_obj.prefix_str = ""
            else:
                assert False, "error state"
        else:
            metadata = {
                "id": int(next_token_id),
                "logprob": float(next_token_logprob),
            }
            output_dict[req_obj.r_id] = (
                req_obj.req_status,
                req_obj.cur_kv_len,
                req_obj.get_output_len(),
                [(int(next_token_id), metadata)],
                req_obj.finish_status.value,
                None,
            )
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

    def _topk_repair(self, run_reqs: list[InferReq]):
        for req_obj in run_reqs:
            if len(req_obj.prefix_str) != 0:
                req_obj.origin_topk = req_obj.sampling_param.top_k
                req_obj.sampling_param.top_k = 1
            else:
                req_obj.origin_topk = req_obj.sampling_param.top_k
        return

    def _topk_recover(self, run_reqs: list[InferReq]):
        for req_obj in run_reqs:
            req_obj.sampling_param.top_k = req_obj.origin_topk
        return
