import torch
from .impl import ContinuesBatchBackend
from lightllm.server.router.model_infer.infer_batch import InferBatch, InferReq, InferSamplingParams, requests_mapping
from .pre_process import prepare_prefill_inputs, prepare_decode_inputs
from lightllm.server.io_struct import ReqRunStatus, FinishStatus


class RewardModelBackend(ContinuesBatchBackend):
    def __init__(self) -> None:
        super().__init__()

    def decode_batch(self, batch_id):
        """
        This function should not be called.
        """
        pass

    def forward(self, batch_id, is_prefill):

        output_dict = {}
        batch: InferBatch = self.cache.pop(batch_id)

        kwargs, run_reqs = prepare_prefill_inputs(batch, self.radix_cache, self.is_multimodal)

        scores: torch.Tensor = self.model.forward(**kwargs)
        scores = scores.detach().cpu().numpy()

        next_token_id = 1
        next_token_logprob = 1.0

        for req_obj, score in zip(run_reqs, scores):
            # prefill and decode is same
            req_obj: InferReq = req_obj
            req_obj.cur_kv_len = len(req_obj.input_token_ids)
            req_obj.input_token_ids.append(next_token_id)
            req_obj.out_token_id_count[next_token_id] += 1
            req_obj.finish_status = FinishStatus.FINISHED_STOP

            metadata = {"id": int(next_token_id), "logprob": float(next_token_logprob), "score": float(score[0])}

            output_dict[req_obj.r_id] = (
                req_obj.req_status,
                req_obj.cur_kv_len,
                req_obj.get_output_len(),
                [(int(next_token_id), metadata)],
                req_obj.finish_status.value,  # 转化为整数，避免传送大对象,
                None,
            )  # 请求状态， 当前占用的kv的长度， 当前输出token的数量， 输出的token的id和元信息列表， 是否推理结束的状态， 额外保留参数

        self.cache[batch.batch_id] = batch
        return output_dict
