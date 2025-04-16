"""
p d 分离模式下, 对于到达的请求，需要将输入的prompt_ids 中的最后一个id，提前处理，然后移入到outputs中
这是 p d 分离模式下，decode 节点的特殊处理点。
"""
from .decode_req import DecodeReq
from .decode import decode_token

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def decode_mode_fix(req_out: DecodeReq, tokenizer, eos_id):
    new_token_id = req_out.prompt_ids[-1]
    decode_token(tokenizer, req_out, new_token_id, eos_id)
    return req_out
