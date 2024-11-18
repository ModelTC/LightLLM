"""
p d 分离模式下, 对于到达的请求，需要将输入的prompt_ids 中的最后一个id，提前处理，然后移入到outputs中
这是 p d 分离模式下，decode 节点的特殊处理点。
"""
from ..io_struct import ReqDetokenizationState
from .decode import decode_token

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def decode_mode_fix(req_out: ReqDetokenizationState, tokenizer, eos_id):
    new_token_id = req_out.prompt_ids[-1]
    req_out.prompt_ids = req_out.prompt_ids[0:-1]
    req_out.output_ids.append(new_token_id)

    out_text = decode_token(
        tokenizer,
        req_out,
        new_token_id,
        eos_id,
    )

    if out_text.endswith("\ufffd"):
        pass
    else:
        req_out.output_str = out_text

    return req_out
