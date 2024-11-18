import base64
import numpy as np
from lightllm.server.sampling_params import SamplingParams
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.server.httpserver.manager import HttpServerManager
from fastapi import Request
from lightllm.server.req_id_generator import ReqIDGenerator
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


_g_health_req_id_gen = ReqIDGenerator()
_g_health_req_id_gen.generate_id()


async def health_check(args, httpserver_manager: HttpServerManager, request: Request):
    try:
        request_dict = {"inputs": "你好！", "parameters": {"do_sample": True, "temperature": 0.8, "max_new_tokens": 2}}
        prompt = request_dict.pop("inputs")
        sample_params_dict = request_dict["parameters"]
        sampling_params = SamplingParams(**sample_params_dict)
        sampling_params.verify()
        if args.run_mode in ["prefill", "decode"]:
            sampling_params.group_request_id = -_g_health_req_id_gen.generate_id()  # health monitor 的 id 是负的
        multimodal_params_dict = request_dict.get("multimodal_params", {})
        multimodal_params = MultimodalParams(**multimodal_params_dict)

        results_generator = httpserver_manager.generate(prompt, sampling_params, multimodal_params, request)
        async for _, _, _, _ in results_generator:
            pass
        return True
    except Exception as e:
        logger.exception(str(e))
        return False
