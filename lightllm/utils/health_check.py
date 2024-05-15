import base64
import numpy as np
from lightllm.server.sampling_params import SamplingParams
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.server.httpserver.manager import HttpServerManager
from fastapi.responses import Response
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


async def health_check(httpserver_manager: HttpServerManager, g_id_gen, request):
    try:
        request_dict = {"inputs": "你好！", "parameters": {"do_sample": True, "temperature": 0.8, "max_new_tokens": 2}}

        request_dict = await request.json()
        prompt = request_dict.pop("inputs")
        sample_params_dict = request_dict["parameters"]
        sampling_params = SamplingParams(**sample_params_dict)
        sampling_params.verify()
        multimodal_params_dict = request_dict.get("multimodal_params", {})
        multimodal_params = MultimodalParams(**multimodal_params_dict)

        group_request_id = g_id_gen.generate_id()
        results_generator = httpserver_manager.generate(
            prompt, sampling_params, group_request_id, multimodal_params, request=request
        )
        async for _, _, _, _ in results_generator:
            pass
        return True
    except Exception as e:
        logger.error("health_check error:", e)
        return False
