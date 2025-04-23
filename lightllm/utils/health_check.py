import os
import time
import asyncio
import numpy as np
from dataclasses import dataclass
from lightllm.server.core.objs import SamplingParams
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.server.httpserver.manager import HttpServerManager
from lightllm.server.router.dynamic_prompt.shared_arr import SharedInt
from fastapi import Request
from lightllm.server.req_id_generator import ReqIDGenerator
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_unique_server_name, get_env_start_args

logger = init_logger(__name__)


_g_health_req_id_gen = ReqIDGenerator()


@dataclass
class HealthObj:
    _is_health: bool = False
    _is_health_checking: bool = False
    _failure_count: int = 0
    _failure_threshold: int = int(os.getenv("HEALTH_FAILURE_THRESHOLD", 3))
    timeout: int = int(os.getenv("HEALTH_TIMEOUT", 100))
    dynamic_timeout: int = int(os.getenv("HEALTH_TIMEOUT", 100))
    latest_success_infer_time_mark = SharedInt(f"{get_unique_server_name()}_latest_success_infer_time_mark")

    def begin_check(self):
        self._is_health_checking = True

    def end_check(self):
        self._is_health_checking = False

    def set_unhealth(self):
        self._failure_count += 1
        self.dynamic_timeout += self.timeout
        if self._failure_count > self._failure_threshold:
            self._is_health = False

    def set_health(self):
        self._is_health = True
        self._failure_count = 0
        self.dynamic_timeout = self.timeout

    def is_health(self):
        return self._is_health

    def is_checking(self):
        return self._is_health_checking

    def has_latest_inference(self):
        last_timemark = self.latest_success_infer_time_mark.get_value()
        time_diff = time.time() - last_timemark
        return time_diff < self.timeout


health_obj = HealthObj()


async def health_check(args, httpserver_manager: HttpServerManager, request: Request):
    if health_obj.is_checking():
        return health_obj.is_health()

    if health_obj.is_health() and health_obj.has_latest_inference():
        return health_obj.is_health()

    health_obj.begin_check()
    try:
        request_dict = {"inputs": "你好！", "parameters": {"do_sample": True, "temperature": 0.8, "max_new_tokens": 2}}
        if args.run_mode in ["prefill", "nixl_prefill"]:
            request_dict["parameters"]["max_new_tokens"] = 1
        prompt = request_dict.pop("inputs")
        sample_params_dict = request_dict["parameters"]
        sampling_params = SamplingParams()
        sampling_params.init(tokenizer=httpserver_manager.tokenizer, **sample_params_dict)
        sampling_params.verify()

        if get_env_start_args().run_mode == "pd_master":
            # Since the id assigned by pd master needs to be passed to prefill and decode nodes for inference,
            # a normal request id is required instead of a negative id.
            sampling_params.group_request_id = _g_health_req_id_gen.generate_id()
        else:
            sampling_params.group_request_id = -_g_health_req_id_gen.generate_id()  # health monitor 的 id 是负的
        multimodal_params_dict = request_dict.get("multimodal_params", {})
        multimodal_params = MultimodalParams(**multimodal_params_dict)
        results_generator = httpserver_manager.generate(
            prompt, sampling_params, multimodal_params, request, is_health_req=True
        )

        async def check_timeout(results_generator):
            async for _, _, _, _ in results_generator:
                pass

        try:
            await asyncio.wait_for(check_timeout(results_generator), timeout=health_obj.dynamic_timeout)
            health_obj.set_health()
        except asyncio.TimeoutError:
            health_obj.set_unhealth()
            logger.warning(f"Health check timeout! The failure count is: {str(health_obj._failure_count)}")
        return health_obj.is_health()
    except Exception as e:
        logger.exception(str(e))
        health_obj.set_unhealth()
        return health_obj.is_health()
    finally:
        health_obj.end_check()
