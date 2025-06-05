import asyncio
import collections
import time
import uvloop
import requests
import base64
import os
from io import BytesIO
import pickle

from .function_call_parser import TOOLS_TAG_LIST, FunctionCallParser
from .build_prompt import build_prompt, init_tokenizer

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import ujson as json
from http import HTTPStatus
import uuid
from PIL import Image
import multiprocessing as mp
from typing import AsyncGenerator, Union
from typing import Callable
from lightllm.server import TokenLoad
from fastapi import BackgroundTasks, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse, JSONResponse
from lightllm.server.core.objs.sampling_params import SamplingParams
from .multimodal_params import MultimodalParams
from .httpserver.manager import HttpServerManager
from .httpserver_for_pd_master.manager import HttpServerManagerForPDMaster
from .api_lightllm import lightllm_get_score
from lightllm.utils.envs_utils import get_env_start_args, get_lightllm_websocket_max_message_size

from lightllm.utils.log_utils import init_logger
from lightllm.server.metrics.manager import MetricClient
from lightllm.utils.envs_utils import get_unique_server_name
from dataclasses import dataclass

from .api_models import (
    ChatCompletionRequest,
    FunctionResponse,
    ToolCall,
    UsageInfo,
    ChatMessage,
    ChatCompletionResponseChoice,
    ChatCompletionResponse,
    DeltaMessage,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
)

logger = init_logger(__name__)


def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
    from .api_http import g_objs

    g_objs.metric_client.counter_inc("lightllm_request_failure")
    return JSONResponse({"message": message}, status_code=status_code.value)


async def chat_completions_impl(request: ChatCompletionRequest, raw_request: Request) -> Response:
    from .api_http import g_objs

    if request.logit_bias is not None:
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            "The logit_bias parameter is not currently supported",
        )

    if request.function_call != "none":
        return create_error_response(HTTPStatus.BAD_REQUEST, "The function call feature is not supported")

    created_time = int(time.time())

    multimodal_params_dict = {"images": []}
    for message in request.messages:
        if isinstance(message.content, list):
            texts = []
            for content in message.content:
                if content.type == "text" and content.text:
                    texts.append(content.text)
                elif content.type == "image_url" and content.image_url is not None:
                    img = content.image_url.url
                    if img.startswith("http://") or img.startswith("https://"):
                        multimodal_params_dict["images"].append({"type": "url", "data": img})
                    elif img.startswith("data:image"):
                        # "data:image/jpeg;base64,{base64_image}"
                        data_str = img.split(";", 1)[1]
                        if data_str.startswith("base64,"):
                            data = data_str[7:]
                            multimodal_params_dict["images"].append({"type": "base64", "data": data})
                        else:
                            raise ValueError("Unrecognized image input.")
                    else:
                        raise ValueError(
                            "Unrecognized image input. Supports local path, http url, base64, and PIL.Image."
                        )

            message.content = "\n".join(texts)

    tools = None
    if request.tools and request.tool_choice != "none":
        # request.skip_special_tokens = False
        if not isinstance(request.tool_choice, str):
            tools = [
                item.function.model_dump()
                for item in request.tools
                if item.function.name == request.tool_choice.function.name
            ]
        else:
            tools = [item.function.model_dump() for item in request.tools]

    prompt = await build_prompt(request, tools)
    sampling_params_dict = {
        "do_sample": True,
        "presence_penalty": request.presence_penalty,
        "frequency_penalty": request.frequency_penalty,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "ignore_eos": request.ignore_eos,
        "max_new_tokens": request.max_tokens,
        "stop_sequences": request.stop,
        "n": request.n,
        "best_of": request.n,
        "add_special_tokens": False,
    }
    if request.response_format:
        obj = request.response_format.get("schema")
        if obj:
            # guided_json takes str instead of dict obj
            sampling_params_dict["guided_json"] = json.dumps(obj)
    sampling_params = SamplingParams()
    sampling_params.init(tokenizer=g_objs.httpserver_manager.tokenizer, **sampling_params_dict)

    sampling_params.verify()
    multimodal_params = MultimodalParams(**multimodal_params_dict)

    results_generator = g_objs.httpserver_manager.generate(
        prompt, sampling_params, multimodal_params, request=raw_request
    )

    # Non-streaming case
    if not request.stream:
        final_output_dict = collections.defaultdict(list)
        count_output_tokens_dict = collections.defaultdict(lambda: 0)
        finish_reason_dict = {}
        prompt_tokens_dict = {}
        completion_tokens = 0
        async for sub_req_id, request_output, metadata, finish_status in results_generator:
            from .req_id_generator import convert_sub_id_to_group_id

            group_request_id = convert_sub_id_to_group_id(sub_req_id)
            count_output_tokens_dict[sub_req_id] += 1
            final_output_dict[sub_req_id].append(request_output)
            if finish_status.is_finished():
                finish_reason_dict[sub_req_id] = finish_status.get_finish_reason()
                prompt_tokens_dict[sub_req_id] = metadata["prompt_tokens"]
        choices = []
        sub_ids = list(final_output_dict.keys())[: request.n]
        for i in range(request.n):
            sub_req_id = sub_ids[i]
            prompt_tokens = prompt_tokens_dict[sub_req_id]
            completion_tokens = count_output_tokens_dict[sub_req_id]
            usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

            finish_reason = finish_reason_dict[sub_req_id]
            text = "".join(final_output_dict[sub_req_id])
            tool_calls = None
            tool_choice = request.tool_choice
            tools = request.tools

            if tool_choice != "none" and any([i in text for i in TOOLS_TAG_LIST]):
                if finish_reason == "stop":
                    finish_reason = "function_call"
                try:
                    parser = FunctionCallParser(tools, g_objs.args.tool_call_parser)
                    full_normal_text, call_info_list = parser.parse_non_stream(text)
                    tool_calls = [
                        ToolCall(
                            id=str(call_info.tool_index),
                            function=FunctionResponse(name=call_info.name, arguments=call_info.parameters),
                        )
                        for call_info in call_info_list
                    ]
                except Exception as e:
                    logger.error(f"Exception: {e}")
                    return create_error_response(
                        HTTPStatus.BAD_REQUEST,
                        "Failed to parse fc related info to json format!",
                    )

            chat_message = ChatMessage(role="assistant", content=text, tool_calls=tool_calls)
            choice = ChatCompletionResponseChoice(
                index=i,
                message=chat_message,
                finish_reason=finish_reason,
            )
            choices.append(choice)
        resp = ChatCompletionResponse(
            id=group_request_id, created=created_time, model=request.model, choices=choices, usage=usage
        )
        return resp

    if sampling_params.n != 1:
        raise Exception("stream api only support n = 1")

    parser_dict = {}

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        finish_reason = None
        from .req_id_generator import convert_sub_id_to_group_id

        async for sub_req_id, request_output, metadata, finish_status in results_generator:
            if request.tool_choice != "none" and request.tools:
                delta = request_output
                group_request_id = convert_sub_id_to_group_id(sub_req_id)
                index = metadata["id"]
                finish_reason = finish_status.get_finish_reason()

                if index not in parser_dict:
                    parser_dict[index] = FunctionCallParser(
                        tools=request.tools,
                        tool_call_parser=g_objs.args.tool_call_parser,
                    )
                parser = parser_dict[index]

                # parse_increment => returns (normal_text, calls)
                normal_text, calls = parser.parse_stream_chunk(delta)

                # 1) if there's normal_text, output it as normal content
                if normal_text:
                    choice_data = ChatCompletionStreamResponseChoice(
                        index=0,
                        delta=DeltaMessage(content=normal_text),
                        finish_reason=finish_reason if finish_reason else "",
                    )
                    chunk = ChatCompletionStreamResponse(
                        id=group_request_id,
                        created=created_time,
                        choices=[choice_data],
                        model=request.model,
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

                # 2) if we found calls, we output them as separate chunk(s)
                for call_item in calls:
                    # transform call_item -> FunctionResponse + ToolCall
                    if finish_reason == "stop":
                        latest_delta_len = 0
                        if isinstance(call_item.parameters, str):
                            latest_delta_len = len(call_item.parameters)

                        expected_call = json.dumps(
                            parser.multi_format_parser.detectors[0].prev_tool_call_arr[index].get("arguments", {}),
                            ensure_ascii=False,
                        )
                        actual_call = parser.multi_format_parser.detectors[0].streamed_args_for_tool[index]
                        if latest_delta_len > 0:
                            actual_call = actual_call[:-latest_delta_len]
                        remaining_call = expected_call.replace(actual_call, "", 1)
                        call_item.parameters = remaining_call

                    tool_call = ToolCall(
                        id=str(call_item.tool_index),
                        function=FunctionResponse(
                            name=call_item.name,
                            arguments=call_item.parameters,
                        ),
                    )
                    choice_data = ChatCompletionStreamResponseChoice(
                        index=0,
                        delta=DeltaMessage(role="assistant", tool_calls=[tool_call]),
                        finish_reason="function_call",
                    )
                    chunk = ChatCompletionStreamResponse(
                        id=group_request_id,
                        created=created_time,
                        choices=[choice_data],
                        model=request.model,
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
            else:
                group_request_id = convert_sub_id_to_group_id(sub_req_id)

                delta_message = DeltaMessage(role="assistant", content=request_output)
                if finish_status.is_finished():
                    finish_reason = finish_status.get_finish_reason()
                stream_choice = ChatCompletionStreamResponseChoice(
                    index=0, delta=delta_message, finish_reason=finish_reason
                )
                stream_resp = ChatCompletionStreamResponse(
                    id=group_request_id,
                    created=created_time,
                    model=request.model,
                    choices=[stream_choice],
                )
                yield ("data: " + json.dumps(stream_resp.dict(), ensure_ascii=False) + "\n\n").encode("utf-8")

    background_tasks = BackgroundTasks()
    return StreamingResponse(stream_results(), media_type="text/event-stream", background=background_tasks)
