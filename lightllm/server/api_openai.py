import asyncio
import collections
import time
import uvloop
import requests
import base64
import os
from io import BytesIO
import pickle
import uuid

from .function_call_parser import TOOLS_TAG_LIST, FunctionCallParser
from .build_prompt import build_prompt, init_tokenizer

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import ujson as json
from http import HTTPStatus
from PIL import Image
import multiprocessing as mp
from typing import AsyncGenerator, Union, List, Dict
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
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    CompletionLogprobs,
    CompletionStreamResponse,
    CompletionStreamChoice,
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
        "repetition_penalty": request.repetition_penalty,
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
                    # 为 tool_call_parser 提供默认值
                    tool_parser = getattr(g_objs.args, "tool_call_parser", None) or "llama3"
                    parser = FunctionCallParser(tools, tool_parser)
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
        return create_error_response(HTTPStatus.BAD_REQUEST, "stream api only support n = 1")

    parser_dict = {}

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        finish_reason = None
        from .req_id_generator import convert_sub_id_to_group_id

        prompt_tokens = 0
        completion_tokens = 0
        async for sub_req_id, request_output, metadata, finish_status in results_generator:
            prompt_tokens = metadata["prompt_tokens"]
            completion_tokens += 1
            if request.tool_choice != "none" and request.tools:
                delta = request_output
                group_request_id = convert_sub_id_to_group_id(sub_req_id)
                index = metadata["id"]
                finish_reason = finish_status.get_finish_reason()

                if index not in parser_dict:
                    # 为 tool_call_parser 提供默认值
                    tool_parser = getattr(g_objs.args, "tool_call_parser", None) or "llama3"
                    parser_dict[index] = FunctionCallParser(
                        tools=request.tools,
                        tool_call_parser=tool_parser,
                    )
                parser = parser_dict[index]

                # parse_increment => returns (normal_text, calls)
                normal_text, calls = parser.parse_stream_chunk(delta)

                # 1) if there's normal_text, output it as normal content
                if normal_text:
                    choice_data = ChatCompletionStreamResponseChoice(
                        index=0,
                        delta=DeltaMessage(content=normal_text),
                        finish_reason=finish_reason if finish_reason else None,
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
                # Additional usage chunk

        if request.stream_options and request.stream_options.include_usage:
            usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
            usage_chunk = ChatCompletionStreamResponse(
                id=group_request_id,
                created=created_time,
                choices=[],  # Empty choices array as per OpenAI spec
                model=request.model,
                usage=usage,
            )
            yield f"data: {usage_chunk.model_dump_json()}\n\n"

    background_tasks = BackgroundTasks()
    return StreamingResponse(stream_results(), media_type="text/event-stream", background=background_tasks)


async def completions_impl(request: CompletionRequest, raw_request: Request) -> Response:
    from .api_http import g_objs

    if request.logit_bias is not None:
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            "The logit_bias parameter is not currently supported",
        )

    created_time = int(time.time())

    # Parse and normalize prompts
    prompts = []
    if isinstance(request.prompt, list):
        if len(request.prompt) == 0:
            return create_error_response(
                HTTPStatus.BAD_REQUEST,
                "Prompt cannot be empty",
            )

        # Check if it's a list of integers (token IDs)
        if isinstance(request.prompt[0], int):
            prompts.append(request.prompt)
        elif isinstance(request.prompt[0], list):
            for token_list in request.prompt:
                prompts.append(token_list)
        else:
            # List of strings
            prompts = request.prompt
    else:
        # Single string
        prompts = [request.prompt]

    # Handle suffix for completion mode
    if request.suffix:
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            "The suffix parameter is not currently supported",
        )

    # Prepare sampling parameters - same as g_generate_stream_func pattern
    sampling_params_dict = {
        "do_sample": request.do_sample,
        "presence_penalty": request.presence_penalty,
        "frequency_penalty": request.frequency_penalty,
        "repetition_penalty": request.repetition_penalty,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "ignore_eos": request.ignore_eos,
        "max_new_tokens": request.max_tokens,
        "stop_sequences": request.stop,
        "n": request.n,
        "best_of": request.best_of,
        "add_special_tokens": False,
    }

    sampling_params = SamplingParams()
    sampling_params.init(tokenizer=g_objs.httpserver_manager.tokenizer, **sampling_params_dict)
    sampling_params.verify()

    # v1/completions does not support multimodal inputs, so we use an empty MultimodalParams
    multimodal_params = MultimodalParams()

    return await _process_prompts_completion(
        prompts, sampling_params, sampling_params_dict, multimodal_params, raw_request, request, created_time
    )


async def _process_prompts_completion(
    prompts: Union[List[str], List[List[int]]],
    sampling_params: SamplingParams,
    sampling_params_dict: Dict,
    multimodal_params: MultimodalParams,
    raw_request: Request,
    request: CompletionRequest,
    created_time: int,
) -> Response:
    from .api_http import g_objs
    import asyncio

    if request.stream:
        if len(prompts) > 1:
            return create_error_response(
                HTTPStatus.BAD_REQUEST,
                "Streaming is not supported for batch requests",
            )

        if sampling_params.n != 1:
            return create_error_response(HTTPStatus.BAD_REQUEST, "stream api only support n = 1")

        return await _handle_streaming_completion(
            prompts[0], sampling_params, multimodal_params, raw_request, request, created_time
        )

    async def process_single_prompt(prompt: Union[str, List[int]], prompt_index: int):
        if len(prompts) > 1:
            individual_sampling_params = SamplingParams()
            individual_sampling_params.init(tokenizer=g_objs.httpserver_manager.tokenizer, **sampling_params_dict)
            individual_sampling_params.verify()
        else:
            individual_sampling_params = sampling_params

        # Convert token array to string for _collect_generation_results
        prompt_str = prompt
        if isinstance(prompt, list):
            prompt_str = g_objs.httpserver_manager.tokenizer.decode(prompt, skip_special_tokens=False)

        generator = g_objs.httpserver_manager.generate(
            prompt, individual_sampling_params, multimodal_params, request=raw_request
        )

        return await _collect_generation_results(
            generator, request, prompt_str, prompt_index, individual_sampling_params
        )

    tasks = [asyncio.create_task(process_single_prompt(prompt, i)) for i, prompt in enumerate(prompts)]

    results = await asyncio.gather(*tasks)
    return _build_completion_response(results, request, created_time, len(prompts) > 1)


async def _handle_streaming_completion(
    prompt: Union[str, List[int]],
    sampling_params: SamplingParams,
    multimodal_params: MultimodalParams,
    raw_request: Request,
    request: CompletionRequest,
    created_time: int,
) -> Response:
    from .api_http import g_objs

    results_generator = g_objs.httpserver_manager.generate(
        prompt, sampling_params, multimodal_params, request=raw_request
    )

    async def stream_results() -> AsyncGenerator[bytes, None]:
        from .req_id_generator import convert_sub_id_to_group_id

        prompt_tokens = 0
        completion_tokens = 0

        async for sub_req_id, request_output, metadata, finish_status in results_generator:
            group_request_id = convert_sub_id_to_group_id(sub_req_id)
            prompt_tokens = metadata["prompt_tokens"]
            completion_tokens += 1
            current_finish_reason = None
            if finish_status.is_finished():
                current_finish_reason = finish_status.get_finish_reason()

            output_text = request_output
            if request.echo and metadata.get("is_first_token", False):
                prompt_str = prompt
                if isinstance(prompt, list):
                    prompt_str = g_objs.httpserver_manager.tokenizer.decode(prompt, skip_special_tokens=False)
                output_text = prompt_str + output_text

            stream_choice = CompletionStreamChoice(
                index=0,
                text=output_text,
                finish_reason=current_finish_reason,
                logprobs=None if request.logprobs is None else {},
            )
            stream_resp = CompletionStreamResponse(
                id=group_request_id,
                created=created_time,
                model=request.model,
                choices=[stream_choice],
            )
            yield ("data: " + json.dumps(stream_resp.dict(), ensure_ascii=False) + "\n\n").encode("utf-8")

        yield "data: [DONE]\n\n".encode("utf-8")

        if request.stream_options and request.stream_options.include_usage:
            usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
            usage_chunk = CompletionStreamResponse(
                id=group_request_id,
                created=created_time,
                choices=[],  # Empty choices array as per OpenAI spec
                model=request.model,
                usage=usage,
            )
            yield f"data: {usage_chunk.model_dump_json()}\n\n"

    background_tasks = BackgroundTasks()
    return StreamingResponse(stream_results(), media_type="text/event-stream", background=background_tasks)


async def _collect_generation_results(
    generator, request: CompletionRequest, prompt: str, prompt_index: int, sampling_params: SamplingParams
):
    final_output = []
    count_output_tokens = 0
    finish_reason = None
    prompt_tokens = 0
    token_infos = [] if request.logprobs is not None else None
    prompt_logprobs = None
    prompt_token_ids = None
    is_first_metadata = True

    async for sub_req_id, request_output, metadata, finish_status in generator:
        if is_first_metadata:
            prompt_logprobs = metadata.get("prompt_logprobs", None)
            prompt_token_ids = metadata.get("prompt_token_ids", None)
            is_first_metadata = False

        count_output_tokens += 1
        final_output.append(request_output)

        if request.logprobs is not None and token_infos is not None:
            token_info = {
                "text": request_output,
                "logprob": metadata.get("logprob", None),
                "id": metadata.get("id", None),
            }
            token_infos.append(token_info)

        if finish_status.is_finished():
            finish_reason = finish_status.get_finish_reason()
            prompt_tokens = metadata["prompt_tokens"]

    # 处理停止序列剔除
    final_text = "".join(final_output)
    if finish_reason == "stop" and sampling_params.stop_sequences.size > 0:
        valid_stop_strings = sampling_params.stop_sequences.to_strings()
        for stop_str in valid_stop_strings:
            stop_index = final_text.rfind(stop_str, max(0, len(final_text) - len(stop_str) - 20), len(final_text))
            if stop_index != -1:
                logger.debug(f"removed stop sequence in tail: '{final_text[stop_index:]}'")
                final_text = final_text[:stop_index]
                break

    return {
        "index": prompt_index,
        "text": final_text,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": count_output_tokens,
        "token_infos": token_infos,
        "prompt_logprobs": prompt_logprobs,
        "prompt_token_ids": prompt_token_ids,
        "prompt_text": prompt,
    }


def _build_completion_response(results: List[Dict], request: CompletionRequest, created_time: int, is_batch: bool):
    from .api_http import g_objs

    choices = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for result in results:
        text = result["text"]
        if request.echo:
            text = result["prompt_text"] + text

        logprobs_data = _build_logprobs_data(result, request, g_objs.httpserver_manager.tokenizer)

        choice = CompletionChoice(
            index=result["index"],
            text=text,
            finish_reason=result["finish_reason"],
            logprobs=CompletionLogprobs(**logprobs_data) if logprobs_data else None,
        )
        choices.append(choice)

        total_prompt_tokens += result["prompt_tokens"]
        total_completion_tokens += result["completion_tokens"]

    usage = UsageInfo(
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=total_prompt_tokens + total_completion_tokens,
    )

    if is_batch:
        group_request_id = f"cmpl-batch-{uuid.uuid4().hex[:8]}"
    else:
        group_request_id = f"cmpl-{uuid.uuid4().hex[:8]}"

    return CompletionResponse(
        id=group_request_id, created=created_time, model=request.model, choices=choices, usage=usage
    )


def _build_logprobs_data(result: Dict, request: CompletionRequest, tokenizer) -> Dict:
    if request.logprobs is None:
        return None

    all_tokens = []
    all_token_logprobs = []
    all_text_offsets = []
    offset = 0

    def add_tokens_to_logprobs(token_ids=None, token_infos=None, logprob_map=None):
        nonlocal offset

        def add_single_token(token_text: str, logprob: float):
            nonlocal offset
            all_tokens.append(token_text)
            all_token_logprobs.append(logprob)
            all_text_offsets.append(offset)
            offset += len(token_text)

        if token_ids is not None:
            for token_id in token_ids:
                token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                logprob = logprob_map.get(token_id, None) if logprob_map else None
                add_single_token(token_text, logprob)
        elif token_infos is not None:
            for token_info in token_infos:
                add_single_token(token_info["text"], token_info["logprob"])

    # 处理 echo 模式下的 prompt tokens
    if request.echo and result.get("prompt_logprobs") is not None:
        prompt_logprobs = result["prompt_logprobs"]
        prompt_token_ids = result.get("prompt_token_ids")

        # 创建 token_id 到 logprob 的映射
        logprob_map = {}
        for current_token_id, logprobs_dict in prompt_logprobs:
            for next_token_id, logprob in logprobs_dict.items():
                logprob_map[int(next_token_id)] = logprob

        # 处理所有 prompt tokens
        if prompt_token_ids is not None:
            add_tokens_to_logprobs(token_ids=prompt_token_ids, logprob_map=logprob_map)

    elif request.echo:
        # echo=True 但没有 prompt logprobs
        prompt_token_ids = result.get("prompt_token_ids")
        if prompt_token_ids is not None:
            add_tokens_to_logprobs(token_ids=prompt_token_ids)
        else:
            # 回退：重新 tokenize prompt
            prompt_tokens = tokenizer.encode(result["prompt_text"], add_special_tokens=False)
            add_tokens_to_logprobs(token_ids=prompt_tokens)

    # 添加生成的 tokens 和 logprobs
    if result.get("token_infos"):
        add_tokens_to_logprobs(token_infos=result["token_infos"])

    top_logprobs_list = []
    for i, (token, logprob) in enumerate(zip(all_tokens, all_token_logprobs)):
        if logprob is not None:
            # TODO: 标准实现需要从后端获取top-k个logprobs数据
            # 目前后端不支持，只能获取所选token的logprobs
            top_logprobs_list.append({token: logprob})
        else:
            top_logprobs_list.append(None)

    return {
        "tokens": all_tokens,
        "token_logprobs": all_token_logprobs,
        "top_logprobs": top_logprobs_list,
        "text_offset": all_text_offsets,
    }
