from typing import AsyncGenerator
from fastapi import BackgroundTasks, Request
from fastapi.responses import Response, StreamingResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from .sampling_params import SamplingParams
from .multimodal_params import MultimodalParams
import json


def format_tgi_params(params):
    """
    tgi params format -> lightllm server params format
    pub(crate) struct GenerateParameters {
        pub best_of: Option<usize>,
        pub temperature: Option<f32>,
        pub repetition_penalty: Option<f32>,
        pub frequency_penalty: Option<f32>,
        pub presence_penalty: Option<f32>,
        pub top_k: Option<i32>,
        pub top_p: Option<f32>,
        pub typical_p: Option<f32>,
        pub do_sample: bool,
        pub max_new_tokens: u32,
        pub return_full_text: Option<bool>,
        pub stop: Vec<String>,
        pub truncate: Option<usize>,
        pub watermark: bool,
        pub details: bool,
        pub decoder_input_details: bool,
        pub seed: Option<u64>,
    }
    """
    # same keys: temperature, repetition_penalty, frequency_penalty, presence_penalty, top_k, top_p, do_sample, max_new_tokens
    # keys re-map
    if "return_details" not in params:
        params["return_details"] = params.pop("details", False)
    if "stop_sequences" not in params:
        params["stop_sequences"] = params.pop("stop", None)
    # remove keys lightllm not used
    params.pop("best_of", 1)
    params.pop("typical_p", 0.0)
    params.pop("return_full_text", False)
    params.pop("stop", None)
    params.pop("truncate", None)
    params.pop("watermark", False)
    params.pop("details", False)
    params.pop("decoder_input_details", False)
    params.pop("seed", 0)
    return params


async def tgi_generate_impl(request: Request, g_id_gen, httpserver_manager) -> Response:
    request_dict = await request.json()
    prompt = request_dict.pop("inputs")
    sample_params_dict = format_tgi_params(request_dict["parameters"])
    return_details = sample_params_dict.pop("return_details", False)
    sampling_params = SamplingParams(**sample_params_dict)
    sampling_params.verify()
    multimodal_params_dict = request_dict.get("multimodal_params", {})
    multimodal_params = MultimodalParams(**multimodal_params_dict)

    request_id = g_id_gen.generate_id()
    results_generator = httpserver_manager.generate(prompt, sampling_params, request_id, multimodal_params)

    # Non-streaming case
    final_output = []
    count_output_tokens = 0
    tokens = []
    prompt_logprobs = None
    prompt_token_ids = None
    is_first_metadata = True
    async for request_output, metadata, finish_status in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await httpserver_manager.abort(request_id)
            return Response(status_code=499)

        # when set "--return_all_prompt_logprobs", the first token metadata will contains
        # prompt_logprobs and prompt_token_ids
        if is_first_metadata:
            prompt_logprobs = metadata.get("prompt_logprobs", None)
            prompt_token_ids = metadata.get("prompt_token_ids", None)
            if prompt_logprobs is not None:
                del metadata["prompt_logprobs"]
            if prompt_token_ids is not None:
                del metadata["prompt_token_ids"]
            is_first_metadata = False

        count_output_tokens += 1
        final_output.append(request_output)
        if return_details:
            metadata["text"] = request_output
            tokens.append(metadata)

    assert final_output is not None
    ret = {
        "generated_text": "".join(final_output),
        "count_output_tokens": count_output_tokens,
        "finish_reason": finish_status.get_finish_reason(),
    }
    if return_details:
        ret["details"] = {
            "tokens": tokens,
            "generated_tokens": count_output_tokens,
            "finish_reason": finish_status.get_finish_reason(),
        }
    if prompt_token_ids is not None:
        ret["prompt_token_ids"] = prompt_token_ids
    if prompt_logprobs is not None:
        ret["prompt_logprobs"] = prompt_logprobs
    # wrap generation inside a Vec to match api-inference
    ret = [ret]
    json_compatible_item_data = jsonable_encoder(ret)
    return JSONResponse(content=json_compatible_item_data)


async def tgi_generate_stream_impl(request: Request, g_id_gen, httpserver_manager) -> Response:
    request_dict = await request.json()
    prompt = request_dict.pop("inputs")
    sample_params_dict = format_tgi_params(request_dict["parameters"])
    return_details = sample_params_dict.pop("return_details", False)
    sampling_params = SamplingParams(**sample_params_dict)
    sampling_params.verify()
    multimodal_params_dict = request_dict.get("multimodal_params", {})
    multimodal_params = MultimodalParams(**multimodal_params_dict)

    request_id = g_id_gen.generate_id()
    results_generator = httpserver_manager.generate(prompt, sampling_params, request_id, multimodal_params)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        final_output = []
        async for request_output, metadata, finish_status in results_generator:
            ret = {
                "token": {
                    "id": metadata.get("id", None),
                    "text": request_output,
                    "logprob": metadata.get("logprob", None),
                    "special": metadata.get("special", False),
                    "count_output_tokens": metadata.get("count_output_tokens", 0),
                },
                "generated_text": None,
                "finished": finish_status.is_finished(),
                "finish_reason": finish_status.get_finish_reason(),
                "details": None,
            }
            final_output.append(request_output)
            if ret["finished"]:
                ret["generated_text"] = "".join(final_output)
                if return_details:
                    ret["details"] = {
                        "generated_tokens": len(final_output),
                        "finish_reason": finish_status.get_finish_reason(),
                    }

            yield ("data:" + json.dumps(ret, ensure_ascii=False) + "\n\n").encode("utf-8")

    async def abort_request() -> None:
        await httpserver_manager.abort(request_id)

    background_tasks = BackgroundTasks()
    # Abort the request if the client disconnects.
    background_tasks.add_task(abort_request)

    return StreamingResponse(stream_results(), media_type="text/event-stream", background=background_tasks)
