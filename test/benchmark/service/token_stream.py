import codecs
from dataclasses import dataclass, field
import json
import math


class SSEDecoder:
    def __init__(self):
        self.decoder = codecs.getincrementaldecoder("utf-8")()
        self.buffer = ""
        self.data = []

    def feed(self, chunk: bytes) -> list[str]:
        self.buffer += self.decoder.decode(chunk)
        events = []
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            line = line.removesuffix("\r")
            if not line:
                if self.data:
                    events.append("\n".join(self.data))
                    self.data.clear()
            elif line.startswith("data:"):
                self.data.append(line[5:].removeprefix(" "))
        return events

    def finish(self):
        self.buffer += self.decoder.decode(b"", final=True)
        if self.buffer.strip() or self.data:
            raise ValueError("Truncated SSE event at EOF")


@dataclass
class TokenStream:
    framework: str
    prompt_ids: list[int]
    output_length: int
    started: float
    token_ids: list[int] = field(default_factory=list, init=False)
    arrivals: list[float] = field(default_factory=list, init=False)
    finished: bool = field(default=False, init=False)
    done: bool = field(default=False, init=False)
    events: int = field(default=0, init=False)
    last_received: float = field(default=0.0, init=False, repr=False)
    cached_input_tokens: int | None = field(default=None, init=False)

    def __post_init__(self):
        if self.framework not in ("lightllm", "vllm", "sglang") or self.output_length < 1:
            raise ValueError("Expected a supported framework and positive output length")
        if not self.prompt_ids or any(type(token) is not int or token < 0 for token in self.prompt_ids):
            raise ValueError("Prompt must contain exact integer token IDs")
        self.last_received = self.started
        if not math.isfinite(self.started):
            raise ValueError("Start timestamp must be finite")

    def record(self, payload: str, received: float):
        if not math.isfinite(received) or received < self.last_received:
            raise ValueError("Token arrival timestamps must be finite and monotonic")
        self.last_received = received
        if payload == "[DONE]":
            if not self.finished:
                raise ValueError("Stream ended before the terminal generation event")
            self.done = True
            return
        if self.done:
            raise ValueError("Received data after DONE")
        body = json.loads(payload)
        if body.get("error"):
            raise ValueError(f"Service stream error: {body['error']}")
        self.events += 1
        if self.framework == "lightllm":
            token = body["token"]
            if token["prompt_tokens"] != len(self.prompt_ids):
                raise ValueError("Stream prompt count differs from actual request IDs")
            if token["count_output_tokens"] != len(self.token_ids) + 1:
                raise ValueError("Native stream skipped or duplicated an output position")
            ids = [token["id"]]
            self._record_cache_count(token.get("prompt_cache_len"))
            reason = body.get("finish_reason") if body.get("finished") is True else None
        else:
            choices = body.get("choices", [])
            if len(choices) > 1 or (choices and choices[0].get("index") != 0):
                raise ValueError("Benchmark requires one completion choice")
            choice = choices[0] if choices else {}
            ids = choice.get("token_ids")
            if ids is None:
                ids = []
            if choice.get("text") and not ids:
                raise ValueError("Nonempty text event omitted explicit token IDs")
            prompt = choice.get("prompt_token_ids")
            if prompt is not None and prompt != self.prompt_ids:
                raise ValueError("Service changed the exact tokenized prompt")
            reason = choice.get("finish_reason")
        if not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("Stream must expose integer token IDs")
        if ids and self.finished:
            raise ValueError("Received output tokens after terminal generation")
        self.token_ids.extend(ids)
        self.arrivals.extend([received] * len(ids))
        if len(self.token_ids) > self.output_length:
            raise ValueError("Service exceeded the requested output length")
        if reason is not None:
            if reason != "length" or len(self.token_ids) != self.output_length:
                raise ValueError("Fixed-length stream did not finish at its requested token count")
            self.finished = True
        usage = body.get("usage")
        if usage is not None:
            self._record_cache_count((usage.get("prompt_tokens_details") or {}).get("cached_tokens"))
        if usage is not None and (
            usage["prompt_tokens"] != len(self.prompt_ids) or usage["completion_tokens"] != len(self.token_ids)
        ):
            raise ValueError("Stream usage does not match observed exact token counts")

    def _record_cache_count(self, count):
        if count is None:
            return
        if type(count) is not int or not 0 <= count <= len(self.prompt_ids):
            raise ValueError("Cached input count must be an integer within the exact prompt")
        if self.cached_input_tokens is not None and count != self.cached_input_tokens:
            raise ValueError("Cached input count changed within one request")
        self.cached_input_tokens = count

    def summarize(self, ended: float) -> dict:
        if not self.finished or len(self.token_ids) != self.output_length:
            raise ValueError("Incomplete generation cannot be included in benchmark results")
        if self.framework != "lightllm" and not self.done:
            raise ValueError("OpenAI-compatible stream omitted DONE")
        if not math.isfinite(ended) or ended < self.last_received:
            raise ValueError("Invalid request completion timestamp")
        gaps = [right - left for left, right in zip(self.arrivals, self.arrivals[1:])]
        return {
            "input_tokens": len(self.prompt_ids),
            "cached_input_tokens": self.cached_input_tokens,
            "uncached_input_tokens": (
                len(self.prompt_ids) - self.cached_input_tokens if self.cached_input_tokens is not None else None
            ),
            "output_tokens": len(self.token_ids),
            "events": self.events,
            "ttft_s": self.arrivals[0] - self.started,
            "tpot_s": sum(gaps) / len(gaps) if gaps else None,
            "e2e_s": ended - self.started,
            "receiver_itl_s": gaps,
            "itl_definition": "client arrival gaps; IDs delivered in one event share a timestamp",
        }


def request_stream(
    base_url,
    framework,
    model,
    prompt_ids,
    output_length,
    timeout,
    response_path,
    *,
    disable_prompt_cache=False,
    cache_salt=None,
):
    import time
    import requests

    if disable_prompt_cache and framework != "lightllm":
        raise ValueError("Per-request prompt-cache disabling is only implemented for native requests")
    if cache_salt is not None:
        if framework not in ("vllm", "sglang"):
            raise ValueError("Cache salts are only implemented for baseline requests")
        if (
            not isinstance(cache_salt, str)
            or not 1 <= len(cache_salt) <= 128
            or any(char in cache_salt for char in "@/\\\x00")
        ):
            raise ValueError("Cache salt must contain 1--128 characters without '@', '/', backslash or NUL")
    if framework == "lightllm":
        endpoint = "/generate_stream"
        payload = {
            "inputs": prompt_ids,
            "parameters": {
                "do_sample": False,
                "ignore_eos": True,
                "max_new_tokens": output_length,
                "return_details": True,
            },
        }
        if disable_prompt_cache:
            payload["parameters"]["disable_prompt_cache"] = True
    elif framework in ("vllm", "sglang"):
        endpoint = "/v1/completions"
        payload = {
            "model": model,
            "prompt": prompt_ids,
            "temperature": 0,
            "seed": 3107,
            "ignore_eos": True,
            "max_tokens": output_length,
            "stream": True,
            "stream_options": {"include_usage": True},
            "return_token_ids": True,
        }
        if cache_salt is not None:
            payload["cache_salt"] = cache_salt
    else:
        raise ValueError("Unsupported serving framework")
    raw = []
    decoder = SSEDecoder()
    try:
        with requests.Session() as client:
            client.trust_env = False
            started = time.perf_counter()
            stream = TokenStream(framework, prompt_ids, output_length, started)
            with client.post(base_url.rstrip("/") + endpoint, json=payload, stream=True, timeout=timeout) as response:
                if response.status_code != 200:
                    raw.append(response.content)
                    response.raise_for_status()
                if not response.headers.get("content-type", "").startswith("text/event-stream"):
                    raw.append(response.content)
                    raise ValueError("Expected an SSE response")
                if "chunked" not in response.headers.get("transfer-encoding", "").lower():
                    raise ValueError("Token timing requires HTTP/1.1 chunked transfer")
                for chunk in response.iter_content(chunk_size=None):
                    received = time.perf_counter()
                    raw.append(chunk)
                    for event in decoder.feed(chunk):
                        stream.record(event, received)
                ended = time.perf_counter()
                decoder.finish()
                return {
                    "token_ids": stream.token_ids,
                    "metrics": stream.summarize(ended),
                    "timeline": {
                        "started": started,
                        "first_token": stream.arrivals[0],
                        "last_token": stream.arrivals[-1],
                        "ended": ended,
                    },
                }
    finally:
        response_path.write_bytes(b"".join(raw))
