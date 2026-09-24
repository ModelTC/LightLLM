"""Custom streaming response behavior for errors raised before the first chunk.

Starlette's ``StreamingResponse`` sends ``http.response.start`` before it
starts iterating over the response body. This is normally desirable because
the client receives the HTTP status and headers immediately. However, the
generation body used by LightLLM is an async generator, so request validation
and admission do not run until that generator is iterated. Consequently, an
error may be raised during the first iteration, after Starlette has already
sent HTTP 200.

An HTTP status cannot be changed after ``http.response.start`` has been sent.
This module therefore delays that event until the body has produced its first
chunk. If request setup fails before then, no response has started and
FastAPI's exception handler can still return the appropriate HTTP error. Once
the first chunk exists, the response starts normally.

The ``--disable_delay_response_start`` startup option restores Starlette's
original behavior for clients and proxies that require response status and
headers immediately.
"""

from fastapi.responses import StreamingResponse
from starlette.types import Send

from lightllm.utils.envs_utils import get_env_start_args


class CustomStreamingResponse(StreamingResponse):
    """Send the HTTP status only after the first body chunk is ready.

    The first chunk is sent directly after the response headers; it is not
    discarded or replayed through another iterator. Empty streams still send
    the configured status followed by an empty final body.

    This mechanism only covers failures raised before the first chunk. After
    the response has started, HTTP no longer allows changing its status code;
    later generation failures must be reported in the stream body (for
    example, as an SSE error event) or by closing the connection.
    """

    async def stream_response(self, send: Send) -> None:
        # Some clients and proxies require headers before first-token work is
        # complete. Let deployments opt out of the delayed response start.
        if get_env_start_args().disable_delay_response_start:
            await super().stream_response(send)
            return

        async def send_chunk(chunk):
            if not isinstance(chunk, (bytes, memoryview)):
                chunk = chunk.encode(self.charset)
            await send({"type": "http.response.body", "body": chunk, "more_body": True})

        async def send_response_start():
            # Read status and headers at send time. The first body iteration
            # may update them while performing admission or request setup.
            await send(
                {
                    "type": "http.response.start",
                    "status": self.status_code,
                    "headers": self.raw_headers,
                }
            )

        # Iterating here starts the otherwise-lazy generation pipeline. A
        # An error raised before the first yield escapes without an
        # http.response.start event, allowing FastAPI to handle it.
        async for first_chunk in self.body_iterator:
            await send_response_start()
            await send_chunk(first_chunk)
            break
        else:
            # An empty iterator is still a valid response and needs headers.
            await send_response_start()

        # The first chunk has already been sent; continue from the same
        # iterator without restarting or duplicating it.
        async for chunk in self.body_iterator:
            await send_chunk(chunk)

        await send({"type": "http.response.body", "body": b"", "more_body": False})
