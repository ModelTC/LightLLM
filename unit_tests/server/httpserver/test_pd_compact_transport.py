import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from lightllm.server.httpserver.pd_loop import _pd_process_generate
from lightllm.server.pd_io_struct import (
    PD_COMPACT_TOKEN_INFO_LEN,
    build_pd_compact_token_info,
    unpack_pd_compact_token_info,
)


def _compact_metadata(token_id):
    return {
        "id": token_id,
        "count_output_tokens": 1,
        "prompt_tokens": 4,
        "prompt_cache_len": 2,
        "mtp_accepted_token_num": 0,
        "mtp_verify_token_num": 0,
        "mtp_verify_step_num": 0,
    }


def test_pd_compact_token_roundtrip_preserves_token_id():
    finish_status = SimpleNamespace(status=1)

    for token_id in (42, None):
        metadata = _compact_metadata(token_id)
        compact = build_pd_compact_token_info(123, "token", metadata, finish_status)

        assert len(compact) == PD_COMPACT_TOKEN_INFO_LEN
        assert unpack_pd_compact_token_info(compact) == (123, "token", metadata, finish_status.status)


def test_pd_transport_without_logprobs_keeps_token_id_in_compact_packet():
    metadata = {
        **_compact_metadata(42),
        "logprob": -0.1,
        "cumlogprob": -0.1,
        "special": False,
        "logprobs": {42: -0.1},
    }
    finish_status = SimpleNamespace(status=1)

    class _Manager:
        args = SimpleNamespace(run_mode="decode")
        cancel_pd_request_registration = MagicMock()

        async def generate(self, **_kwargs):
            yield 123, "token", metadata, finish_status

    async def run():
        forwarding_queue = AsyncMock()
        await _pd_process_generate(
            manager=_Manager(),
            prompt="prompt",
            sampling_params=SimpleNamespace(group_request_id=123, return_output_logprobs=False),
            multimodal_params=MagicMock(),
            forwarding_queue=forwarding_queue,
            pd_upload_websocket=AsyncMock(),
            pd_event=asyncio.Event(),
        )
        return forwarding_queue.put.await_args.args[0]

    compact = asyncio.run(run())
    assert len(compact) == PD_COMPACT_TOKEN_INFO_LEN

    _, _, unpacked_metadata, _ = unpack_pd_compact_token_info(compact)
    assert unpacked_metadata["id"] == 42
    assert "logprob" not in unpacked_metadata
    assert "cumlogprob" not in unpacked_metadata
    assert "special" not in unpacked_metadata
    assert "logprobs" not in unpacked_metadata
