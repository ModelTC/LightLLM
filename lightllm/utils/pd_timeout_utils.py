"""Timeout budgets shared by PD Master and GPU request handlers."""

import os


def get_pd_handshake_timeout_seconds(resource_wait_seconds: int, prefill: bool = False) -> int:
    # Decode may wait once for a shared request slot and again in the router.
    # Its own resource limit must fire before an outer handshake expires. The
    # inference wait loop polls every five seconds; leave time to forward its error.
    timeout = max(180, 2 * max(0, int(resource_wait_seconds)) + 15)
    # P starts waiting slightly earlier and must let Master cancel the request.
    return timeout + 15 if prefill else timeout


def get_pd_kv_transfer_timeout_seconds() -> int:
    timeout = int(os.getenv("LIGHTLLM_PD_KV_TRANSFER_TIMEOUT_SECONDS", "180"))
    if timeout <= 0:
        raise ValueError("LIGHTLLM_PD_KV_TRANSFER_TIMEOUT_SECONDS must be positive")
    return timeout
