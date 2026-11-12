from typing import Optional


def get_pd_handshake_timeout_seconds(resource_wait_seconds: int, prefill: bool = False) -> Optional[int]:
    """Let node resource timeouts expire before their enclosing PD handshake."""
    if resource_wait_seconds < 0:
        return None
    # Decode can wait for a shared request slot, then independently in the router.
    # Allow its five-second polling loop and error forwarding to finish first.
    timeout = max(180, 2 * resource_wait_seconds + 15)
    # Prefill waits before Master sends to Decode; leave time for Master's abort.
    return timeout + 15 if prefill else timeout
