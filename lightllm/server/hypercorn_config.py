"""Default Hypercorn settings for LightLLM HTTP servers."""

from lightllm.utils.envs_utils import get_lightllm_websocket_max_message_size

keep_alive_timeout = 10
websocket_max_message_size = get_lightllm_websocket_max_message_size()
