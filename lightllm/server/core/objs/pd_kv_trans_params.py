import os
import ctypes
from typing import Optional

LIGHTLLM_PD_KV_TRANS_PARAM_OBJ_MAX_BYTES = int(os.getenv("LIGHTLLM_PD_KV_TRANS_PARAM_OBJ_MAX_BYTES", 8 * 1024))


class PDKVTransParamObj(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("data", ctypes.c_ubyte * LIGHTLLM_PD_KV_TRANS_PARAM_OBJ_MAX_BYTES),
        ("data_len", ctypes.c_int),
    ]

    def __init__(self):
        self.data_len = 0

    def set(self, obj_bytes: Optional[bytes]):
        if obj_bytes is None:
            self.data_len = 0
            return

        assert len(obj_bytes) <= LIGHTLLM_PD_KV_TRANS_PARAM_OBJ_MAX_BYTES, (
            f"PD_KV_TRANS_PARAM_OBJ bytes len {len(obj_bytes)} exceeds length of "
            f"{LIGHTLLM_PD_KV_TRANS_PARAM_OBJ_MAX_BYTES} bytes."
        )
        ctypes.memmove(self.data, obj_bytes, len(obj_bytes))
        self.data_len = len(obj_bytes)
        return

    def get(self) -> bytes:
        return bytes(self.data[: self.data_len])
