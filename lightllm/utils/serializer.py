# copied from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/utils/common.py

import base64
import pickle
import io
from dataclasses import dataclass
from multiprocessing.reduction import ForkingPickler
from typing import List


class MultiprocessingSerializer:
    @staticmethod
    def serialize(obj, output_str: bool = False):
        """
        Serialize a Python object using ForkingPickler.

        Args:
            obj: The object to serialize.
            output_str (bool): If True, return a base64-encoded string instead of raw bytes.

        Returns:
            bytes or str: The serialized object.
        """
        buf = io.BytesIO()
        ForkingPickler(buf).dump(obj)
        buf.seek(0)
        output = buf.read()

        if output_str:
            # Convert bytes to base64-encoded string
            output = base64.b64encode(output).decode("utf-8")

        return output

    @staticmethod
    def deserialize(data):
        """
        Deserialize a previously serialized object.

        Args:
            data (bytes or str): The serialized data, optionally base64-encoded.

        Returns:
            The deserialized Python object.
        """
        if isinstance(data, str):
            # Decode base64 string to bytes
            data = base64.b64decode(data, validate=True)

        return SafeUnpickler(io.BytesIO(data)).load()


class SafeUnpickler(pickle.Unpickler):
    ALLOWED_MODULE_PREFIXES = {
        # --- Python types ---
        "builtins.",
        "collections.",
        "copyreg.",
        "functools.",
        "itertools.",
        "operator.",
        "types.",
        "weakref.",
        # --- PyTorch types ---
        "torch.",
        "torch._tensor.",
        "torch.storage.",
        "torch.nn.parameter.",
        "torch.autograd.function.",
        # --- torch distributed ---
        "torch.distributed.",
        "torch.distributed._shard.",
        "torch.distributed._composable.",
        "torch._C._distributed_c10d.",
        "torch._C._distributed_fsdp.",
        "torch.distributed.optim.",
        # --- multiprocessing ---
        "multiprocessing.resource_sharer.",
        "multiprocessing.reduction.",
        "pickletools.",
        # --- PEFT / LoRA ---
        "peft.",
        "transformers.",
        "huggingface_hub.",
        # --- SGLang & Unitest ---
        "sglang.srt.weight_sync.tensor_bucket.",
        "sglang.srt.model_executor.model_runner.",
        "sglang.srt.layers.",
        "sglang.srt.utils.",
        # --- LightLLM ---
        "lightllm.utils.",
    }

    DENY_CLASSES = {
        ("builtins", "eval"),
        ("builtins", "exec"),
        ("builtins", "compile"),
        ("os", "system"),
        ("subprocess", "Popen"),
        ("subprocess", "run"),
        ("codecs", "decode"),
        ("types", "CodeType"),
        ("types", "FunctionType"),
    }

    def find_class(self, module, name):
        # Block deterministic attacks
        if (module, name) in self.DENY_CLASSES:
            raise RuntimeError(
                f"Blocked unsafe class loading ({module}.{name}), " f"to prevent exploitation of CVE-2025-10164"
            )
        # Allowlist of safe-to-load modules.
        if any((module + ".").startswith(prefix) for prefix in self.ALLOWED_MODULE_PREFIXES):
            return super().find_class(module, name)

        # Block everything else. (Potential attack surface)
        raise RuntimeError(
            f"Blocked unsafe class loading ({module}.{name}), " f"to prevent exploitation of CVE-2025-10164"
        )


@dataclass
class LocalSerializedTensor:
    """torch.Tensor that gets serialized by MultiprocessingSerializer
    (which only serializes a pointer and not the data).
    The i-th element in the list corresponds to i-th rank's GPU."""

    values: List[bytes]

    def get(self, rank: int):
        return MultiprocessingSerializer.deserialize(self.values[rank])
