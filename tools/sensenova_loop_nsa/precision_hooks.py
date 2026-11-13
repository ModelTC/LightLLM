"""Full-vocabulary capture outside graph replay, for private A/B snapshots only."""

import json
import os
from pathlib import Path

import torch


def install(model_class):
    original_forward = model_class.forward
    capture = Path(os.environ["DSA_PRECISION_DIR"])

    def forward(self, *args, **kwargs):
        output = original_forward(self, *args, **kwargs)
        arm = capture / "armed.json"
        if not arm.exists() or not self.dsa:
            return output
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return output
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Precision capture must stay outside CUDA graph capture")
        case = json.loads(arm.read_text())
        model_input = args[0] if args else kwargs["model_input"]
        self._bsh_capture_counter = getattr(self, "_bsh_capture_counter", 0) + 1
        record = {
            "case": case["id"],
            "is_prefill": model_input.is_prefill,
            "kv_length": model_input.max_kv_seq_len,
            "input_ids": model_input.input_ids.detach().cpu(),
            "logits": output.logits.detach().cpu(),
        }
        torch.save(record, capture / f'{case["id"]}_{os.getpid()}_{self._bsh_capture_counter}.pt')
        return output

    model_class.forward = forward
