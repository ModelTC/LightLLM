from dataclasses import dataclass
from typing import Optional

import torch

from lightllm.common.basemodel.batch_objs import ModelOutput
from lightllm.utils.tensor_utils import tensor_to_no_ref_tensor


@dataclass
class DSparkModelOutput(ModelOutput):
    confidence_logits: Optional[torch.Tensor] = None
    draft_token_ids: Optional[torch.Tensor] = None

    def to_no_ref_tensor(self):
        super().to_no_ref_tensor()
        if self.confidence_logits is not None:
            self.confidence_logits = tensor_to_no_ref_tensor(self.confidence_logits)
        if self.draft_token_ids is not None:
            self.draft_token_ids = tensor_to_no_ref_tensor(self.draft_token_ids)
