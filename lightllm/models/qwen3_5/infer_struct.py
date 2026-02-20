"""
Qwen3.5 Multimodal Inference State

This module provides inference state for Qwen3.5 multimodal model that combines:
- Qwen3Next features (output gating, MTP-aware batching, hybrid attention buffer management)
- Qwen3VL multimodal support (mrope position encoding for images/videos)
"""

import torch
from typing import List

from lightllm.models.qwen2_vl.infer_struct import Qwen2VLInferStateInfo
from lightllm.utils.envs_utils import get_env_start_args


class Qwen35InferStateInfo(Qwen2VLInferStateInfo):
    """
    Inference state for Qwen3.5 multimodal model with:
    - gate_value attribute for output gating in full attention layers
    - MTP-aware batching for multi-token prediction
    - Custom buffer management for hybrid attention (full + linear)
    - mrope position encoding support for multimodal inputs
    """

    def __init__(self):
        super().__init__()
        # For output gating in full attention layers (from Qwen3Next)
        self.gate_value = None
        # MTP-aware attributes (from Qwen3Next)
        self.b_att_seq_len = None
        self.att_batch_size = None
        self.real_req_idx = None
        self.mtp_buffer_idx_list = None
        self.b_buffer_idx = None

    def _compute_mrope_delta(self, images: List) -> int:
        """Compute the position delta for mrope based on image tokens.

        The position delta is the sum of all image position deltas (grid_thwd[3])
        which accounts for the extra position IDs consumed by multimodal content.
        """
        position_delta = 0
        for image in images:
            position_delta += image["grid_thwd"][3]
        return position_delta

    def init_some_extra_state(self, model):
        """Initialize Qwen3.5-specific state including mrope and MTP support"""
        # First, initialize mrope position encoding using parent class
        # which now has the corrected delta computation
        rope_scaling = model.config.get("rope_scaling", {})
        self.rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))

        # Call the grandparent's (LlamaInferStateInfo) init_some_extra_state first
        # to set up basic state
        from lightllm.common.basemodel.infer_struct import InferStateInfo

        InferStateInfo.init_some_extra_state(self, model)

        # Now handle mrope position encoding with corrected delta computation
        if self.is_prefill:
            self.position_ids = self.get_mrope_position(self.multimodal_params)
        else:
            # Decode phase: compute correct mrope delta
            b_position_delta = [0 for _ in range(self.b_seq_len.shape[0])]
            for batch_idx, p in enumerate(self.multimodal_params):
                b_position_delta[batch_idx] = self._compute_mrope_delta(p.get("images", []))

            position_ids = self.position_ids + torch.tensor(b_position_delta, device=self.position_ids.device)
            self.position_ids = position_ids.unsqueeze(0).expand(3, -1)

        self.position_ids = self.position_ids.contiguous()
        self.position_cos = model._cos_cached[self.position_ids]
        self.position_sin = model._sin_cached[self.position_ids]

        # Now handle MTP-aware batching (from Qwen3Next)
        args_mtp_step = get_env_start_args().mtp_step
        mtp_size = args_mtp_step + 1

        if self.is_prefill:
            # Prefill: Standard initialization
            self.b_att_seq_len = self.b_seq_len
            self.b_buffer_idx = model.req_manager.req_to_buffer_index[self.b_req_idx, 0].contiguous()
        else:
            # Decode: MTP-aware handling
            # In MTP mode, each request has (mtp_step + 1) tokens
            # att_batch_size is the number of unique requests
            self.att_batch_size = self.batch_size // mtp_size

            # Use only the sequence lengths for the last token of each MTP group
            if args_mtp_step > 0:
                self.b_att_seq_len = self.b_seq_len[args_mtp_step::mtp_size].contiguous()
                self.real_req_idx = self.b_req_idx[args_mtp_step::mtp_size]
            else:
                self.b_att_seq_len = self.b_seq_len
                self.real_req_idx = self.b_req_idx

            # Buffer indices for Mamba cache (conv and SSM states)
            self.b_buffer_idx = model.req_manager.req_to_buffer_index[self.real_req_idx, :].flatten().contiguous()

            # Create per-step buffer indices for MTP
            if args_mtp_step > 0:
                buffer_idx_list = []
                for step_id in range(mtp_size):
                    buffer_idx_list.append(self.b_buffer_idx[step_id::mtp_size].tolist())
                self.mtp_buffer_idx_list = torch.tensor(
                    buffer_idx_list, dtype=torch.int32, device=self.b_buffer_idx.device
                )

        return
