from typing import Optional, List

import torch

from lightllm.models.registry import ModelRegistry
from lightllm.models.deepseek2.model import Deepseek2TpPartModel
from lightllm.models.deepseek3_2.layer_weights.transformer_layer_weight import Deepseek3_2TransformerLayerWeight
from lightllm.models.deepseek3_2.layer_infer.transformer_layer_infer import Deepseek3_2TransformerLayerInfer
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.models.deepseek3_2.infer_struct import Deepseek3_2FlashAttentionStateInfo
from lightllm.models.deepseek3_2.mem_manager import Deepseek3_2MemoryManager, Deepseek3_2FP8KVMemoryManager
from lightllm.common.basemodel.cache_ops import PrefillHookProvider
from lightllm.common.basemodel.cache_utils import capture_old_positions
from lightllm.models.deepseek3_2.triton_kernel.copy_indexer_ks import copy_indexer_ks


@ModelRegistry(["deepseek_v32"])
class Deepseek3_2TpPartModel(Deepseek2TpPartModel, PrefillHookProvider):
    # weight class
    transformer_weight_class = Deepseek3_2TransformerLayerWeight

    # infer class
    transformer_layer_infer_class = Deepseek3_2TransformerLayerInfer

    # infer state class
    infer_state_class = Deepseek3_2FlashAttentionStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
        self.index_topk = self.config["index_topk"]
        return

    def _init_inferstate_cls(self):
        self.infer_state_class = Deepseek3_2FlashAttentionStateInfo

    def _init_mem_manager(self):
        manager_class = Deepseek3_2MemoryManager
        if "triton_fp8kv" in self.mode:
            manager_class = Deepseek3_2FP8KVMemoryManager

        # mtp 模式下需要在mem manger上扩展draft model使用的layer
        added_mtp_layer_num = 0
        if get_env_start_args().mtp_mode == "deepseekv3_eagle":
            added_mtp_layer_num += 1
        elif get_env_start_args().mtp_mode == "deepseekv3_vanilla":
            added_mtp_layer_num += get_env_start_args().mtp_step

        self.mem_manager = manager_class(
            self.max_total_token_num,
            dtype=self.data_type,
            head_num=1,
            head_dim=self.config["kv_lora_rank"] + self.config["qk_rope_head_dim"],
            layer_num=self.config["num_hidden_layers"] + added_mtp_layer_num,
            mem_fraction=self.mem_fraction,
        )
        return

    # ===== PrefillHookProvider Implementation =====

    def capture_prefill_state(
        self,
        req_to_token_indexs: torch.Tensor,
        b_req_idx: torch.Tensor,
        b_ready_cache_len: torch.Tensor,
    ) -> Optional[List[Optional[torch.Tensor]]]:
        """
        Capture old indexer_ks positions before KV cache reorganization.

        DeepSeek V3.2's NSA mechanism requires maintaining both KV cache and
        indexer_ks buffers. When prefix cache is hit, we need to track the old
        positions of cached tokens to copy their indexer_ks data to new positions.

        Args:
            req_to_token_indexs: Request to token index mapping
            b_req_idx: Batch request indices
            b_ready_cache_len: Batch ready cache lengths (for prefix cache)

        Returns:
            List of old positions for each request (None if no cached tokens)
        """
        return capture_old_positions(req_to_token_indexs, b_req_idx, b_ready_cache_len)

    def sync_prefill_buffers(
        self,
        captured_state: List[Optional[torch.Tensor]],
        req_to_token_indexs: torch.Tensor,
        b_req_idx: torch.Tensor,
        b_ready_cache_len: torch.Tensor,
    ) -> None:
        """
        Synchronize indexer_ks buffer with KV cache after prefix cache hit.

        When prefix cache is hit, the KV cache is relocated to contiguous memory.
        This method copies the indexer_ks data to match the new KV cache layout,
        ensuring consistency for the NSA indexer's top-k selection.

        Args:
            captured_state: Old indexer_ks positions from capture_prefill_state()
            req_to_token_indexs: Request to token index mapping (new positions)
            b_req_idx: Batch request indices
            b_ready_cache_len: Batch ready cache lengths
        """
        old_indexer_ks_positions = captured_state
        if old_indexer_ks_positions is None:
            return

        mem_manager = self.req_manager.mem_manager
        if not hasattr(mem_manager, "indexer_ks_mem_manager"):
            return

        num_layers = len(mem_manager.indexer_ks_mem_manager.kv_buffer)
        indexer_buffer = mem_manager.indexer_ks_mem_manager.kv_buffer

        for layer_idx in range(num_layers):
            for i in range(b_req_idx.shape[0]):
                req_idx = b_req_idx[i].item()
                ready_cache_len = b_ready_cache_len[i].item()
                old_positions = old_indexer_ks_positions[i]

                if ready_cache_len > 0 and old_positions is not None:
                    # New positions after KV cache reorganization
                    new_positions = req_to_token_indexs[req_idx, 0:ready_cache_len]

                    # Copy indexer_ks: old_positions -> new_positions
                    copy_indexer_ks(
                        buffer=indexer_buffer[layer_idx],
                        src_loc=old_positions,
                        dest_loc=new_positions,
                    )
