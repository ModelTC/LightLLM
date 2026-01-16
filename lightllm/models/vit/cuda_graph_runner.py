# Copyright 2024 ModelTC Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ViT CUDA Graph Runner for vision encoder optimization."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Dict, Tuple, Optional

import torch

from lightllm.utils.log_utils import init_logger

if TYPE_CHECKING:
    from lightllm.models.vit.model import VisionTransformer

logger = init_logger(__name__)


class ViTCudaGraphRunner:
    """CUDA Graph runner for Vision Transformer encoder.

    Captures the forward pass of ViT layers as CUDA graphs for faster inference.
    Graphs are keyed by (batch_size, seq_len) since ViT uses [B, S, H] tensor shape.
    """

    def __init__(self, vit_model: "VisionTransformer", max_batch_size: int) -> None:
        self.vit_model = vit_model
        self.max_batch_size = max_batch_size

        # Graph storage: key -> (graph, input_buffer, output_buffer)
        self.graphs: Dict[Tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.input_buffers: Dict[Tuple[int, int], torch.Tensor] = {}
        self.output_buffers: Dict[Tuple[int, int], torch.Tensor] = {}

        # Stable cu_seqlens buffers for flash attention
        self.cu_seqlens_buffers: Dict[Tuple[int, int], torch.Tensor] = {}

        # Memory pool for graph capture (shared across graphs)
        self.mempool = torch.cuda.graph_pool_handle() if torch.cuda.is_available() else None

        # Track which shapes have been warmed up
        self._warmed_up: set = set()

        # Statistics tracking
        self._capture_times: Dict[Tuple[int, int], float] = {}
        self._replay_counts: Dict[Tuple[int, int], int] = {}

    @property
    def device(self) -> torch.device:
        return self.vit_model.pre_post_weight.class_embedding.device

    @property
    def dtype(self) -> torch.dtype:
        return self.vit_model.data_type

    def _graph_key(self, x: torch.Tensor) -> Tuple[int, int]:
        if x.ndim == 4:
            # Pixel values: [B, C, H, W] -> batch_size only matters
            # After patch embedding, seq_len is determined by image size
            batch_size = x.shape[0]
            # Calculate expected sequence length after patch embedding
            patch_size = self.vit_model.config.get("patch_size", 14)
            image_h, image_w = x.shape[2], x.shape[3]
            seq_len = (image_h // patch_size) * (image_w // patch_size) + 1  # +1 for class token
            return (batch_size, seq_len)
        else:
            # Already embedded: [B, S, H]
            return (x.shape[0], x.shape[1])

    def _build_cu_seqlens(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.arange(0, (batch_size + 1) * seq_len, step=seq_len, device=device, dtype=torch.int32)

    def _forward_layers(self, pixel_values: torch.Tensor, cu_seqlens: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-layer inference
        input_embs = self.vit_model.pre_infer.forward(pixel_values, self.vit_model.pre_post_weight)

        # Transformer layers
        for i in range(self.vit_model.layers_num + self.vit_model.select_layer + 1):
            input_embs = self.vit_model.layers_infer[i].forward(
                input_embs,
                self.vit_model.trans_layers_weight[i],
                cu_seqlens=cu_seqlens,
            )

        # Post-layer inference
        output = self.vit_model.post_infer.forward(input_embs[:, 1:, :], self.vit_model.pre_post_weight)

        return output

    def _warmup(self, pixel_values: torch.Tensor, key: Tuple[int, int]) -> torch.Tensor:
        if key in self._warmed_up:
            return None

        logger.info(f"ViT CUDA Graph warmup for shape: batch_size={key[0]}, seq_len={key[1]}")

        # Run eager forward pass to initialize any lazy components
        # Note: We don't use cache_env_in/out here to avoid cache manager interference
        with torch.no_grad():
            output = self._forward_layers(pixel_values)

        self._warmed_up.add(key)
        return output

    def _capture_graph(self, pixel_values: torch.Tensor, key: Tuple[int, int]) -> None:
        batch_size, seq_len = key
        device = pixel_values.device

        logger.info(f"Capturing ViT CUDA Graph for shape: batch_size={batch_size}, seq_len={seq_len}")

        # Track capture time
        capture_start = time.perf_counter()

        # Create stable input buffer
        self.input_buffers[key] = torch.empty_like(pixel_values).contiguous()
        self.input_buffers[key].copy_(pixel_values)

        # Create stable cu_seqlens buffer
        self.cu_seqlens_buffers[key] = self._build_cu_seqlens(batch_size, seq_len, device)

        # Synchronize before capture
        torch.cuda.synchronize()

        # Create and capture the graph
        graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(graph, pool=self.mempool):
            output = self._forward_layers(
                self.input_buffers[key],
                cu_seqlens=self.cu_seqlens_buffers[key],
            )
            # Store output buffer reference (must be inside graph context)
            self.output_buffers[key] = output

        self.graphs[key] = graph

        # Synchronize after capture
        torch.cuda.synchronize()

        # Record capture time and initialize replay count
        capture_time = (time.perf_counter() - capture_start) * 1000  # ms
        self._capture_times[key] = capture_time
        self._replay_counts[key] = 0

        logger.info(
            f"ViT CUDA Graph captured successfully for shape: batch_size={batch_size}, "
            f"seq_len={seq_len}, capture_time={capture_time:.2f}ms"
        )

    @torch.no_grad()
    def warmup(self) -> None:
        """Capture CUDA graphs for all batch sizes from 1 to max_batch_size."""
        logger.info(f"Begin capturing ViT CUDA graphs for batch sizes 1 to {self.max_batch_size}")

        image_h = self.vit_model.IMAGE_H
        image_w = self.vit_model.IMAGE_W

        for batch_size in range(self.max_batch_size, 0, -1):
            # Create dummy input
            dummy_input = torch.randn(batch_size, 3, image_h, image_w, dtype=self.dtype, device=self.device)

            key = self._graph_key(dummy_input)

            # Warmup pass
            self._warmup(dummy_input, key)

            # Capture graph
            self._capture_graph(dummy_input, key)

            # Clean up
            del dummy_input
            torch.cuda.empty_cache()

        logger.info(
            f"ViT CUDA graph capture complete. {len(self.graphs)} graphs captured. "
            f"Batch sizes 1-{self.max_batch_size} will use CUDA graph."
        )

    def run(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.contiguous()
        key = self._graph_key(pixel_values)

        if key not in self.graphs:
            batch_size = key[0]
            raise RuntimeError(
                f"No CUDA graph captured for batch_size={batch_size}. "
                f"Max captured batch_size={self.max_batch_size}. "
                f"Increase --vit_cudagraph_max_size or use --disable_vit_cudagraph."
            )

        # Copy input data to stable buffer
        self.input_buffers[key].copy_(pixel_values)

        # Replay the captured graph
        self.graphs[key].replay()

        # Update replay count
        self._replay_counts[key] = self._replay_counts.get(key, 0) + 1

        # Return the output from stable buffer
        return self.output_buffers[key]

    def get_captured_graphs_info(self) -> list:
        info = []
        for key in self.graphs:
            batch_size, seq_len = key
            info.append(
                {
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "capture_time_ms": self._capture_times.get(key, 0),
                    "replay_count": self._replay_counts.get(key, 0),
                }
            )
        return sorted(info, key=lambda x: x["batch_size"])

    def print_stats(self) -> None:
        info = self.get_captured_graphs_info()
        if not info:
            logger.info("No ViT CUDA Graphs captured yet.")
            return

        logger.info(f"ViT CUDA Graph Statistics ({len(info)} graphs captured):")
        logger.info(f"{'Batch':<8} {'SeqLen':<10} {'CaptureTime':<15} {'Replays':<10}")
        logger.info("-" * 45)
        for g in info:
            logger.info(
                f"{g['batch_size']:<8} {g['seq_len']:<10} " f"{g['capture_time_ms']:<15.2f} {g['replay_count']:<10}"
            )

    def has_graph(self, batch_size: int, seq_len: Optional[int] = None) -> bool:
        if seq_len is None:
            patch_size = self.vit_model.config.get("patch_size", 14)
            image_h = self.vit_model.IMAGE_H
            image_w = self.vit_model.IMAGE_W
            seq_len = (image_h // patch_size) * (image_w // patch_size) + 1

        return (batch_size, seq_len) in self.graphs

    @property
    def num_graphs(self) -> int:
        return len(self.graphs)

    def clear(self) -> None:
        num_graphs = len(self.graphs)
        self.graphs.clear()
        self.input_buffers.clear()
        self.output_buffers.clear()
        self.cu_seqlens_buffers.clear()
        self._warmed_up.clear()
        self._capture_times.clear()
        self._replay_counts.clear()
        torch.cuda.empty_cache()
        logger.info(f"ViT CUDA Graph cache cleared ({num_graphs} graphs removed)")
