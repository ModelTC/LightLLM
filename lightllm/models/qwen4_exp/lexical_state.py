from dataclasses import dataclass
from functools import lru_cache
from math import prod

import torch

from lightllm.common.attention_state.bounded_att import BoundedAttentionState
from lightllm.common.attention_state.bounded_att.copy_kernel import copy_bounded_att_tensors
from lightllm.common.attention_state.bounded_att.transfer_utils import bounded_att_buffer_copies
from lightllm.common.attention_state.bounded_att.utils import create_page_pool_tensor, register_page_pool_tensor
from lightllm.common.attention_state.types import AxisMapping, TensorField, TransferRegion
from lightllm.common.attention_state.utils import deepcopy_result
from .lexical_cache import LexicalRequestCache
from .speculative_cache import LexicalSpeculativeCache


@dataclass(frozen=True)
class LexicalStateConfig:
    logical_layer_ids: tuple[int, ...]
    ngram_context: int
    conv_context: int
    width: int
    eos_token_id: int
    dtype: torch.dtype
    tp_rank: int = 0


class LexicalPages:
    def __init__(self, capacity, config, device, pin_memory):
        count = len(config.logical_layer_ids)
        self.ngram = create_page_pool_tensor(
            size=(capacity, count, config.ngram_context),
            dtype=torch.int64,
            device=torch.device(device),
            pin_memory=pin_memory,
        )
        self.convolution = create_page_pool_tensor(
            size=(capacity, count, config.conv_context, config.width),
            dtype=config.dtype,
            device=torch.device(device),
            pin_memory=pin_memory,
        )

    def register_pin_memory(self):
        for tensor in (self.ngram, self.convolution):
            register_page_pool_tensor(tensor=tensor)


class Qwen4ExpLexicalState(BoundedAttentionState):
    def __init__(self, config, request_capacity, mtp_step=0, device="meta"):
        self.config = config
        self.state_name = None
        self.request_capacity = request_capacity
        self.mtp_step = mtp_step
        count = len(config.logical_layer_ids)
        self.ngram = torch.full(
            (count, request_capacity, config.ngram_context), config.eos_token_id, dtype=torch.int64, device=device
        )
        self.convolution = torch.zeros(
            (count, request_capacity, config.conv_context, config.width), dtype=config.dtype, device=device
        )
        self.big_pages_pool = self.small_pages_pool = None
        self.lexical_caches = {}
        self.speculative_lexical_caches = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["lexical_caches"] = {}
        state["speculative_lexical_caches"] = {}
        return state

    def get_state_type(self):
        return "qwen4_lexical"

    def contains_logical_layer(self, logical_layer_id):
        return logical_layer_id in self.config.logical_layer_ids

    def get_layer_view(self, logical_layer_id):
        layer = self.config.logical_layer_ids.index(logical_layer_id)
        return self.ngram[layer], self.convolution[layer]

    def bind_weights(self, layer_weights):
        for weight in layer_weights:
            if not self.contains_logical_layer(weight.layer_num_):
                continue
            cache = LexicalRequestCache.__new__(LexicalRequestCache)
            cache.layer_weight = weight
            cache.eos_token_id = self.config.eos_token_id
            cache.ngram, cache.convolution = self.get_layer_view(weight.layer_num_)
            self.lexical_caches[weight.layer_num_] = cache
            if self.mtp_step:
                self.speculative_lexical_caches[weight.layer_num_] = LexicalSpeculativeCache(cache, self.mtp_step + 1)

    def prepare_lexical(self, metadata, reset_empty_prefix=False):
        return {layer: cache.prepare(metadata, reset_empty_prefix) for layer, cache in self.lexical_caches.items()}

    def commit_lexical(self, metadata, batches, accepted_lengths=None):
        for layer, cache in self.lexical_caches.items():
            cache.commit(metadata, batches[layer], accepted_lengths)

    def stage_speculative_lexical(self, layer, metadata, batch):
        self.speculative_lexical_caches[layer].stage(metadata, batch)

    def initialize_page_pools(self, big_page_capacity, small_page_capacity=0, device="cpu", pin_memory=True):
        self.big_pages_pool = (
            LexicalPages(big_page_capacity, self.config, device, pin_memory) if big_page_capacity else None
        )
        self.small_pages_pool = (
            LexicalPages(small_page_capacity, self.config, device, pin_memory) if small_page_capacity else None
        )

    def reset_bounded_att_state(self, req_indices):
        for request in req_indices:
            self.ngram[:, request].fill_(self.config.eos_token_id)
            self.convolution[:, request].zero_()

    def _pool(self, page_kind):
        assert page_kind in ("big", "small")
        pool = self.big_pages_pool if page_kind == "big" else self.small_pages_pool
        assert pool is not None, f"{page_kind} lexical pages are disabled"
        return pool

    def _copy_pages(self, req_indices, state_token_lens, page_kind, page_slot_ids, write):
        assert len(req_indices) == len(state_token_lens) == len(page_slot_ids)
        pool = self._pool(page_kind)
        for name in ("ngram", "convolution"):
            copies = [
                (getattr(self, name)[:, req], getattr(pool, name)[slot])
                for req, slot in zip(req_indices, page_slot_ids)
            ]
            if not write:
                copies = [(dst, src) for src, dst in copies]
            copy_bounded_att_tensors(copies=copies)

    def save_bounded_att_runtime_to_pages(self, req_indices, state_token_lens, page_kind, page_slot_ids):
        self._copy_pages(req_indices, state_token_lens, page_kind, page_slot_ids, True)

    def restore_bounded_att_runtime_from_pages(self, req_indices, state_token_lens, page_kind, page_slot_ids):
        self._copy_pages(req_indices, state_token_lens, page_kind, page_slot_ids, False)

    def commit_bounded_att_state(self, req_indices, accepted_steps, req_verify_width):
        starts = torch.arange(req_indices.numel(), dtype=torch.int32, device=req_indices.device)
        for cache in self.speculative_lexical_caches.values():
            cache.commit(req_indices, starts, accepted_steps, self.request_capacity - 1)

    @deepcopy_result
    @lru_cache(maxsize=None)
    def get_cpu_transfer_regions(self):
        return self._regions()

    @deepcopy_result
    @lru_cache(maxsize=None)
    def get_pd_transfer_regions(self):
        return self._regions()

    def _regions(self):
        regions = []
        offset = 0
        for name, tensor in (("ngram", self.ngram), ("convolution", self.convolution)):
            shape = tensor.shape[:1] + tensor.shape[2:]
            axes = ("layer", "context") if name == "ngram" else ("layer", "context", "dim")
            end = offset + prod(shape) * tensor.element_size()
            regions.append(
                TransferRegion(
                    field=name,
                    byte_start=offset,
                    byte_end=end,
                    dtype=tensor.dtype,
                    axes=axes,
                    shape=tuple(shape),
                    axis_mappings=tuple(
                        AxisMapping(axis=axis, local_start=0, external_start=0, length=size)
                        for axis, size in zip(axes[1:], shape[1:])
                    ),
                    write_enabled=self.config.tp_rank == 0,
                )
            )
            offset = end
        return tuple(regions)

    def _fields(self, ngram, convolution):
        return {
            "ngram": TensorField(ngram, ("layer", "context")),
            "convolution": TensorField(convolution, ("layer", "context", "dim")),
        }

    def _transfer(self, fields, buffers, pages, layout, write):
        assert layout.state_name == self.state_name
        for copies in bounded_att_buffer_copies(fields, buffers, layout, pages, write):
            copy_bounded_att_tensors(copies=copies)

    def write_state_pages_to_cpu(
        self, page_kind, page_slot_ids, cpu_page_buffers, cpu_page_indices, layout, cpu_page_ready_flags=None
    ):
        assert len(page_slot_ids) == len(cpu_page_indices)
        if cpu_page_ready_flags is not None:
            assert len(cpu_page_ready_flags) == len(cpu_page_indices)
        pool = self._pool(page_kind)
        selected = [i for i in range(len(page_slot_ids)) if cpu_page_ready_flags is None or not cpu_page_ready_flags[i]]
        self._transfer(
            [self._fields(pool.ngram[page_slot_ids[i]], pool.convolution[page_slot_ids[i]]) for i in selected],
            cpu_page_buffers,
            [cpu_page_indices[i] for i in selected],
            layout,
            True,
        )

    def read_state_pages_from_cpu(self, cpu_page_buffers, cpu_page_indices, big_page_slot_ids, layout):
        assert len(cpu_page_indices) == len(big_page_slot_ids)
        pool = self._pool("big")
        self._transfer(
            [self._fields(pool.ngram[slot], pool.convolution[slot]) for slot in big_page_slot_ids],
            cpu_page_buffers,
            cpu_page_indices,
            layout,
            False,
        )

    def copy_big_pages_from_dp(self, src_page_ids, dst_page_ids, remote_dp_bounded_state):
        assert self.config == remote_dp_bounded_state.config
        assert len(src_page_ids) == len(dst_page_ids)
        for name in ("ngram", "convolution"):
            copy_bounded_att_tensors(
                copies=[
                    (getattr(remote_dp_bounded_state._pool("big"), name)[src], getattr(self._pool("big"), name)[dst])
                    for src, dst in zip(src_page_ids, dst_page_ids)
                ]
            )

    def copy_runtime_state_from_dp(self, src_req_indices, dst_req_indices, remote_dp_bounded_state):
        assert self.config == remote_dp_bounded_state.config
        assert len(src_req_indices) == len(dst_req_indices)
        for name in ("ngram", "convolution"):
            copy_bounded_att_tensors(
                copies=[
                    (getattr(remote_dp_bounded_state, name)[:, src], getattr(self, name)[:, dst])
                    for src, dst in zip(src_req_indices, dst_req_indices)
                ]
            )

    def write_to_pd_page(self, req_idx, pd_page_buffers, pd_page_index, state_token_len, layout):
        self._transfer(
            [self._fields(self.ngram[:, req_idx], self.convolution[:, req_idx])],
            pd_page_buffers,
            [pd_page_index],
            layout,
            True,
        )

    def read_from_pd_page(self, pd_page_buffers, req_idx, pd_page_index, state_token_len, layout):
        self._transfer(
            [self._fields(self.ngram[:, req_idx], self.convolution[:, req_idx])],
            pd_page_buffers,
            [pd_page_index],
            layout,
            False,
        )
