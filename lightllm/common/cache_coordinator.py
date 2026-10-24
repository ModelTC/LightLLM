import torch
import triton
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.multimodal_chunk_utils import clamp_mm_cache_len


class CacheCoordinator:
    has_state_cache = False
    state_manager = None
    linear_state_manager = None
    indexer_state_manager = None

    def __init__(self):
        self.mem_manager = None

    def bind_mem_manager(self, mem_manager):
        if self.mem_manager is not None and self.mem_manager is not mem_manager:
            raise ValueError("CacheCoordinator is already bound to a different memory manager")
        self.mem_manager = mem_manager

    def validate_radix_cache(self, radix_cache):
        if radix_cache is not None and radix_cache.mem_manager is not self.mem_manager:
            raise ValueError("Radix cache and coordinator must share the memory manager")

    def create_state_transfer(self):
        return None

    def restore_prefix(self, req, req_manager, radix_cache):
        enabled = not req.sampling_param.disable_prompt_cache and radix_cache is not None
        if enabled and req.get_cur_total_len() > 1 and req.cur_kv_len == 0:
            tokens = req.shm_req.shm_prompt_ids.arr[: req.get_cur_total_len()]
            key = torch.tensor(tokens, dtype=torch.int64, device="cpu")[:-1]
            node, _, indexes = radix_cache.match_prefix(key, update_refs=True, mm_input_spans=req.mm_input_spans)
            if node is not None:
                length = node.node_prefix_total_len
                req_manager.req_to_token_indexs[req.req_idx, :length] = indexes
                req.shared_kv_node = node
                req.cur_kv_len = req.hold_kv_len = int(length)
                assert req.hold_kv_len % req.args.page_size == 0
                req.shm_req.prompt_cache_len = req.cur_kv_len
        req.shm_req.shm_cur_kv_len = req.cur_kv_len

    def save_state_cache(self, b_req_idx, reqs, radix_cache, args):
        """Snapshot after forward, before advancing req.cur_kv_len from the chunk start."""
        return

    def get_cacheable_prefix_len(self, req, args):
        return req.cur_kv_len

    def release_request_cache(self, req, req_manager, radix_cache, args, publish_cache) -> list[torch.Tensor]:
        """Publish reusable cache and release unpublished checkpoints and held radix refs.

        Returns borrowed request-index slices for the caller to batch-free. The caller
        resets request lengths and must consume these slices before reusing the request
        row. Call only after pending CPU offload reads have completed. Published KV and
        checkpoints belong to radix; returned KV indexes still need allocator release.
        """
        indexes = req_manager.req_to_token_indexs[req.req_idx]
        shared_len = 0 if req.shared_kv_node is None else req.shared_kv_node.node_prefix_total_len
        if publish_cache and radix_cache is not None:
            cache_len = req.cur_kv_len // args.page_size * args.page_size
            key = torch.tensor(req.get_input_token_ids()[:cache_len], dtype=torch.int64, device="cpu")
            prefix_len, _ = radix_cache.insert(key, indexes[:cache_len].detach().cpu())
            assert shared_len <= prefix_len
            free_indexes = [indexes[cache_len : req.hold_kv_len], indexes[shared_len:prefix_len]]
        else:
            free_indexes = [indexes[shared_len : req.hold_kv_len]]
        if req.shared_kv_node is not None:
            radix_cache.dec_node_ref_counter(req.shared_kv_node)
            req.shared_kv_node = None
        return free_indexes

    def restore_peer_state(self, req_idx, restored_len, peer_state, peer_req_idx):
        return

    def get_cpu_cache_match_page_num(self, page_indexes, page_lens, cpu_cache_client):
        return len(page_indexes)

    def load_cpu_cache_to_gpu(self, mem_indexes, page_indexes, cpu_cache_client, req, page_indexes_cpu=None) -> None:
        """Enqueue KV and any live request-state restoration on the current stream."""
        self.mem_manager.operator.load_cpu_cache_to_gpu(mem_indexes, page_indexes, cpu_cache_client, req)

    def offload_gpu_kv_to_cpu_cache(
        self,
        mem_indexes,
        page_indexes,
        page_readies,
        cpu_cache_client,
        req,
        radix_cache=None,
        page_indexes_cpu=None,
        page_readies_cpu=None,
    ):
        return self.mem_manager.operator.offload_gpu_kv_to_cpu_cache(
            mem_indexes, page_indexes, page_readies, cpu_cache_client, req
        )


class HybridCacheCoordinator(CacheCoordinator):
    """Coordinate KV prefix reuse and hybrid attention state checkpoints."""

    def __init__(self, linear_state_manager=None, indexer_state_manager=None, *, state_manager=None):
        self.state_manager = state_manager if state_manager is not None else linear_state_manager
        self.linear_state_manager = linear_state_manager
        self.indexer_state_manager = indexer_state_manager
        self.mem_manager = None
        self.small_page_buffers = None
        self.big_page_buffers = None

    @property
    def has_state_cache(self):
        return self.state_manager is not None

    def bind_mem_manager(self, mem_manager):
        if self.mem_manager is mem_manager:
            return
        if self.mem_manager is not None:
            raise ValueError("HybridCacheCoordinator is already bound to a different memory manager")
        if self.has_state_cache:
            args = get_env_start_args()
            big_page_tokens = args.linear_att_page_block_num * args.linear_att_hash_page_size
            # The last two slots are reserved for CPU-cache tail staging, never allocated to requests.
            self.big_page_buffers = self.state_manager.create_checkpoint_pool(
                triton.cdiv(mem_manager.size, big_page_tokens) + 2,
                keep_num=2,
            )
            self.CPU_CACHE_BIG_PAGE_LOAD_TEMP_BUFFER_ID = self.big_page_buffers.size - 2
            self.CPU_CACHE_BIG_PAGE_OFFLOAD_TEMP_BUFFER_ID = self.big_page_buffers.size - 1
            self.cpu_cache_transfer = self.state_manager.create_cpu_transfer(mem_manager, self.big_page_buffers)
        self.mem_manager = mem_manager

    def create_state_transfer(self):
        if not self.has_state_cache:
            return None
        return self.state_manager.create_state_transfer(self.indexer_state_manager)

    def validate_radix_cache(self, radix_cache):
        super().validate_radix_cache(radix_cache)
        if radix_cache is not None and self.has_state_cache:
            if (
                radix_cache.big_page_buffers is not self.big_page_buffers
                or radix_cache.small_page_buffers is not self.small_page_buffers
            ):
                raise ValueError("Radix cache and coordinator must share checkpoint pools")

    def create_small_page_cache_manager(self, size):
        if self.small_page_buffers is not None:
            if self.small_page_buffers.size != size:
                raise ValueError("Cannot replace an initialized checkpoint pool")
            return self.small_page_buffers
        self.small_page_buffers = self.state_manager.create_checkpoint_pool(size)
        return self.small_page_buffers

    def reset(self, req_idx):
        if self.state_manager is not None:
            self.state_manager.reset(req_idx)
        if self.indexer_state_manager is not None:
            self.indexer_state_manager.reset(req_idx)

    def restore_state(self, req_idx, state_cache_manager, buffer_idx, restored_len):
        assert state_cache_manager.has_checkpoint(buffer_idx, restored_len)
        self.state_manager.restore_checkpoint(req_idx, restored_len, state_cache_manager, buffer_idx)
        if self.indexer_state_manager is not None:
            self.indexer_state_manager.reset(req_idx)

    def restore_peer_state(self, req_idx, restored_len, peer_state, peer_req_idx):
        self.state_manager.restore_peer_state(req_idx, restored_len, peer_state, peer_req_idx)
        if self.indexer_state_manager is not None:
            self.indexer_state_manager.state.buffer[:, req_idx].copy_(
                peer_state.req_to_indexer_tail.buffer[:, peer_req_idx], non_blocking=True
            )

    def save_state(self, req_idx, buffer_idx, state_cache_manager, checkpoint_len):
        self.state_manager.save_state(req_idx, buffer_idx, state_cache_manager, checkpoint_len)
        state_cache_manager.checkpoint_lens[buffer_idx] = checkpoint_len

    def save_big_page_states(self, b_req_idx, req_indexes, buffer_indexes, checkpoint_lens):
        self.state_manager.save_checkpoints(
            b_req_idx, req_indexes, buffer_indexes, self.big_page_buffers, checkpoint_lens
        )
        for slot, length in zip(buffer_indexes, checkpoint_lens):
            self.big_page_buffers.checkpoint_lens[slot] = length

    def _restore_shared_prefix(
        self, req_idx, token_indexes, cached_indexes, restored_len, checkpoint_pool, checkpoint_slot
    ):
        token_indexes[:restored_len] = cached_indexes[:restored_len]
        self.restore_state(req_idx, checkpoint_pool, checkpoint_slot, restored_len)

    def restore_prefix(self, req, req_manager, radix_cache):
        enable_prompt_cache = (not req.sampling_param.disable_prompt_cache) and radix_cache is not None
        block_hashs = req.shm_req.hybrid_token_hash_list.get_all()
        hash_page_size = req.args.linear_att_hash_page_size
        match_tokens = max(0, min(len(block_hashs) * hash_page_size, req.get_cur_total_len() - 1))
        match_tokens = (match_tokens // hash_page_size) * hash_page_size
        block_hashs = block_hashs[: match_tokens // hash_page_size]
        assert len(block_hashs) == req.shm_req.hybrid_token_hash_list.size
        big_page_token_num = hash_page_size * req.args.linear_att_page_block_num
        big_page_is_disable = big_page_token_num > req.args.max_req_total_len
        restored_kv_len = req.cur_kv_len
        retained_node = req.shared_kv_node
        if enable_prompt_cache and match_tokens > 1 and block_hashs and req.cur_kv_len == 0:
            input_token_ids = req.shm_req.shm_prompt_ids.arr[: req.get_cur_total_len()]
            key = torch.tensor(input_token_ids[:match_tokens], dtype=torch.int64, device="cpu")
            share_node, _, value_tensor = radix_cache.match_prefix(
                key, block_hashs=block_hashs, update_refs=True, mm_input_spans=req.mm_input_spans
            )
            if share_node is not None:
                assert req.tail_small_page_buffer_id is None
                token_indexes = req_manager.req_to_token_indexs[req.req_idx]
                if not share_node.is_big_page_node() and not big_page_is_disable:
                    matched_len = share_node.node_prefix_total_len
                    big_page_len = (matched_len // big_page_token_num) * big_page_token_num
                    need_tokens = matched_len - big_page_len
                    available_tokens = (
                        self.mem_manager.allocator.can_use_mem_size
                        + radix_cache.get_tree_total_tokens_num()
                        - radix_cache.get_refed_tokens_num()
                    )
                    if available_tokens > need_tokens:
                        assert need_tokens % req.args.page_size == 0
                        radix_cache.free_radix_cache_to_get_enough_token(need_token_num=need_tokens)
                        tail_mems = self.mem_manager.alloc(need_size=need_tokens)
                        token_indexes[:big_page_len] = value_tensor[:big_page_len]
                        token_indexes[big_page_len:matched_len] = tail_mems
                        self.mem_manager.operator.copy_mem_to_mem(value_tensor[big_page_len:matched_len], tail_mems)
                        req.prompt_selected_logprobs.copy_capture_slots_if_needed(
                            source_indexes=value_tensor[big_page_len:matched_len], destination_indexes=tail_mems
                        )
                        self.restore_state(
                            req.req_idx, self.small_page_buffers, share_node.small_page_buffer_idx, matched_len
                        )
                        restored_kv_len = matched_len
                        retained_node = radix_cache.deref_to_first_big_page_node(node=share_node, require_state=False)
                        share_node = None
                    else:
                        share_node = radix_cache.deref_to_first_big_page_node(
                            node=share_node, mm_input_spans=req.mm_input_spans
                        )
                if share_node is not None:
                    restored_kv_len = share_node.node_prefix_total_len
                    retained_node = share_node
                    if share_node.is_big_page_node():
                        pool, slot = self.big_page_buffers, share_node.big_page_buffer_idx
                    else:
                        pool, slot = self.small_page_buffers, share_node.small_page_buffer_idx
                    self._restore_shared_prefix(req.req_idx, token_indexes, value_tensor, restored_kv_len, pool, slot)
                if restored_kv_len > 0:
                    req.shm_req.prompt_cache_len = int(restored_kv_len)

        req.shared_kv_node = retained_node
        req.cur_kv_len = int(restored_kv_len)
        req.shm_req.shm_cur_kv_len = req.cur_kv_len
        req.hold_kv_len = req.cur_kv_len
        assert req.hold_kv_len % req.args.page_size == 0
        if req.cur_kv_len == 0:
            self.reset(req.req_idx)

    def save_state_cache(self, b_req_idx, reqs, radix_cache, args):
        """Snapshot after forward, before advancing req.cur_kv_len from the chunk start."""
        if radix_cache is None:
            return
        # Request-state snapshot at a big-page boundary.
        big_page_token_num = args.linear_att_hash_page_size * args.linear_att_page_block_num
        big_page_buffer_ids, checkpoint_lens, req_rows = [], [], []
        for row, req in enumerate(reqs):
            chunk_end = req.get_chuncked_input_token_len()
            first_boundary = (req.cur_kv_len // big_page_token_num + 1) * big_page_token_num
            for length in range(first_boundary, min(chunk_end, req.hybrid_cache_len) + 1, big_page_token_num):
                slot = self.big_page_buffers.alloc_one_state_cache()
                assert slot is not None and length not in req.hybrid_len_to_big_page_id
                req.hybrid_len_to_big_page_id[length] = slot
                if self.state_manager.capture_intermediate_checkpoints or length == chunk_end:
                    big_page_buffer_ids.append(slot)
                    checkpoint_lens.append(length)
                    req_rows.append(row)
        if big_page_buffer_ids:
            selected_rows = torch.tensor(req_rows, dtype=torch.int64, device=b_req_idx.device)
            self.save_big_page_states(
                b_req_idx=b_req_idx[selected_rows],
                req_indexes=[reqs[row].req_idx for row in req_rows],
                buffer_indexes=big_page_buffer_ids,
                checkpoint_lens=checkpoint_lens,
            )

        assert (
            not args.disable_chunked_prefill or self.state_manager.capture_intermediate_checkpoints
        ), "chunked prefill must be enabled when only chunk-end state can be captured"

        # Request-state snapshot at the final small-page boundary.
        for req in reqs:
            # 判断本次prefill 完以后 kv 的长度是否到达 hybrid checkpoint 的存储边界。
            if req.cur_kv_len < req.hybrid_cache_len <= req.get_chuncked_input_token_len() and (
                self.state_manager.capture_intermediate_checkpoints
                or req.get_chuncked_input_token_len() == req.hybrid_cache_len
            ):
                assert req.tail_small_page_buffer_id is None
                if req.hybrid_cache_len % big_page_token_num != 0:
                    radix_cache.free_one_small_page_buffer()
                    req.tail_small_page_buffer_id = self.small_page_buffers.alloc_one_state_cache()
                    if req.tail_small_page_buffer_id is not None:
                        dst_buffer_idx = req.tail_small_page_buffer_id
                        self.save_state(
                            req_idx=req.req_idx,
                            buffer_idx=dst_buffer_idx,
                            state_cache_manager=self.small_page_buffers,
                            checkpoint_len=req.hybrid_cache_len,
                        )
        return

    def get_cacheable_prefix_len(self, req, args):
        """Peer runtime restores permit continuation, but do not supply historical checkpoints."""
        page_size = args.linear_att_hash_page_size * args.linear_att_page_block_num
        shared_len = 0 if req.shared_kv_node is None else req.shared_kv_node.node_prefix_total_len
        end = min(req.cur_kv_len, req.hybrid_cache_len)
        cacheable_len = 0
        node = req.shared_kv_node
        while node is not None and node.node_prefix_total_len:
            length = node.node_prefix_total_len
            pool, slot = (
                (self.big_page_buffers, node.big_page_buffer_idx)
                if length % page_size == 0
                else (self.small_page_buffers, node.small_page_buffer_idx)
            )
            if pool.has_checkpoint(slot, length):
                cacheable_len = length
                break
            node = node.parent
        for boundary in range((shared_len // page_size + 1) * page_size, end + 1, page_size):
            if boundary not in req.hybrid_len_to_big_page_id:
                return cacheable_len
            slot = req.hybrid_len_to_big_page_id[boundary]
            if self.big_page_buffers.has_checkpoint(slot, boundary):
                cacheable_len = boundary
        if self.small_page_buffers.has_checkpoint(req.tail_small_page_buffer_id, end):
            cacheable_len = end
        return cacheable_len

    def release_request_cache(self, req, req_manager, radix_cache, args, publish_cache) -> list[torch.Tensor]:
        """Publish reusable cache and release unpublished checkpoints and held radix refs.

        Returns borrowed request-index slices for the caller to batch-free. The caller
        resets request lengths and must consume these slices before reusing the request
        row. Call only after pending CPU offload reads have completed. Published KV and
        checkpoints belong to radix; returned KV indexes still need allocator release.
        """
        token_indexes = req_manager.req_to_token_indexs[req.req_idx]
        shared_len = 0 if req.shared_kv_node is None else req.shared_kv_node.node_prefix_total_len
        publish_len = None
        if publish_cache and radix_cache is not None:
            hash_page_size = args.linear_att_hash_page_size
            big_page_tokens = hash_page_size * args.linear_att_page_block_num
            last_big_page = req.hybrid_cache_len // big_page_tokens * big_page_tokens
            assert req.hybrid_cache_len >= shared_len
            if req.tail_small_page_buffer_id is not None:
                assert req.hybrid_cache_len <= req.cur_kv_len
                assert req.hybrid_cache_len % big_page_tokens != 0
                publish_len = req.hybrid_cache_len
            elif shared_len < last_big_page <= req.cur_kv_len:
                publish_len = last_big_page
            else:
                assert shared_len <= req.cur_kv_len

        if publish_len is not None:
            publish_len = min(publish_len, self.get_cacheable_prefix_len(req, args))
            if publish_len <= shared_len:
                publish_len = None

        if publish_len is not None:
            discarded = [length for length in req.hybrid_len_to_big_page_id if length > publish_len]
            if discarded:
                self.big_page_buffers.free_state_cache(
                    [req.hybrid_len_to_big_page_id.pop(length) for length in discarded]
                )
            if req.tail_small_page_buffer_id is not None and publish_len < req.hybrid_cache_len:
                self.small_page_buffers.free_state_cache([req.tail_small_page_buffer_id])
                req.tail_small_page_buffer_id = None
            page_num = publish_len // hash_page_size
            block_hashs = req.shm_req.hybrid_token_hash_list.get_all()[:page_num]
            state_idxs = [None] * page_num
            if req.tail_small_page_buffer_id is not None:
                state_idxs[-1] = req.tail_small_page_buffer_id
            key = torch.tensor(req.get_input_token_ids()[:publish_len], dtype=torch.int64, device="cpu")
            value = token_indexes[:publish_len].detach().cpu()
            prefix_len, _ = radix_cache.insert(
                key,
                value,
                block_hashs=block_hashs,
                block_state_idxs=state_idxs,
                len_to_big_page_id=req.hybrid_len_to_big_page_id,
            )
            assert not req.hybrid_len_to_big_page_id
            req.tail_small_page_buffer_id = None
            assert shared_len <= prefix_len
            free_indexes = [token_indexes[publish_len : req.hold_kv_len], token_indexes[shared_len:prefix_len]]
        else:
            free_indexes = [token_indexes[shared_len : req.hold_kv_len]]
            if req.tail_small_page_buffer_id is not None:
                self.small_page_buffers.free_state_cache([req.tail_small_page_buffer_id])
                req.tail_small_page_buffer_id = None
            if req.hybrid_len_to_big_page_id:
                self.big_page_buffers.free_state_cache(list(req.hybrid_len_to_big_page_id.values()))
                req.hybrid_len_to_big_page_id.clear()

        if req.shared_kv_node is not None:
            radix_cache.dec_node_ref_counter(req.shared_kv_node)
            req.shared_kv_node = None
        return free_indexes

    def get_cpu_cache_match_page_num(self, page_indexes, page_lens, cpu_cache_client):
        lengths = self.cpu_cache_transfer.get_checkpoint_lens(page_indexes, cpu_cache_client)
        for index in range(len(lengths) - 1, -1, -1):
            if lengths[index] == page_lens[index]:
                return index + 1
        return 0

    def load_cpu_cache_to_gpu(self, mem_indexes, page_indexes, cpu_cache_client, req, page_indexes_cpu=None) -> None:
        """Enqueue KV and any live request-state restoration on the current stream."""
        args = get_env_start_args()
        page_size = args.cpu_cache_token_page_size
        assert page_size == args.linear_att_hash_page_size * args.linear_att_page_block_num
        assert len(mem_indexes) % args.linear_att_hash_page_size == 0
        assert triton.cdiv(len(mem_indexes), page_size) == len(page_indexes)
        assert clamp_mm_cache_len(req.cur_kv_len, req.mm_input_spans) == req.cur_kv_len
        if page_indexes_cpu is None:
            page_indexes_cpu = page_indexes.cpu().tolist()
        checkpoint_lens = self.cpu_cache_transfer.get_checkpoint_lens(page_indexes_cpu, cpu_cache_client)
        assert checkpoint_lens[-1] == req.cur_kv_len
        page_count = len(mem_indexes) // page_size
        end = req.cur_kv_len // page_size * page_size
        slots = self.big_page_buffers.alloc_state_cache(page_count)
        assert slots is not None, "Insufficient checkpoint slots for CPU cache restoration"
        first_end = end - (page_count - 1) * page_size
        for index, slot in enumerate(slots):
            length = first_end + index * page_size
            req.hybrid_len_to_big_page_id[length] = slot
            if checkpoint_lens[index] == length:
                self.big_page_buffers.checkpoint_lens[slot] = length
        if len(mem_indexes) % page_size:
            slots.append(self.CPU_CACHE_BIG_PAGE_LOAD_TEMP_BUFFER_ID)
            self.big_page_buffers.checkpoint_lens[slots[-1]] = req.cur_kv_len
        self.cpu_cache_transfer.load(mem_indexes, page_indexes, cpu_cache_client, slots)
        self.restore_state(req.req_idx, self.big_page_buffers, slots[-1], req.cur_kv_len)

    def offload_gpu_kv_to_cpu_cache(
        self,
        mem_indexes,
        page_indexes,
        page_readies,
        cpu_cache_client,
        req,
        radix_cache=None,
        page_indexes_cpu=None,
        page_readies_cpu=None,
    ):
        page_size = get_env_start_args().cpu_cache_token_page_size
        end = len(mem_indexes) // page_size * page_size
        slots = [] if radix_cache is None else radix_cache.get_big_page_ids_by_node(req.shared_kv_node)
        slots = slots[: end // page_size]
        for length in range((len(slots) + 1) * page_size, end + 1, page_size):
            slots.append(req.hybrid_len_to_big_page_id[length])
        tail_state = None
        if len(mem_indexes) % page_size:
            assert req.tail_small_page_buffer_id is not None
            tail_state = self.small_page_buffers.get_state_cache(req.tail_small_page_buffer_id)
            slots.append(self.CPU_CACHE_BIG_PAGE_OFFLOAD_TEMP_BUFFER_ID)
            self.big_page_buffers.checkpoint_lens[slots[-1]] = self.small_page_buffers.checkpoint_lens[
                req.tail_small_page_buffer_id
            ]
        self.cpu_cache_transfer.offload(
            mem_indexes,
            page_indexes,
            page_readies,
            cpu_cache_client,
            slots,
            tail_state,
            page_indexes_cpu=page_indexes_cpu,
            page_readies_cpu=page_readies_cpu,
        )
