import torch
import torch.distributed as dist
import rpyc
import time
from typing import Dict, List, Tuple, Optional, Union
from rpyc.utils.classic import obtain
from .decode_impl import DecodeNode
from lightllm.common.basemodel.infer_lock import acquire_lock_until_ready, release_acquired_lock, g_router_lock
from .decode_task_cache import g_kv_move_task_cache, g_success_kv_move_task_cache
from lightllm.server.pd_io_struct import KVMoveTask
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class PDDecodeInferRpcServer(rpyc.Service):
    def __init__(self, backend: DecodeNode) -> None:
        super().__init__()
        self.backend = backend
        self.device_id = self.backend.current_device_id
        self.dp_rank_in_node = self.backend.dp_rank_in_node
        self.is_master_in_dp = self.backend.is_master_in_dp
        return

    def on_connect(self, conn):
        torch.cuda.set_device(f"cuda:{self.device_id}")
        return

    def judge_token_is_ok(self, key_len, max_new_token):
        # 多 dp 单卡模式下, 每个 dp 各自处理自己的, 不需要同步
        if self.backend.dp_world_size == 1:
            with g_router_lock.obj:
                shared_token_load = self.backend.shared_token_load
                peak_num = shared_token_load.get_estimated_peak_token_count(self.dp_rank_in_node)
                peak_num += shared_token_load.get_frozened_token_count(self.dp_rank_in_node)
                peak_num += key_len + max_new_token

                if peak_num < self.backend.get_max_total_token_num():
                    object_list = [True]
                    shared_token_load.add_frozened_token_count(key_len + max_new_token, self.dp_rank_in_node)
                else:
                    object_list = [False]
            return object_list[0]

        # 普通单dp模式下, 只有主 rank 处理信息，并将数据同步到其他rank上
        if self.is_master_in_dp:
            with g_router_lock.obj:
                shared_token_load = self.backend.shared_token_load
                peak_num = shared_token_load.get_estimated_peak_token_count(self.dp_rank_in_node)
                peak_num += shared_token_load.get_frozened_token_count(self.dp_rank_in_node)
                peak_num += key_len + max_new_token

                if peak_num < self.backend.get_max_total_token_num():
                    object_list = [True]
                    shared_token_load.add_frozened_token_count(key_len + max_new_token, self.dp_rank_in_node)
                else:
                    object_list = [False]
            dist.broadcast_object_list(object_list, src=0, group=self.backend.lock_nccl_group)
        else:
            object_list = [None]
            dist.broadcast_object_list(object_list, src=0, group=self.backend.lock_nccl_group)
        return object_list[0]

    def recover_frozen_token(self, key_len, max_new_token):
        if self.is_master_in_dp:
            with g_router_lock.obj:
                shared_token_load = self.backend.shared_token_load
                shared_token_load.add_frozened_token_count(-(key_len + max_new_token), self.dp_rank_in_node)
        return

    def _alloc_to_frozen_some_tokens(self, move_task: KVMoveTask):
        is_ok = self.judge_token_is_ok(len(move_task.input_tokens), move_task.decode_node.max_new_tokens)
        if not is_ok:
            if self.is_master_in_dp:
                logger.info(f"req_id: {move_task.to_decode_log_info()}  alloc token failed")
                shared_token_load = self.backend.shared_token_load
                dp_rank = self.dp_rank_in_node
                frozen_token_num = shared_token_load.get_frozened_token_count(dp_rank)
                estimated_peak_token_num = shared_token_load.get_estimated_peak_token_count(dp_rank)
                logger.debug(
                    f"radix refed token num {self.backend.radix_cache.get_refed_tokens_num()}\n"
                    f"radix hold token num {self.backend.radix_cache.get_tree_total_tokens_num()}\n"
                    f"mem manager can alloc token num {self.backend.model.mem_manager.can_use_mem_size}\n"
                    f"mem manager total size {self.backend.model.mem_manager.size}"
                    f"frozened token num {frozen_token_num}\n"
                    f"estimated peak token num {estimated_peak_token_num}\n"
                )
            return None

        key = torch.tensor(move_task.input_tokens, dtype=torch.int64, device="cpu")
        tree_node, kv_len, fused_token_indexes = self.backend.radix_cache.match_prefix(key, update_refs=True)
        # 如果没匹配到，说明长度是0， 将fused_token_indexes做一下转换
        fused_token_indexes = [] if fused_token_indexes is None else fused_token_indexes.tolist()
        need_len = len(move_task.input_tokens) - kv_len
        if need_len == 0:
            alloc_token_indexes = []
        else:
            self.backend.radix_cache.free_radix_cache_to_get_enough_token(need_len)
            alloc_token_indexes = self.backend.model.mem_manager.alloc(need_len)
            if alloc_token_indexes is not None:
                alloc_token_indexes = alloc_token_indexes.tolist()

        if alloc_token_indexes is None:
            self.backend.radix_cache.dec_node_ref_counter(tree_node)
            self.recover_frozen_token(len(move_task.input_tokens), move_task.decode_node.max_new_tokens)
            return None

        move_task.decode_token_indexes = alloc_token_indexes
        move_task.move_kv_len = need_len

        g_kv_move_task_cache[move_task.group_request_id] = (move_task, tree_node, fused_token_indexes)
        return move_task.decode_token_indexes

    # 返回 None 代表服务繁忙已经无法调度新的请求进入了
    def exposed_alloc_to_frozen_some_tokens(self, move_tasks: List[KVMoveTask]) -> List[Optional[List[int]]]:
        move_tasks = obtain(move_tasks)
        acquire_lock_until_ready(self.backend.lock_nccl_group)
        try:
            ans_list = []
            for move_task in move_tasks:
                ans_list.append(self._alloc_to_frozen_some_tokens(move_task))
            return ans_list
        except BaseException as e:
            logger.exception(str(e))
            return None
        finally:
            release_acquired_lock()

    def _put_kv_received_to_radix_cache(self, group_req_id: int):
        move_task, tree_node, fused_token_indexes = g_kv_move_task_cache.pop(group_req_id)
        radix_cache = self.backend.radix_cache
        key = torch.tensor(move_task.input_tokens, dtype=torch.int64, device="cpu")
        value = torch.tensor(fused_token_indexes + move_task.decode_token_indexes, dtype=torch.int64, device="cpu")
        prefix_len, _ = radix_cache.insert(key, value)
        assert len(fused_token_indexes) <= prefix_len
        self.backend.model.mem_manager.free(value[len(fused_token_indexes) : prefix_len])
        self.backend.radix_cache.dec_node_ref_counter(tree_node)

        # 申请一段key，把 radix cache 锁住，防止极端情况下被刷掉, decode 端通过减两次引用计数来修正。
        tree_node, kv_len, _ = self.backend.radix_cache.match_prefix(key, update_refs=True)
        assert len(key) == kv_len
        g_success_kv_move_task_cache[group_req_id] = (move_task, tree_node, time.time())
        return

    def exposed_put_kv_received_to_radix_cache(self, group_req_ids: List[int]):
        group_req_ids = obtain(group_req_ids)
        acquire_lock_until_ready(self.backend.lock_nccl_group)
        for group_req_id in group_req_ids:
            self._put_kv_received_to_radix_cache(group_req_id)
        release_acquired_lock()
        return

    def _fail_to_realese_forzen_tokens(self, group_req_id: int):
        move_task, tree_node, fused_token_indexes = g_kv_move_task_cache.pop(group_req_id)
        value = torch.tensor(move_task.decode_token_indexes, dtype=torch.int64, device="cpu")
        self.backend.model.mem_manager.free(value)
        self.backend.radix_cache.dec_node_ref_counter(tree_node)
        self.recover_frozen_token(len(move_task.input_tokens), move_task.decode_node.max_new_tokens)
        return

    def exposed_fail_to_realese_forzen_tokens(self, group_req_ids: List[int]):
        group_req_ids = obtain(group_req_ids)
        acquire_lock_until_ready(self.backend.lock_nccl_group)
        for group_req_id in group_req_ids:
            self._fail_to_realese_forzen_tokens(group_req_id)
        release_acquired_lock()
        return

    def exposed_unfrozen_time_out_reqs_tokens(self):
        acquire_lock_until_ready(self.backend.lock_nccl_group)
        if self.backend.dp_world_size == 1:
            need_release_reqs = self._get_time_out_reqs()
            logger.info(f"kv time out reqs: {need_release_reqs}")
            remove_tokens = self._remove_time_out_reqs(need_release_reqs)
            if remove_tokens != 0:
                with g_router_lock.obj:
                    self.backend.shared_token_load.add_frozened_token_count(-remove_tokens, self.dp_rank_in_node)
        else:
            if self.is_master_in_dp:
                need_release_reqs = self._get_time_out_reqs()
                logger.info(f"kv time out reqs: {need_release_reqs}")
                dist.broadcast_object_list([need_release_reqs], src=0, group=self.backend.lock_nccl_group)
            else:
                receive_objs = [None]
                dist.broadcast_object_list(receive_objs, src=0, group=self.backend.lock_nccl_group)
                need_release_reqs = receive_objs[0]
            remove_tokens = self._remove_time_out_reqs(need_release_reqs)
            if self.is_master_in_dp and remove_tokens != 0:
                with g_router_lock.obj:
                    self.backend.shared_token_load.add_frozened_token_count(-remove_tokens, self.dp_rank_in_node)

        release_acquired_lock()
        return

    def _get_time_out_reqs(self):
        need_release_reqs = []
        for req_id, (_, _, time_mark) in g_success_kv_move_task_cache.items():
            # 6s 这个请求都没有被调度使用，就会主动被删除掉锁定，释放其锁定的token
            if time.time() - time_mark > 6:
                need_release_reqs.append(req_id)
        return need_release_reqs

    def _remove_time_out_reqs(self, need_release_reqs: List[int]) -> int:
        remove_tokens = 0
        for req_id in need_release_reqs:
            task, tree_node, _ = g_success_kv_move_task_cache.pop(req_id)
            self.backend.radix_cache.dec_node_ref_counter(tree_node)
            remove_tokens += len(task.input_tokens) + task.decode_node.max_new_tokens
        return remove_tokens
