from __future__ import annotations

import asyncio
import inspect
import os
import time
import uuid

import rpyc
import torch
import torch.multiprocessing as mp
from rpyc.utils.classic import obtain
from rpyc.utils.factory import unix_connect
from rpyc.utils.server import ThreadedServer

from lightllm.common.kv_cache_mem_manager import MemoryManager
from lightllm.distributed.communication_op import dist_group_manager
from lightllm.models.pi0.config import Pi0VLAConfig
from lightllm.models.pi0.model import Pi0ActionExpertModel
from lightllm.server.actionserver.kv_memory import (
    ActionPrefixContextCache,
    ScopedKVMemoryView,
)
from lightllm.server.actionserver.objs import ActionOutcome, ActionWorkerAck
from lightllm.utils.dist_utils import init_action_distributed_env
from lightllm.utils.envs_utils import (
    get_unique_server_name,
    set_env_start_args,
)
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.log_utils import init_logger
from lightllm.utils.retry_utils import retry

from .model_rpc_client import ActionModelRpcClient

logger = init_logger(__name__)


class ActionModelRpcServer(rpyc.Service):
    def exposed_init_model(self, kvargs):
        kvargs = obtain(kvargs)
        self.args = kvargs["args"]
        set_env_start_args(self.args)
        init_action_distributed_env(kvargs)
        dist_group_manager.create_groups(group_size=1)

        self.tp_rank_id = kvargs["tp_rank_id"]
        self.device_id = kvargs["device_id"]
        self.config = Pi0VLAConfig.from_start_args(self.args)

        shared_mem_manager = MemoryManager.loads_from_shm(self.tp_rank_id)
        self.shared_mem_view = ScopedKVMemoryView(shared_mem_manager)
        self.prefix_contexts = ActionPrefixContextCache(device=self.shared_mem_view.req_to_token_indexs.device)
        self.model = Pi0ActionExpertModel(
            self.config,
            self.shared_mem_view,
            device=f"cuda:{self.device_id}",
            tp_group=dist_group_manager.get_default_group(),
            quant_type=self.args.quant_type or "none",
            quant_cfg_path=self.args.quant_cfg,
        )
        return

    @torch.no_grad()
    def exposed_run_task(self, task):
        task = obtain(task)
        started = time.perf_counter()
        mapping_started = False
        actions_cpu = None
        action_expert_ms = 0.0
        error_info = None
        safe_to_release = True
        outcome = ActionOutcome.SUCCESS
        try:
            suffix_length = task.action_horizon + (0 if self.config.is_pi05 else 1)
            context_identity = task.prefix_context_identity
            if context_identity is None:
                prefix_mem_indexes, scratch_mem_indexes = task.mappings_for_rank(self.tp_rank_id)
                scratch_mem_indexes = scratch_mem_indexes[:suffix_length]
            else:
                has_prefix = task.prefix_mem_indexes is not None
                has_scratch = task.scratch_mem_indexes is not None
                if has_prefix != has_scratch:
                    raise ValueError("prefix context registration requires both KV mappings")
                if has_prefix:
                    prefix_mapping, scratch_mapping = task.mappings_for_rank(self.tp_rank_id)
                    self.prefix_contexts.register(
                        context_identity,
                        prefix_mem_indexes=prefix_mapping,
                        scratch_mem_indexes=scratch_mapping,
                        prefix_seq_lens=task.prefix_seq_lens,
                    )
                prefix_mem_indexes, scratch_mem_indexes = self.prefix_contexts.resolve(
                    context_identity,
                    prefix_seq_lens=task.prefix_seq_lens,
                    suffix_length=suffix_length,
                )
            action_req_indexes = self.shared_mem_view.begin_task_mapping(
                identity=task.identity,
                target_req_indexes=task.prefix_req_indexes,
                prefix_seq_lens=task.prefix_seq_lens,
                scratch_mem_indexes=scratch_mem_indexes,
                prefix_mem_indexes=prefix_mem_indexes,
                action_req_indexes=task.action_req_indexes,
            )
            mapping_started = True
            output = self.model.sample_actions(
                prefix_req_indexes=action_req_indexes,
                prefix_seq_lens=task.prefix_seq_lens,
                scratch_mem_indexes=scratch_mem_indexes,
                state=task.state,
                noise=task.noisy_actions,
                num_steps=task.num_denoise_steps,
                action_dim=task.action_dim,
                action_horizon=task.action_horizon,
            )
            actions_cpu = output.actions.detach().float().cpu()
            action_expert_ms = output.resolve_action_timing()
            if not torch.isfinite(actions_cpu).all():
                raise FloatingPointError("action expert produced non-finite actions")
        except Exception as exc:
            logger.exception("action expert request %s failed", task.request_id)
            outcome = ActionOutcome.ERROR
            error_info = repr(exc)
        finally:
            if mapping_started:
                # An ACK means this rank can no longer touch prefix or scratch
                # pages.  Synchronize before dropping the task-scoped lease,
                # including on an exception path with outstanding kernels.
                try:
                    torch.cuda.synchronize(self.device_id)
                    self.shared_mem_view.end_task_mapping(task.identity)
                except Exception as exc:
                    logger.exception(
                        "failed to retire action KV lease for request %s",
                        task.request_id,
                    )
                    safe_to_release = False
                    outcome = ActionOutcome.RESTART_REQUIRED
                    suffix = repr(exc)
                    error_info = suffix if error_info is None else f"{error_info}; {suffix}"

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return ActionWorkerAck(
            slot_index=task.slot_index,
            generation=task.generation,
            task_id=task.task_id,
            rank=self.tp_rank_id,
            outcome=outcome,
            safe_to_release=safe_to_release,
            actions=actions_cpu if self.tp_rank_id == 0 and outcome is ActionOutcome.SUCCESS else None,
            action_expert_ms=action_expert_ms,
            total_ms=elapsed_ms,
            error_info=error_info,
        )

    def exposed_release_prefix_context(self, identity):
        identity = obtain(identity)
        return self.prefix_contexts.release(identity)


def _init_env(socket_path: str, success_event):
    graceful_registry(inspect.currentframe().f_code.co_name)
    import setproctitle

    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::action_model_infer")
    import lightllm.utils.rpyc_fix_utils as _  # noqa: F401

    server = ThreadedServer(
        ActionModelRpcServer(),
        socket_path=socket_path,
        protocol_config={"allow_pickle": True},
    )
    success_event.set()
    server.start()


async def start_model_process():
    socket_path = f"/tmp/lightllm_action_infer_{uuid.uuid4().hex[:8]}.sock"
    if os.path.exists(socket_path):
        os.remove(socket_path)
    success_event = mp.Event()
    process = mp.Process(target=_init_env, args=(socket_path, success_event))
    process.start()
    await asyncio.to_thread(success_event.wait, timeout=40)
    assert process.is_alive()
    connection = retry(max_attempts=20, wait_time=2)(unix_connect)(socket_path, config={"allow_pickle": True})
    return ActionModelRpcClient(connection)
