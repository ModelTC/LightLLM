"""Action-model worker process."""

from __future__ import annotations

import inspect

import rpyc
import torch
from rpyc.utils.classic import obtain
from rpyc.utils.server import ThreadedServer

from lightllm.distributed.communication_op import dist_group_manager
from lightllm.models.registry import get_generation_model_class
from .protocol import (
    ActionModelTask,
    ActionOutcome,
    ActionWorkerAck,
)
from lightllm.utils.dist_utils import init_generation_distributed_env
from lightllm.utils.envs_utils import get_unique_server_name, set_env_start_args
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.log_utils import init_logger
from lightllm.utils.process_check import start_parent_check_thread
from lightllm.server.generationserver.worker import start_model_worker
from lightllm.server.generationserver.kv_materializer import PagedKVMaterializer

logger = init_logger(__name__)


class ActionModelRpcServer(rpyc.Service):
    """One action rank attached to the corresponding target rank's shared KV."""

    def exposed_init_model(self, kvargs):
        kvargs = obtain(kvargs)
        self.args = kvargs["args"]
        set_env_start_args(self.args)
        init_generation_distributed_env(kvargs)
        dist_group_manager.create_groups(group_size=1)

        # This rank is colocated with the same target-model TP rank. KV storage
        # and logical request rows are attached from their respective owners.
        self.tp_rank_id = int(kvargs["tp_rank_id"])
        self.device_id = int(kvargs["device_id"])
        self.config = self.args.generation_runtime_config
        if self.config is None:
            raise ValueError("action runtime config was not initialized")
        self.kv_materializer = PagedKVMaterializer.from_shared(
            self.tp_rank_id,
            device_index=self.device_id,
        )
        model_class = get_generation_model_class(
            {"type": self.config["model_type"]},
            "action",
        )
        self.model = model_class(
            self.config,
            max_seq_length=self.kv_materializer.max_seq_length,
            dist_group=dist_group_manager.get_default_group(),
            device=f"cuda:{self.device_id}",
            quant_type=self.args.quant_type or "none",
            quant_cfg_path=self.args.quant_cfg,
            enable_cuda_graph=not self.args.disable_cudagraph,
        )

    @torch.no_grad()
    def exposed_run_task(self, task):
        task = obtain(task)
        if not isinstance(task, ActionModelTask):
            raise TypeError(f"invalid action model task: {type(task)!r}")

        # Only rank 0 returns the action tensor; every rank still returns an ACK.
        actions_cpu = None
        action_expert_ms = 0.0
        error_info = None
        # Cleared only when CUDA synchronization cannot prove that shared KV
        # reads are finished. In that case the owner is pinned until restart.
        safe_to_release = True
        outcome = ActionOutcome.SUCCESS
        try:
            # infer_req_idx selects the target request row in req_to_token;
            # seq_len bounds how much of that row belongs to the VLM prefix.
            # Gathering here is the only place the VLA path understands the
            # producer's page table or KV allocator.
            prefix_kv = self.kv_materializer.gather(task.prefix_kv)
            output = self.model.forward(
                prefix_kv=prefix_kv,
                state=task.state,
                noise=task.noisy_actions,
                num_steps=task.num_denoise_steps,
                action_dim=task.action_dim,
                action_horizon=task.action_horizon,
            )
            actions_cpu = output.actions.detach().float().cpu()
            action_expert_ms = output.resolve_timing()
            if not torch.isfinite(actions_cpu).all():
                raise FloatingPointError("action expert produced non-finite actions")
        except Exception as exc:
            logger.exception("action expert task %s failed", task.identity)
            outcome = ActionOutcome.ERROR
            error_info = repr(exc)
        finally:
            # The ACK is the CUDA-safe lease boundary. No action K/V is ever
            # written into the target pool, but reads must finish before the
            # VLAServer releases the ActivationTransfer lease.
            try:
                torch.cuda.synchronize(self.device_id)
            except Exception as exc:
                logger.exception(
                    "failed to retire action KV lease for %s", task.identity
                )
                safe_to_release = False
                outcome = ActionOutcome.RESTART_REQUIRED
                suffix = repr(exc)
                error_info = suffix if error_info is None else f"{error_info}; {suffix}"

        return ActionWorkerAck(
            slot_index=task.identity.slot_index,
            generation=task.identity.generation,
            task_id=task.identity.task_id,
            rank=self.tp_rank_id,
            outcome=outcome,
            safe_to_release=safe_to_release,
            actions=actions_cpu
            if self.tp_rank_id == 0 and outcome is ActionOutcome.SUCCESS
            else None,
            action_expert_ms=action_expert_ms,
            error_info=error_info,
        )


def _init_env(socket_path: str, success_event):
    graceful_registry(inspect.currentframe().f_code.co_name)
    import setproctitle

    setproctitle.setproctitle(
        f"lightllm::{get_unique_server_name()}::generation_action_model"
    )
    start_parent_check_thread()
    import lightllm.utils.rpyc_fix_utils as _  # noqa: F401

    server = ThreadedServer(
        ActionModelRpcServer(),
        socket_path=socket_path,
        protocol_config={"allow_pickle": True},
    )
    success_event.set()
    server.start()


async def start_model_process():
    return await start_model_worker(
        _init_env,
        socket_prefix="lightllm_generation_pi",
        worker_name="action model",
    )
