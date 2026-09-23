"""CPU-only persistent EPLB candidate planner actor."""
import atexit
import multiprocessing as mp
import os
import time
import threading


class EPLBPlannerWorker:
    def __init__(self, timeout_s=120.0):
        self._timeout_s = timeout_s
        self._job_id = 0
        self._rpc_lock = threading.Lock()
        self._parent, child = mp.get_context("spawn").Pipe(duplex=True)
        self._process = mp.get_context("spawn").Process(target=_serve, args=(child, os.getpid()), daemon=True)
        try:
            self._process.start()
            child.close()
            hello = self._receive(timeout_s)
            if hello.get("kind") != "ready" or hello.get("cuda_initialized"):
                raise RuntimeError(f"EPLB planner worker startup failed: {hello}")
            self.info = hello
        except BaseException:
            try:
                child.close()
            except OSError:
                pass
            self.close()
            raise
        atexit.register(self.close)

    def _receive(self, timeout_s):
        if not self._parent.poll(timeout_s):
            raise TimeoutError("EPLB planner worker RPC timed out")
        try:
            reply = self._parent.recv()
        except EOFError as exc:
            raise RuntimeError("EPLB planner worker exited unexpectedly") from exc
        if not isinstance(reply, dict):
            raise RuntimeError("EPLB planner worker returned non-dict response")
        if reply.get("kind") == "error":
            raise RuntimeError("EPLB planner worker failed: " + reply.get("message", "unknown"))
        return reply

    def plan(self, global_load, current_placement, **settings):
        if not self._rpc_lock.acquire(blocking=False):
            raise RuntimeError("EPLB planner worker permits only one in-flight RPC")
        try:
            if self._process is None or not self._process.is_alive():
                raise RuntimeError("EPLB planner worker is not alive")
            if (
                not hasattr(global_load, "device")
                or not hasattr(current_placement, "device")
                or global_load.device.type != "cpu"
                or current_placement.device.type != "cpu"
                or global_load.ndim != 4
                or current_placement.ndim != 3
            ):
                raise ValueError(
                    "EPLB planner worker accepts [samples,layers,nodes,experts] and "
                    "[layers,ranks,slots] CPU tensors only"
                )
            self._job_id += 1
            started = time.perf_counter()
            self._parent.send(
                {
                    "kind": "plan",
                    "job_id": self._job_id,
                    "global_load": global_load,
                    "current_placement": current_placement,
                    "settings": settings,
                }
            )
            reply = self._receive(self._timeout_s)
            if reply.get("kind") != "result" or reply.get("job_id") != self._job_id:
                raise RuntimeError("EPLB planner worker returned mismatched response/job id")
            candidate = reply.get("candidate")
            if (
                not hasattr(candidate, "device")
                or candidate.device.type != "cpu"
                or candidate.dtype != current_placement.dtype
                or candidate.shape != current_placement.shape
            ):
                raise RuntimeError("EPLB planner worker returned invalid candidate")
            return candidate, {
                "compute_ms": reply["compute_ms"],
                "roundtrip_ms": (time.perf_counter() - started) * 1000.0,
            }
        except BaseException:
            self.close()
            raise
        finally:
            self._rpc_lock.release()

    def close(self):
        process = getattr(self, "_process", None)
        if process is None:
            return
        try:
            if process.is_alive():
                try:
                    self._parent.send({"kind": "stop"})
                except (BrokenPipeError, EOFError, OSError):
                    pass
                process.join(5)
            if process.is_alive():
                process.terminate()
                process.join(5)
            if process.is_alive():
                process.kill()
                process.join(5)
            if process.is_alive():
                raise RuntimeError("EPLB planner worker did not exit during close")
        finally:
            try:
                self._parent.close()
            except OSError:
                pass
            if not process.is_alive():
                process.close()
                self._process = None


def _watch_parent(expected_parent_pid):
    while True:
        if os.getppid() != expected_parent_pid:
            os._exit(0)
        time.sleep(0.25)


def _serve(conn, expected_parent_pid):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    threading.Thread(target=_watch_parent, args=(expected_parent_pid,), daemon=True).start()
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe import eplb_planner

        # Startup prewarm is intentionally small and CPU-only.
        eplb_planner.plan_eplb_candidate(
            torch.ones((1, 1, 1, 8), dtype=torch.int64),
            torch.arange(8).view(1, 8, 1),
            full_layout=True,
            world_size=8,
            node_world_size=8,
            num_redundant_experts_per_rank=0,
            expert_alignment=128,
            stickiness=0.1,
        )
        conn.send(
            {
                "kind": "ready",
                "pid": os.getpid(),
                "module": eplb_planner.__file__,
                "torch_threads": torch.get_num_threads(),
                "cuda_initialized": torch.cuda.is_initialized(),
            }
        )
        while True:
            if os.getppid() != expected_parent_pid:
                os._exit(0)
            message = conn.recv()
            if message.get("kind") == "stop":
                return
            if message.get("kind") != "plan":
                raise RuntimeError("unknown worker request")
            started = time.perf_counter()
            candidate = eplb_planner.plan_eplb_candidate(
                message["global_load"], message["current_placement"], **message["settings"]
            )
            conn.send(
                {
                    "kind": "result",
                    "job_id": message["job_id"],
                    "candidate": candidate,
                    "compute_ms": (time.perf_counter() - started) * 1000.0,
                }
            )
    except BaseException as exc:
        try:
            conn.send({"kind": "error", "message": f"{type(exc).__name__}: {exc}"})
        except BaseException:
            pass
    finally:
        conn.close()
