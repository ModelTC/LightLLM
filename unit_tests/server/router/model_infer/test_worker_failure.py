import subprocess
import sys
import textwrap


def test_inference_thread_failure_terminates_rpc_worker():
    child = textwrap.dedent(
        """
        import threading
        from types import SimpleNamespace
        import torch
        from lightllm.server.router.model_infer import model_rpc
        from lightllm.server.router.model_infer.mode_backend.overlap_events import OverlapEventManager
        from lightllm.server.router.model_infer.mode_backend.chunked_prefill import impl

        model_rpc.graceful_registry = lambda *args: None
        model_rpc.start_parent_check_thread = lambda: None
        model_rpc.get_unique_server_name = lambda: "failure-test"
        model_rpc.ModelRpcServer = lambda *args: None
        stream = SimpleNamespace(wait_stream=lambda previous: None)
        torch.cuda.set_device = lambda device: None
        torch.cuda.current_stream = lambda: stream
        impl.g_infer_context.get_overlap_stream = lambda: stream

        def fail_forward(**kwargs):
            print("FORWARD_FAILURE_REACHED", flush=True)
            raise RuntimeError("injected model forward failure")

        backend = SimpleNamespace(
            overlap_event_manager=OverlapEventManager(),
            support_overlap=True,
            _try_read_new_reqs=lambda: None,
            _get_classed_reqs=lambda **kwargs: ([], [], 0),
            classed_req_no_decode=False,
            classed_req_strict_prefill=False,
            control_state_machine=SimpleNamespace(
                try_recover_paused_reqs=lambda: False,
                select_run_way=lambda **kwargs: SimpleNamespace(is_prefill=lambda: False, is_decode=lambda: True),
            ),
            decode=fail_forward,
            logger=SimpleNamespace(exception=lambda message: None),
        )

        class Server:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                threading.Thread(
                    target=lambda: impl.ChunkedPrefillBackend.infer_loop(backend),
                    name="failing-inference", daemon=True,
                ).start()
                threading.Event().wait(20)
                raise RuntimeError("RPC worker survived failed inference thread")

        model_rpc.ThreadedServer = Server
        model_rpc._init_env(None, 0, 0, 1, None, None, threading.Event())
        """
    )
    result = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=30)
    assert "FORWARD_FAILURE_REACHED" in result.stdout, result.stderr
    assert result.returncode == 1, result.stderr
    assert "fatal background thread failure in failing-inference" in result.stdout + result.stderr
    assert "RPC worker survived" not in result.stdout + result.stderr
