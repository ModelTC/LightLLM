"""用独立进程验证 atexit；所有信号只发送给本测试创建的进程。"""

import selectors
import signal
import subprocess
import sys
import textwrap

import pytest


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Exit codes and signals below target Linux")

PROBE_SCRIPT = textwrap.dedent(
    """
    import atexit
    import multiprocessing as mp
    import os
    import resource
    import signal
    import sys


    def record(marker_path, event):
        with open(marker_path, "a", encoding="utf-8") as output:
            output.write(event + "\\n")


    def cleanup(marker_path, interrupt=False):
        record(marker_path, "started")
        if interrupt:
            print("READY", flush=True)
            while True:
                signal.pause()
        record(marker_path, "finished")


    def worker(marker_path):
        atexit.register(cleanup, marker_path)
        record(marker_path, "worker_registered")


    def main():
        # SIGABRT/SIGSEGV 实验不生成 core dump。
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        mode, marker_path, signal_number = sys.argv[1:]
        signal_number = int(signal_number)
        signal.signal(signal.SIGINT, signal.default_int_handler)

        if mode.startswith("multiprocessing_"):
            context = mp.get_context(mode.removeprefix("multiprocessing_"))
            process = context.Process(target=worker, args=(marker_path,))
            process.start()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
                raise RuntimeError("multiprocessing worker failed to exit")
            assert process.exitcode == 0, process.exitcode
            return

        atexit.register(cleanup, marker_path, mode == "interrupt_cleanup")

        if mode in ("normal_return", "interrupt_cleanup"):
            return
        if mode == "sys_exit_0":
            sys.exit(0)
        if mode == "sys_exit_7":
            sys.exit(7)
        if mode == "unhandled_exception":
            raise RuntimeError("intentional test exception")
        if mode == "keyboard_interrupt":
            raise KeyboardInterrupt
        if mode == "os_exit_0":
            os._exit(0)
        if mode == "os_exit_7":
            os._exit(7)
        if mode == "abort":
            os.abort()
        if mode == "exec_replace":
            os.execv(sys.executable, [sys.executable, "-I", "-c", "pass"])

        if mode == "signal_sys_exit":
            signal.signal(signal_number, lambda signum, frame: sys.exit(0))
        elif mode == "signal_os_exit":
            signal.signal(signal_number, lambda signum, frame: os._exit(7))
        elif mode == "signal_os_default" or (mode == "signal_default" and signal_number != signal.SIGINT):
            if signal_number != signal.SIGKILL:
                signal.signal(signal_number, signal.SIG_DFL)
        elif mode != "signal_default":
            raise ValueError(mode)

        # 父进程收到 READY 后才发信号，确保回调和信号处理方式已注册。
        print("READY", flush=True)
        while True:
            signal.pause()


    if __name__ == "__main__":
        main()
    """
)


@pytest.fixture
def probe_script(tmp_path):
    script_path = tmp_path / "atexit_probe.py"
    script_path.write_text(PROBE_SCRIPT, encoding="utf-8")
    return script_path


def run_probe(probe_script, tmp_path, mode, signum=0):
    marker_path = tmp_path / "callback_events.txt"
    with subprocess.Popen(
        [sys.executable, "-I", str(probe_script), mode, str(marker_path), str(int(signum))],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            if signum:
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    assert selector.select(timeout=10), "Probe did not become ready"
                assert process.stdout.readline().strip() == "READY", "Probe exited before registering its callback"
                process.send_signal(signum)
            stdout, stderr = process.communicate(timeout=30)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)

    events = marker_path.read_text(encoding="utf-8").splitlines() if marker_path.exists() else []
    print(f"{mode} signal={int(signum)} returncode={process.returncode} events={events}")
    return process.returncode, events, stdout + stderr


@pytest.mark.parametrize(
    "mode,signum,expected_returncode,callback_completed",
    [
        pytest.param("normal_return", 0, 0, True, id="normal-return"),
        pytest.param("sys_exit_0", 0, 0, True, id="sys-exit-0"),
        pytest.param("sys_exit_7", 0, 7, True, id="sys-exit-nonzero"),
        pytest.param("unhandled_exception", 0, 1, True, id="unhandled-exception"),
        pytest.param("keyboard_interrupt", 0, -signal.SIGINT, True, id="unhandled-keyboard-interrupt"),
        pytest.param("signal_default", signal.SIGINT, -signal.SIGINT, True, id="sigint-python-default"),
        pytest.param("signal_os_default", signal.SIGINT, -signal.SIGINT, False, id="sigint-os-default"),
        pytest.param("signal_default", signal.SIGTERM, -signal.SIGTERM, False, id="sigterm-default"),
        pytest.param("signal_default", signal.SIGHUP, -signal.SIGHUP, False, id="sighup-default"),
        pytest.param("signal_default", signal.SIGQUIT, -signal.SIGQUIT, False, id="sigquit-default"),
        pytest.param("signal_default", signal.SIGSEGV, -signal.SIGSEGV, False, id="sigsegv-default"),
        pytest.param("signal_default", signal.SIGKILL, -signal.SIGKILL, False, id="sigkill"),
        pytest.param("signal_sys_exit", signal.SIGINT, 0, True, id="sigint-handler-sys-exit"),
        pytest.param("signal_sys_exit", signal.SIGTERM, 0, True, id="sigterm-handler-sys-exit"),
        pytest.param("signal_sys_exit", signal.SIGHUP, 0, True, id="sighup-handler-sys-exit"),
        pytest.param("signal_os_exit", signal.SIGTERM, 7, False, id="sigterm-handler-os-exit"),
        pytest.param("os_exit_0", 0, 0, False, id="os-exit-0"),
        pytest.param("os_exit_7", 0, 7, False, id="os-exit-nonzero"),
        pytest.param("abort", 0, -signal.SIGABRT, False, id="abort"),
        pytest.param("exec_replace", 0, 0, False, id="exec-replaces-interpreter"),
    ],
)
def test_atexit_on_process_exit(probe_script, tmp_path, mode, signum, expected_returncode, callback_completed):
    returncode, events, output = run_probe(probe_script, tmp_path, mode, signum)

    assert returncode == expected_returncode, output
    assert events == (["started", "finished"] if callback_completed else []), output


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 10), reason="This experiment records Python 3.10 multiprocessing behavior"
)
@pytest.mark.parametrize("start_method,callback_completed", [("spawn", True), ("fork", False)])
def test_atexit_in_multiprocessing_worker(probe_script, tmp_path, start_method, callback_completed):
    returncode, events, output = run_probe(probe_script, tmp_path, f"multiprocessing_{start_method}")

    assert returncode == 0, output
    # 本机 Python 3.10：spawn_main() 使用 sys.exit()，fork 的 _launch() 使用 os._exit()。
    expected_events = ["worker_registered"] + (["started", "finished"] if callback_completed else [])
    assert events == expected_events, output


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGKILL])
def test_signal_can_interrupt_atexit_callback(probe_script, tmp_path, signum):
    _returncode, events, output = run_probe(probe_script, tmp_path, "interrupt_cleanup", signum)

    assert events == ["started"], output
