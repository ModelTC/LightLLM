from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import time
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp  # type: ignore[import-untyped]

# 随机 MP3 默认写入 asr_model_test/data，与「在该目录下 python -m http.server 8888」一致
_DEFAULT_AUDIO_SERVE_DIR = Path("/data0/wanzihao/asr_model_test/data")
# --real：使用该目录下固定四段真实语音（无 system 角色）
_REAL_AUDIO_FILENAMES: list[str] = ["ja.mp3", "ko.mp3", "yue.mp3", "zh.mp3"]


def format_local_time_ms(dt: datetime) -> str:
    return dt.strftime("%H:%M:%S.") + f"{dt.microsecond // 1000:03d}"


def _delta_has_first_token(delta: Any) -> bool:
    """与 asr_model_test `clients._openai_sse_delta_has_token` 一致（首 token / 首 reasoning 片）。"""
    if not isinstance(delta, dict):
        return False
    for key in ("content", "reasoning_content"):
        v = delta.get(key)
        if isinstance(v, str) and len(v) > 0:
            return True
    return False


async def _iter_sse_lines(stream_reader: Any) -> AsyncIterator[str]:
    """
    aiohttp 的 response.content 按 TCP 块迭代，不是按行。
    必须按 \\n 缓冲后再解析 `data: ...`，否则与 requests 侧行缓冲的 ttfb 不可比，且易解析失败、首字偏慢。
    """
    buf = b""
    async for chunk in stream_reader:
        if not chunk:
            continue
        buf += chunk
        while True:
            nl = buf.find(b"\n")
            if nl == -1:
                break
            raw = buf[:nl]
            buf = buf[nl + 1 :]
            line = raw.strip()
            if line:
                yield line.decode("utf-8", errors="replace")
    if buf.strip():
        yield buf.strip().decode("utf-8", errors="replace")


def _write_random_noise_mp3(path: Path, *, duration_sec: float, sample_rate: int = 44100) -> None:
    """写入一段随机噪声 MP3（需 pydub + ffmpeg）。"""
    try:
        from pydub import AudioSegment  # type: ignore[import-untyped]
    except ImportError as e:
        raise SystemExit("生成随机 MP3 需要 pydub，请先安装: pip install pydub\n" "并确保系统已安装 ffmpeg。\n" f"原始错误: {e}") from e

    num_samples = max(int(sample_rate * duration_sec), 1)
    raw_pcm = os.urandom(num_samples * 2)
    audio = AudioSegment(
        raw_pcm,
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )
    audio.export(str(path), format="mp3", bitrate="128k")


def audio_url_for_request(
    name: str,
    *,
    audio_dir: Path,
    audio_base: str,
    via: str,
) -> str:
    """via 为 \"url\" 时返回 HTTP URL；为 \"base64\" 时返回 data:audio/...;base64,...。"""
    if via == "url":
        return f"{audio_base}/{name}"
    if via != "base64":
        raise ValueError(f"unknown via: {via!r}")
    path = audio_dir / name
    raw = path.read_bytes()
    ext = Path(name).suffix.lower().lstrip(".") or "mp3"
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:audio/{ext};base64,{b64}"


def prepare_random_mp3_files(
    count: int = 1,
    *,
    duration_sec: float = 10.0,
    out_dir: Path | None = None,
) -> list[str]:
    """
    在 out_dir 下生成 count 个随机 MP3，返回仅文件名的列表（供 URL path 使用）。
    固定命名为 random_0.mp3 … 便于重复运行覆盖旧文件。
    """
    out = out_dir or _DEFAULT_AUDIO_SERVE_DIR
    out.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for i in range(count):
        name = f"random_{i}.mp3"
        names.append(name)
        _write_random_noise_mp3(out / name, duration_sec=duration_sec)
    return names


def _build_chat_completions_payload(
    audio_url: str,
    *,
    include_system: bool = True,
    ignore_eos: bool = True,
) -> dict:
    """LightLLM / vLLM 共用请求体（除顶层 model 外一致）。
    include_system=False 时不含 system 消息；ignore_eos=False 时不发送 ignore_eos 字段。
    """
    user_msg = {
        "role": "user",
        "content": [
            {"type": "audio_url", "audio_url": {"url": audio_url}},
            {
                "type": "text",
                "text": "请将这段音频完整准确地转换为文本。要求：1) 处理整个音频文件，不要截断；2) 每句话只转写一次，避免重复输出相同内容；3) 每句话之间用换行符分隔。",
            },
        ],
    }
    if include_system:
        messages: list[dict] = [
            {
                "role": "system",
                "content": "You are a helpful assistant, you can understand my audio and output what it says." * 180,
            },
            user_msg,
        ]
    else:
        messages = [user_msg]
    out: dict[str, Any] = {
        "messages": messages,
        "max_tokens": 50,
        "stream": True,
    }
    if ignore_eos:
        out["ignore_eos"] = True
    return out


def build_payload_lightllm(
    audio_url: str,
    model: str,
    *,
    include_system: bool = True,
    ignore_eos: bool = True,
) -> dict:
    """在共用体上增加顶层 model（LightLLM）。"""
    return {
        "model": model,
        **_build_chat_completions_payload(audio_url, include_system=include_system, ignore_eos=ignore_eos),
    }


def build_payload_vllm(
    audio_url: str,
    *,
    include_system: bool = True,
    ignore_eos: bool = True,
) -> dict:
    """与 LightLLM 相同，但不带顶层 model（vLLM 由服务端固定模型）。"""
    return _build_chat_completions_payload(audio_url, include_system=include_system, ignore_eos=ignore_eos)


def _input_text_chars_from_payload(payload: dict) -> int:
    n = 0
    for msg in payload.get("messages", []):
        content = msg.get("content")
        if isinstance(content, str):
            n += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    n += len(part.get("text", ""))
    return n


async def run_request_lightllm(session, url, payload, req_id):
    ttfb = None
    full_text = ""
    send_at = datetime.now()
    start_time = time.time()

    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error_msg = await response.text()
                return {
                    "req_id": req_id,
                    "send_ts": format_local_time_ms(send_at),
                    "ttfb": None,
                    "total_latency": time.time() - start_time,
                    "answer": f"HTTP Error {response.status}: {error_msg}",
                }
            async for line in _iter_sse_lines(response.content):
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        if _delta_has_first_token(delta):
                            if ttfb is None:
                                ttfb = time.time() - start_time
                        c = delta.get("content", "")
                        if isinstance(c, str) and c:
                            full_text += c
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        full_text = f"Error: {e}"

    total_latency = time.time() - start_time
    return {
        "req_id": req_id,
        "send_ts": format_local_time_ms(send_at),
        "ttfb": ttfb,
        "total_latency": total_latency,
        "answer": full_text,
    }


async def run_request_vllm(session, url, payload, req_id):
    start_time = time.time()
    ttfb = None
    full_text = ""
    usage = None
    input_text_chars = _input_text_chars_from_payload(payload)

    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error_msg = await response.text()
                return {
                    "req_id": req_id,
                    "ttfb": None,
                    "total_latency": time.time() - start_time,
                    "answer": f"HTTP Error {response.status}: {error_msg}",
                    "input_text_chars": input_text_chars,
                    "output_text_chars": 0,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                }
            async for line in _iter_sse_lines(response.content):
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        if isinstance(data.get("usage"), dict):
                            usage = data["usage"]
                        choices = data.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        if _delta_has_first_token(delta):
                            if ttfb is None:
                                ttfb = time.time() - start_time
                        content = delta.get("content", "")
                        if isinstance(content, str) and content:
                            full_text += content
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        full_text = f"Error: {e}"

    total_latency = time.time() - start_time
    return {
        "req_id": req_id,
        "ttfb": ttfb,
        "total_latency": total_latency,
        "answer": full_text,
        "input_text_chars": input_text_chars,
        "output_text_chars": len(full_text),
        "prompt_tokens": usage.get("prompt_tokens") if usage else None,
        "completion_tokens": usage.get("completion_tokens") if usage else None,
        "total_tokens": usage.get("total_tokens") if usage else None,
    }


def _print_results_lightllm(results: list[dict]) -> None:
    for res in results:
        print(f"\n{'='*40}")
        print(f"Request ID: {res['req_id']}")
        print(f"请求发送时间: {res['send_ts']}")
        print(f"TTFB (首字延迟): {res['ttfb']:.4f}s" if res["ttfb"] else "TTFB (首字延迟): N/A")
        print(f"Total Latency (总延迟): {res['total_latency']:.4f}s")
        print(f"Answer:\n{res['answer']}")
        print(f"{'='*40}")


def _linear_percentile(sorted_vals: list[float], p: float) -> float | None:
    """p ∈ [0, 100]，线性插值；sorted_vals 须已升序。"""
    if not sorted_vals:
        return None
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    k = (n - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, n - 1)
    w = k - lo
    return sorted_vals[lo] * (1 - w) + sorted_vals[hi] * w


def _format_latency_stats_line(
    label: str,
    values: list[float],
) -> str:
    if not values:
        return f"{label}: (无有效样本) N/A"
    s = sorted(values)
    mean = sum(values) / len(values)
    p50 = _linear_percentile(s, 50)
    p90 = _linear_percentile(s, 90)
    p99 = _linear_percentile(s, 99)
    return f"{label}: n={len(values)}  mean={mean:.4f}s  " f"p50={p50:.4f}s  p90={p90:.4f}s  p99={p99:.4f}s"


def print_latency_summary(results: list[dict], *, title: str) -> None:
    """对单次批次的 TTFB / total_latency 打印 mean 与 p50、p90、p99。"""
    totals = [float(r["total_latency"]) for r in results]
    ttfbs = [float(r["ttfb"]) for r in results if r.get("ttfb") is not None]

    print(f"\n{'─'*60}")
    print(f"延迟统计 — {title}")
    print(f"  总请求数: {len(results)}")
    print(f"  {_format_latency_stats_line('TTFB (首字)', ttfbs)}")
    print(f"  {_format_latency_stats_line('Total Latency', totals)}")
    print(f"{'─'*60}")


def _print_results_vllm(results: list[dict]) -> None:
    for res in results:
        print(f"\n{'='*40}")
        print(f"Request ID: {res['req_id']}")
        print(f"TTFB (首字延迟): {res['ttfb']:.4f}s" if res["ttfb"] else "TTFB (首字延迟): N/A")
        print(f"Total Latency (总延迟): {res['total_latency']:.4f}s")
        print("Input Length (输入长度): " f"text_chars={res['input_text_chars']}, " f"prompt_tokens={res['prompt_tokens']}")
        print(
            "Output Length (输出长度): "
            f"text_chars={res['output_text_chars']}, "
            f"completion_tokens={res['completion_tokens']}"
        )
        print(f"Total Tokens: {res['total_tokens']}")
        print(f"Answer:\n{res['answer']}")
        print(f"{'='*40}")


async def main(
    port: int,
    model: str,
    audio_host: str,
    audio_port: int,
    audio_filenames: list[str],
    audio_dir: Path,
    *,
    vllm: bool,
    audio_via: str,
    include_system: bool,
    ignore_eos: bool,
) -> None:
    url = f"http://localhost:{port}/v1/chat/completions"
    audio_base = f"http://{audio_host}:{audio_port}"

    mode = "vLLM" if vllm else "LightLLM"
    print(f"[{mode}] 将依次测试 {len(audio_filenames)} 个音频（每个音频固定 1 次请求）: {audio_filenames}")
    if audio_via == "url":
        print(f"音频经 HTTP URL: {audio_base}/")
    else:
        print("音频经 data:audio/...;base64,... 内联（不访问静态 HTTP 音频）")

    all_results: list[dict] = []
    for idx, name in enumerate(audio_filenames):
        audio_url = audio_url_for_request(
            name,
            audio_dir=audio_dir,
            audio_base=audio_base,
            via=audio_via,
        )
        if vllm:
            payload = build_payload_vllm(audio_url, include_system=include_system, ignore_eos=ignore_eos)
            runner = run_request_vllm
        else:
            payload = build_payload_lightllm(audio_url, model, include_system=include_system, ignore_eos=ignore_eos)
            runner = run_request_lightllm

        print(f"\n{'#'*60}\n[{idx + 1}/{len(audio_filenames)}] 音频: {name}\n{'#'*60}")
        print("发起 1 次请求...")

        async with aiohttp.ClientSession() as session:
            res = await runner(session, url, payload, 0)
            results = [res]

            all_results.extend(results)
            if vllm:
                _print_results_vllm(results)
            else:
                _print_results_lightllm(results)
            print_latency_summary(results, title=f"[{idx + 1}/{len(audio_filenames)}] {name}")

    if len(audio_filenames) > 1:
        print_latency_summary(all_results, title="全部音频累计")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="流式 chat/completions 音频测试。默认 LightLLM 请求体与输出；--vllm 使用原 test.py 格式。")
    parser.add_argument(
        "--vllm",
        action="store_true",
        help="使用 vLLM 的 JSON 请求体（无顶层 model，其余与 LightLLM 一致）与终端输出格式",
    )
    parser.add_argument(
        "--audio-via",
        choices=("url", "base64"),
        default="url",
        help="音频入参方式：url=HTTP 静态文件 URL（默认）；base64=data:audio/...;base64,... 内联",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=None,
        help="服务端端口（默认 8000）",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="default",
        help="LightLLM 时 JSON 中的 model 字段（--vllm 时不发送顶层 model）",
    )
    parser.add_argument("--audio-host", type=str, default="127.0.0.1", help="静态音频 HTTP 主机")
    parser.add_argument("--audio-port", type=int, default=8888, help="静态音频 HTTP 端口")
    parser.add_argument("--random-audio-count", type=int, default=1, help="测试前生成的随机 MP3 个数")
    parser.add_argument("--random-audio-duration", type=float, default=10.0, help="每个随机 MP3 时长（秒）")
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=_DEFAULT_AUDIO_SERVE_DIR,
        help="随机 MP3 输出目录（须与静态 HTTP 根目录一致）；--real 时从此目录读取 ja/ko/yue/zh.mp3",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="使用 audio-dir 下 ja.mp3、ko.mp3、yue.mp3、zh.mp3 四段真实音频，不生成随机 MP3；请求不含 system 消息，且不发送 ignore_eos",
    )
    args = parser.parse_args()
    if args.port is None:
        args.port = 8000
    audio_dir = args.audio_dir.resolve()
    include_system = not args.real
    ignore_eos = not args.real
    if args.real:
        audio_filenames = _REAL_AUDIO_FILENAMES.copy()
        missing = [n for n in audio_filenames if not (audio_dir / n).is_file()]
        if missing:
            raise SystemExit(
                f"--real 需要在 {audio_dir} 下存在: {', '.join(_REAL_AUDIO_FILENAMES)}。\n" f"缺失文件: {', '.join(missing)}"
            )
        print(f"[--real] 使用真实音频（无 system、无 ignore_eos）: {audio_filenames}")
    else:
        print(f"正在生成 {args.random_audio_count} 个随机音频 → {audio_dir} " f"(每段 {args.random_audio_duration}s) ...")
        audio_filenames = prepare_random_mp3_files(
            args.random_audio_count,
            duration_sec=args.random_audio_duration,
            out_dir=audio_dir,
        )
        print("生成完成:", audio_filenames)
    if args.audio_via == "url":
        print("请确保静态 HTTP 根目录为该文件夹（示例）:\n" f"  cd {audio_dir} && python3 -m http.server {args.audio_port}")
    else:
        print("已选择 --audio-via base64，无需为音频单独起 HTTP 服务。")

    asyncio.run(
        main(
            args.port,
            args.model,
            args.audio_host,
            args.audio_port,
            audio_filenames,
            audio_dir,
            vllm=args.vllm,
            audio_via=args.audio_via,
            include_system=include_system,
            ignore_eos=ignore_eos,
        )
    )
