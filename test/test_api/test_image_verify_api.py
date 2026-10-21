"""验证残缺图片在 OpenAI /v1/chat/completions 接口被前端拦截为 4xx。

启动 server：
    python -m lightllm.server.api_server --port 8000 --model_dir <your_vlm> --tp 1

运行：
    python test/test_api/test_image_verify_api.py
"""
import argparse
import base64
import os
from io import BytesIO

import requests
from PIL import Image


def make_jpeg(w=512, h=512) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (w, h), color=(255, 0, 0)).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def truncate(data: bytes, ratio: float = 0.3) -> bytes:
    return data[: int(len(data) * (1 - ratio))]


def data_url(img_bytes: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode("ascii")


def call(url: str, model: str, img_bytes: bytes):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url(img_bytes)}},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ],
        "max_tokens": 16,
        "temperature": 0.0,
    }
    return requests.post(f"{url}/v1/chat/completions", json=payload, timeout=30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="your_model_name")
    args = parser.parse_args()

    cases = [
        ("intact JPEG", make_jpeg(), 200),
        ("truncated JPEG", truncate(make_jpeg(1024, 1024), 0.3), 400),
        ("garbage bytes", os.urandom(4096), 400),
    ]

    for name, img, expected in cases:
        resp = call(args.url, args.model, img)
        ok = resp.status_code == expected
        print(f"[{'OK' if ok else 'FAIL'}] {name:18s} -> {resp.status_code} (expected {expected})")
        if not ok:
            print(f"       body: {resp.text[:200]}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
