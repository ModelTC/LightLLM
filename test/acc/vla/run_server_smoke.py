"""Send one deterministic request through the normal LightLLM VLA endpoint.

Start ``lightllm.server.api_server`` separately with a pi0/pi0.5 checkpoint;
this client intentionally tests the same HTTP/visualserver/router lifecycle as
production instead of constructing a second in-process VLA engine.
"""

import argparse
import base64
import io
import json
import time

import numpy as np
import requests
from PIL import Image


DOMAIN_INPUTS = {
    "libero": (8, 7, ("agentview", "wrist")),
    "droid": (
        8,
        8,
        ("observation/exterior_image_1_left", "observation/wrist_image_left"),
    ),
    # LightX2V publishes the Cosmos-oriented three-camera contract. A Pi/DROID
    # checkpoint consumes the exterior-left and wrist views; the other shoulder
    # view is intentionally ignored by the built-in robolab profile.
    "robolab": (
        8,
        8,
        (
            "wrist_cam",
            "over_shoulder_left_camera",
            "over_shoulder_right_camera",
        ),
    ),
    "aloha": (14, 14, ("cam_high", "cam_left_wrist", "cam_right_wrist")),
    "robotwin": (14, 14, ("head_camera", "left_camera", "right_camera")),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/vla/actions")
    parser.add_argument("--domain", choices=tuple(DOMAIN_INPUTS))
    parser.add_argument("--action-dim", type=int, default=None)
    parser.add_argument("--model-action-dim", type=int, default=32)
    parser.add_argument("--action-horizon", type=int, default=1)
    parser.add_argument("--num-denoise-steps", type=int, default=1)
    parser.add_argument("--prompt", default="pick up the block")
    parser.add_argument("--image-offset", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--timing-only", action="store_true")
    return parser.parse_args()


def image_payload(offset: int) -> str:
    values = np.arange(180 * 260 * 3, dtype=np.uint32)
    image = ((values + offset) % 256).astype(np.uint8).reshape(180, 260, 3)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=96)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def main():
    args = parse_args()
    if args.domain is None:
        state_dim = args.action_dim or 32
        action_dim = args.action_dim or 32
        camera_names = (
            "observation.images.base_0_rgb",
            "observation.images.left_wrist_0_rgb",
            "observation.images.right_wrist_0_rgb",
        )
    else:
        state_dim, domain_action_dim, camera_names = DOMAIN_INPUTS[args.domain]
        action_dim = args.action_dim or domain_action_dim
    payload = {
        "request_id": "normal-pipeline-smoke",
        "prompt": args.prompt,
        "images": {
            name: image_payload(args.image_offset + index * 17)
            for index, name in enumerate(camera_names)
        },
        "state": [0.0] * state_dim,
        "noise": [[[0.0] * args.model_action_dim for _ in range(args.action_horizon)]],
        "action_dim": action_dim,
        "action_horizon": args.action_horizon,
        "num_denoise_steps": args.num_denoise_steps,
        "timeout": args.timeout,
    }
    if args.domain is not None:
        payload["domain"] = args.domain
    started = time.perf_counter()
    response = requests.post(args.url, json=payload, timeout=args.timeout + 10)
    wall_ms = (time.perf_counter() - started) * 1000.0
    response.raise_for_status()
    body = response.json()
    actions = np.asarray(body["actions"], dtype=np.float32)
    assert actions.shape == (args.action_horizon, action_dim)
    assert np.isfinite(actions).all()
    assert body["finish_status"] == "finished"
    if args.timing_only:
        print(
            json.dumps(
                {
                    "wall_ms": wall_ms,
                    "action_shape": list(actions.shape),
                    "policy_timing": body["policy_timing"],
                    "finish_status": body["finish_status"],
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(body, indent=2))


if __name__ == "__main__":
    main()
