import argparse
import base64
import os
import shutil
import sys

from lightllm.common.basemodel.layer_weights.weight_crypto import (
    encrypt_file,
    get_key,
    is_encrypted,
)

WEIGHT_EXTENSIONS = (".safetensors", ".bin")


def _cmd_gen_key() -> int:
    key = os.urandom(32)
    encoded = base64.b64encode(key).decode("ascii")
    print(f"LIGHTLLM_WEIGHT_KEY={encoded}")
    return 0


def _cmd_encrypt(src: str, dst: str) -> int:
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    if not os.path.isdir(src):
        print(f"error: source dir does not exist: {src}", file=sys.stderr)
        return 1
    if os.path.abspath(dst) == src:
        print("error: --dst must differ from --src", file=sys.stderr)
        return 1

    key = get_key()
    if key is None:
        print(
            "error: LIGHTLLM_WEIGHT_KEY is not set. "
            "Generate one with: python -m lightllm.tools.encrypt_model --gen-key",
            file=sys.stderr,
        )
        return 1

    os.makedirs(dst, exist_ok=True)
    n_encrypted = 0
    n_copied = 0
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        out_dir = os.path.join(dst, rel) if rel != "." else dst
        os.makedirs(out_dir, exist_ok=True)
        for name in files:
            src_file = os.path.join(root, name)
            dst_file = os.path.join(out_dir, name)
            if name.endswith(WEIGHT_EXTENSIONS):
                if is_encrypted(src_file):
                    print(f"skip (already encrypted): {src_file}")
                    shutil.copy2(src_file, dst_file)
                    n_copied += 1
                    continue
                print(f"encrypt: {src_file} -> {dst_file}")
                encrypt_file(src_file, dst_file, key)
                n_encrypted += 1
            else:
                shutil.copy2(src_file, dst_file)
                n_copied += 1
    print(f"done: encrypted={n_encrypted} copied={n_copied} dst={dst}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Encrypt a LightLLM model directory with AES-256-GCM."
    )
    parser.add_argument(
        "--gen-key",
        action="store_true",
        help="Print a fresh random base64 AES-256 key and exit.",
    )
    parser.add_argument("--src", help="Plaintext model directory to encrypt.")
    parser.add_argument("--dst", help="Output directory for encrypted model.")
    args = parser.parse_args()

    if args.gen_key:
        return _cmd_gen_key()
    if not args.src or not args.dst:
        parser.error("--src and --dst are required unless --gen-key is passed")
    return _cmd_encrypt(args.src, args.dst)


if __name__ == "__main__":
    sys.exit(main())
