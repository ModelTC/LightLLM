import base64
import io
import json
import os
import struct
from typing import Any, Dict, List, Optional

import torch

MAGIC = b"LLENC\x01\x00\x00"
VERSION = 1
NONCE_LEN = 12
TAG_LEN = 16
CHUNK_SIZE = 64 * 1024 * 1024  # 64 MiB plaintext per AES-GCM call (well under 2**32 limit)
# header layout: magic(8) | version(1) | reserved(3) | chunk_size(4) | plaintext_len(8)
HEADER_LEN = len(MAGIC) + 1 + 3 + 4 + 8

_KEY_ENV = "LIGHTLLM_WEIGHT_KEY"


def _aesgcm():
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as e:
        raise RuntimeError(
            "Encrypted model weights require the 'cryptography' package. "
            "Install it with: pip install cryptography"
        ) from e
    return AESGCM


def get_key() -> Optional[bytes]:
    raw = os.environ.get(_KEY_ENV)
    if not raw:
        return None
    try:
        key = base64.b64decode(raw, validate=True)
    except Exception as e:
        raise RuntimeError(f"{_KEY_ENV} must be valid base64: {e}") from e
    if len(key) != 32:
        raise RuntimeError(f"{_KEY_ENV} must decode to 32 bytes (AES-256), got {len(key)}")
    return key


def is_encrypted(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(len(MAGIC))
    except OSError:
        return False
    return head == MAGIC


def _pack_header(plaintext_len: int) -> bytes:
    return (
        MAGIC
        + bytes([VERSION])
        + b"\x00\x00\x00"
        + struct.pack(">I", CHUNK_SIZE)
        + struct.pack(">Q", plaintext_len)
    )


def _parse_header(header: bytes, path: str):
    if len(header) != HEADER_LEN or header[: len(MAGIC)] != MAGIC:
        raise RuntimeError(f"File '{path}' is not a valid encrypted weight file.")
    offset = len(MAGIC)
    version = header[offset]
    offset += 1
    if version != VERSION:
        raise RuntimeError(f"Unsupported weight-crypto version {version} in '{path}'.")
    offset += 3  # reserved
    (chunk_size,) = struct.unpack(">I", header[offset : offset + 4])
    offset += 4
    (plaintext_len,) = struct.unpack(">Q", header[offset : offset + 8])
    return chunk_size, plaintext_len


def decrypt_file_to_bytes(path: str, key: Optional[bytes] = None) -> bytes:
    if key is None:
        key = get_key()
    if key is None:
        raise RuntimeError(
            f"Encrypted weight file '{path}' detected but {_KEY_ENV} is not set."
        )
    AESGCM = _aesgcm()
    aead = AESGCM(key)
    with open(path, "rb") as f:
        header = f.read(HEADER_LEN)
        chunk_size, plaintext_len = _parse_header(header, path)
        if plaintext_len == 0:
            return b""
        out = bytearray(plaintext_len)
        write_off = 0
        remaining = plaintext_len
        chunk_idx = 0
        while remaining > 0:
            nonce = f.read(NONCE_LEN)
            tag = f.read(TAG_LEN)
            if len(nonce) != NONCE_LEN or len(tag) != TAG_LEN:
                raise RuntimeError(f"Truncated encrypted weight file '{path}'.")
            this_chunk = min(chunk_size, remaining)
            ct = f.read(this_chunk)
            if len(ct) != this_chunk:
                raise RuntimeError(f"Truncated encrypted weight file '{path}'.")
            try:
                pt = aead.decrypt(nonce, ct + tag, None)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to decrypt '{path}' at chunk {chunk_idx}: "
                    f"key mismatch or file tampered ({e})."
                ) from e
            out[write_off : write_off + this_chunk] = pt
            write_off += this_chunk
            remaining -= this_chunk
            chunk_idx += 1
    return bytes(out)


def encrypt_file(src_path: str, dst_path: str, key: bytes) -> None:
    if len(key) != 32:
        raise ValueError("AES-256 key must be exactly 32 bytes")
    AESGCM = _aesgcm()
    aead = AESGCM(key)
    plaintext_len = os.path.getsize(src_path)
    tmp_path = dst_path + ".partial"
    with open(src_path, "rb") as fi, open(tmp_path, "wb") as fo:
        fo.write(_pack_header(plaintext_len))
        remaining = plaintext_len
        while remaining > 0:
            this_chunk = min(CHUNK_SIZE, remaining)
            chunk = fi.read(this_chunk)
            if len(chunk) != this_chunk:
                raise RuntimeError(f"Short read on '{src_path}'.")
            nonce = os.urandom(NONCE_LEN)
            sealed = aead.encrypt(nonce, chunk, None)
            ct, tag = sealed[:-TAG_LEN], sealed[-TAG_LEN:]
            fo.write(nonce)
            fo.write(tag)
            fo.write(ct)
            remaining -= this_chunk
    os.replace(tmp_path, dst_path)


def filter_shards_by_prefix(
    weight_dir: str, prefix: str, extension: str
) -> Optional[List[str]]:
    """Return the list of shard filenames in ``weight_dir`` that contain at least
    one tensor whose key matches ``prefix``, by reading the HF sharded weight index.

    Returns None if no matching index file is present; callers should fall back to
    scanning every shard in that case.
    """
    if extension == ".safetensors":
        index_path = os.path.join(weight_dir, "model.safetensors.index.json")
    elif extension == ".bin":
        index_path = os.path.join(weight_dir, "pytorch_model.bin.index.json")
    else:
        return None
    if not os.path.isfile(index_path):
        return None
    try:
        with open(index_path, "r") as f:
            idx = json.load(f)
        wm = idx.get("weight_map", {})
    except (OSError, ValueError):
        return None
    shards = set()
    for k, shard in wm.items():
        if prefix in k:
            shards.add(shard)
    return sorted(shards)


def smart_load_safetensors(path: str) -> Dict[str, torch.Tensor]:
    from safetensors import safe_open
    from safetensors.torch import load as st_load_bytes

    if is_encrypted(path):
        plaintext = decrypt_file_to_bytes(path)
        return st_load_bytes(plaintext)
    with safe_open(path, "pt", "cpu") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def smart_load_torch(path: str, map_location: Any = "cpu") -> Dict[str, torch.Tensor]:
    if is_encrypted(path):
        plaintext = decrypt_file_to_bytes(path)
        buffer = io.BytesIO(plaintext)
        return torch.load(buffer, map_location=map_location, weights_only=True)
    import lightllm.utils.petrel_helper as petrel_utils

    return petrel_utils.PetrelHelper.load(path, map_location=map_location)
