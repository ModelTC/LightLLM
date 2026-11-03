"""Immutable, same-GPU weights owned by a process outside the serving launcher.

Only allocation/loading participates in this protocol. Inference uses ordinary
CUDA tensors and has no daemon calls. CUDA IPC rebuilding remains in the existing
torch_cuda_ipc utility; the daemon owns lifetime through process-bound leases.
"""

import ast
import hashlib
import json
import os
import stat
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from multiprocessing.connection import Client
from pathlib import Path

import torch
import triton
from filelock import FileLock

from lightllm.utils.dist_utils import get_current_device_id, get_global_rank
from lightllm.utils.log_utils import init_logger
from lightllm.utils.rl.torch_cuda_ipc import rebuild_cuda_ipc_tensor

logger = init_logger(__name__)
_preloading = ContextVar("preloading_gpu_weights", default=False)
_initializing_client = ContextVar("initializing_weight_cache_client", default=None)


@contextmanager
def preload_weights():
    token = _preloading.set(True)
    try:
        yield
    finally:
        _preloading.reset(token)


def is_preloading_weights():
    return _preloading.get()


def initializing_weight_client():
    return _initializing_client.get()


def validate_weight_cache(model):
    args = model.args
    if (
        model.config["model_type"] != "deepseek_v41"
        or model.weight_dict is not None
        or model.finetune_config is not None
        or args.enable_rl
        or args.enable_torch_memory_saver
        or args.enable_prefill_eplb
        or args.mtp_mode not in (None, "dspark")
        or args.hardware_platform != "cuda"
        or torch.cuda.get_device_capability(get_current_device_id())[0] != 9
    ):
        raise ValueError(
            "Resident weights support immutable CUDA SM90 V4.1 weights with optional DSpark; "
            "RL, EPLB and memory saver are unsupported"
        )
    selected = {model.quant_cfg.quant_type}
    for config in model.quant_cfg.quant_cfg.values():
        selected.update(config.values())
    unsupported = selected - {None, "none", "fp8w8a8-b128-deepgemm"}
    if unsupported:
        raise ValueError(f"Resident weights do not support selected quantization methods: {unsupported}")


@lru_cache(maxsize=1)
def _layout_digest():
    package = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), package / "utils/rl/torch_cuda_ipc.py"]
    for directory in (
        "common/basemodel/layer_weights",
        "common/quantization",
        "models/deepseek_v4/layer_weights",
        "models/deepseek_v41/layer_weights",
        "models/deepseek_v4_dspark/layer_weights",
        "models/deepseek_v41_dspark/layer_weights",
    ):
        paths.extend((package / directory).rglob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(path.read_bytes())
    # Changes to inference methods alone do not invalidate retained weights.
    # Include constructor/config/loading methods and class-level selectors.
    weight_methods = {
        "__init__",
        "_init_config",
        "_init_quant",
        "_init_weights",
        "load_weights",
        "_filter_hf_weight_keys",
        "_load_context_wkv_weights",
    }
    for model in ("deepseek_v4", "deepseek_v41", "deepseek_v4_dspark", "deepseek_v41_dspark"):
        tree = ast.parse((package / f"models/{model}/model.py").read_text())
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Import, ast.ImportFrom)):
                digest.update(ast.dump(node, include_attributes=False).encode())
            elif isinstance(node, ast.ClassDef):
                digest.update(node.name.encode())
                for item in node.body:
                    if isinstance(item, (ast.Assign, ast.AnnAssign)) or (
                        isinstance(item, ast.FunctionDef) and item.name in weight_methods
                    ):
                        digest.update(ast.dump(item, include_attributes=False).encode())
    # Engram lookup/hash kernels may change without changing the immutable table.
    tree = ast.parse((package / "models/deepseek_v41/engram.py").read_text())
    embedding = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "EngramEmbeddingWeight"
    )
    digest.update(ast.dump(embedding, include_attributes=False).encode())
    return digest.hexdigest()


def weight_cache_key(args):
    import deep_gemm

    directories = [args.model_dir, *(args.mtp_draft_model_dir or [])]
    checkpoints = []
    for directory in directories:
        root = Path(directory).resolve()
        index = root / "model.safetensors.index.json"
        if index.exists():
            index_data = json.loads(index.read_text())
            shards = sorted(set(index_data["weight_map"].values()))
        else:
            index_data = None
            shards = sorted(path.name for path in root.glob("*.safetensors"))
        if not shards:
            raise ValueError(f"Resident weights require a local safetensors checkpoint: {root}")
        checkpoints.append(
            dict(
                path=str(root),
                config=json.loads((root / "config.json").read_text()),
                index=index_data,
                shards=[(name, (root / name).stat().st_size, (root / name).stat().st_mtime_ns) for name in shards],
            )
        )
    options = {
        name: getattr(args, name)
        for name in (
            "tp",
            "dp",
            "nnodes",
            "data_type",
            "quant_type",
            "expert_dtype",
            "mtp_mode",
            "mtp_step",
            "enable_ep_moe",
            "ep_moe_backend",
            "enable_fused_shared_experts",
            "weight_version",
        )
    }
    custom_quant = Path(args.quant_cfg).read_text() if args.quant_cfg else None
    identity = dict(
        format=1,
        checkpoints=checkpoints,
        options=options,
        rank=get_global_rank(),
        gpu=str(torch.cuda.get_device_properties(get_current_device_id()).uuid),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        layout=_layout_digest(),
        triton=triton.__version__,
        deepgemm=hashlib.sha256(Path(deep_gemm.__file__).read_bytes()).hexdigest(),
        custom_quant=custom_quant,
        ue8m0=os.environ.get("LIGHTLLM_USE_UE8M0_SCALES", "0"),
    )
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def cache_socket_path(directory):
    uuid = str(torch.cuda.get_device_properties(get_current_device_id()).uuid)
    return str(Path(directory) / (hashlib.sha256(uuid.encode()).hexdigest()[:24] + ".sock"))


def cache_authkey(directory, create=False):
    root = Path(directory)
    if len(os.fsencode(str(root / ("0" * 24 + ".sock")))) >= 108:
        raise ValueError("Use a shorter local weight-cache directory for Unix sockets")
    if create:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = root.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise PermissionError(f"Weight cache directory must be private and owned by this user: {root}")
    path = root / "authkey"
    if create:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "wb") as file:
                file.write(os.urandom(32))
    info = path.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise PermissionError(f"Weight cache authkey must be private: {path}")
    return path.read_bytes()


def _has_tensor(value):
    from lightllm.common.quantization.quantize_method import WeightPack

    if isinstance(value, (torch.Tensor, WeightPack)):
        return True
    if isinstance(value, (list, tuple)):
        return any(_has_tensor(item) for item in value)
    if isinstance(value, dict):
        return any(_has_tensor(item) for item in value.values())
    return False


def export_weight_state(models):
    """Export only owner-declared weight modules, preserving packs, views and load flags."""
    from lightllm.common.quantization.quantize_method import WeightPack
    from torch.multiprocessing.reductions import reduce_tensor

    memo = {}

    def encode(value):
        if isinstance(value, (torch.Tensor, WeightPack)):
            if id(value) in memo:
                return memo[id(value)]
            node = {}
            memo[id(value)] = node
            if isinstance(value, torch.Tensor):
                if not value.is_cuda:
                    raise ValueError("Only CUDA weight tensors belong in the GPU cache")
                node.update(kind="tensor", handle=reduce_tensor(value))
            else:
                node["kind"] = "pack"
            node["attrs"] = {name: encode(item) for name, item in vars(value).items()}
            return node
        if isinstance(value, list):
            return [encode(item) for item in value]
        if isinstance(value, tuple):
            return tuple(encode(item) for item in value)
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        return value

    return {
        role: {
            name: (
                type(weight).__module__ + "." + type(weight).__qualname__,
                {
                    attr: encode(value)
                    for attr, value in vars(weight).items()
                    if _has_tensor(value) or attr == "_moe_weight_finalized"
                },
            )
            for name, weight in model.named_resident_weights().items()
        }
        for role, model in models.items()
    }


def rebuild_weight_state(state):
    from lightllm.common.quantization.quantize_method import WeightPack

    memo = {}

    def decode(value):
        if isinstance(value, dict) and value.get("kind") in ("tensor", "pack"):
            if id(value) not in memo:
                if value["kind"] == "tensor":
                    result = rebuild_cuda_ipc_tensor(value["handle"], get_current_device_id())
                else:
                    result = WeightPack.__new__(WeightPack)
                memo[id(value)] = result
                vars(result).update({name: decode(item) for name, item in value["attrs"].items()})
            return memo[id(value)]
        if isinstance(value, list):
            return [decode(item) for item in value]
        if isinstance(value, tuple):
            return tuple(decode(item) for item in value)
        if isinstance(value, dict):
            return {key: decode(item) for key, item in value.items()}
        return value

    return decode(state)


class WeightCacheClient:
    def __init__(self, args):
        directory = args.weight_cache_dir
        key = weight_cache_key(args)
        authkey = cache_authkey(directory)
        with FileLock(str(Path(directory) / "control.lock")):
            self.connection = Client(cache_socket_path(directory), family="AF_UNIX", authkey=authkey)
            self.connection.send(dict(op="acquire", key=key))
            response = self.connection.recv()
            if response["status"] != "ok":
                self.connection.close()
                raise RuntimeError(f"Resident weights rejected: {response['error']}")
            self.host_namespace = response["host_namespace"]
            self.owner_pid = response["pid"]
            self.state = rebuild_weight_state(response["state"])
            self.memory_bytes = response["memory_bytes"]
        threading.Thread(target=self._watch_owner, daemon=True, name="weight-owner-watchdog").start()
        logger.info(f"Resident weights attached to owner {self.owner_pid}: {self.memory_bytes} GPU bytes")

    def _watch_owner(self):
        try:
            self.connection.recv()
        except (EOFError, OSError):
            logger.error(f"Weight owner {self.owner_pid} disconnected; stopping worker before using invalid CUDA IPC")
            os._exit(1)

    @contextmanager
    def initialize_model(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        token = _initializing_client.set(self)
        try:
            with FakeTensorMode(allow_non_fake_inputs=True, allow_fallback_kernels=False):
                yield
        finally:
            _initializing_client.reset(token)

    def bind_model(self, model):
        from torch._subclasses.fake_tensor import FakeTensor
        from lightllm.common.quantization.quantize_method import WeightPack

        def check_layout(expected, actual, path):
            if isinstance(expected, torch.Tensor):
                if (
                    not isinstance(actual, torch.Tensor)
                    or isinstance(actual, FakeTensor)
                    or (expected.shape, expected.dtype, expected.stride())
                    != (actual.shape, actual.dtype, actual.stride())
                ):
                    raise RuntimeError(f"Resident weight layout differs: {path}")
            elif isinstance(expected, WeightPack):
                if not isinstance(actual, WeightPack):
                    raise RuntimeError(f"Resident weight pack differs: {path}")
                for attr, value in vars(expected).items():
                    check_layout(value, vars(actual).get(attr), f"{path}.{attr}")
            elif isinstance(expected, (list, tuple)) and _has_tensor(expected):
                if not isinstance(actual, type(expected)) or len(expected) != len(actual):
                    raise RuntimeError(f"Resident weight list differs: {path}")
                for index, (left, right) in enumerate(zip(expected, actual)):
                    check_layout(left, right, f"{path}.{index}")
            elif isinstance(expected, dict) and _has_tensor(expected):
                if not isinstance(actual, dict):
                    raise RuntimeError(f"Resident weight mapping differs: {path}")
                for key, value in expected.items():
                    check_layout(value, actual.get(key), f"{path}.{key}")

        role = "draft" if model.is_mtp_draft_model else "target"
        weights = model.named_resident_weights()
        cached = self.state[role]
        if weights.keys() != cached.keys():
            raise RuntimeError(f"Resident {role} weight-module names differ from the selected model")
        for name, weight in weights.items():
            expected_class, attrs = cached[name]
            if type(weight).__module__ + "." + type(weight).__qualname__ != expected_class:
                raise RuntimeError(f"Resident weight class differs: {name}")
            for attr, value in vars(weight).items():
                if _has_tensor(value):
                    check_layout(value, attrs.get(attr), f"{name}.{attr}")
            vars(weight).update(attrs)
            for value in vars(weight).values():
                if isinstance(value, FakeTensor):
                    raise RuntimeError(f"Resident weight was not rebound: {name}")
