import ast
import __future__
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_methods(path, class_name, method_names, optimize=0):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in method_names]
    module = ast.Module(body=[cls], type_ignores=[])
    namespace = {"logger": Mock()}
    exec(compile(module, path, "exec", flags=__future__.annotations.compiler_flag, optimize=optimize), namespace)
    return namespace[class_name]


@pytest.mark.parametrize("optimize", [0, 1, 2])
@pytest.mark.parametrize("prefix", [0, 64])
def test_cpu_cache_restore_allocates_before_read(optimize, prefix):
    cache_class = load_methods(
        "lightllm/server/router/model_infer/mode_backend/multi_level_kv_cache.py",
        "MultiLevelKvCacheModule",
        {"read_req_from_cpu"},
        optimize,
    )
    req = SimpleNamespace(cur_kv_len=prefix, hold_kv_len=prefix)
    indices = list(range(prefix))

    def allocate(count):
        indices.extend(range(req.hold_kv_len, req.hold_kv_len + count))
        req.hold_kv_len += count
        return indices[-count:]

    def get_indices(start, end):
        selected = indices[start:end]
        return SimpleNamespace(values=selected, numel=lambda: len(selected))

    req.kv_cache_alloc_need = Mock(side_effect=lambda length: length - req.hold_kv_len)
    req.alloc_kv_mem = Mock(side_effect=allocate)
    req.token_index = SimpleNamespace(get_gpu_token_indexes=get_indices)
    operator = SimpleNamespace(read_req=Mock(), get_page_layouts=lambda: [])
    cache = cache_class()
    cache.backend = SimpleNamespace(
        model=SimpleNamespace(att_state_manager=SimpleNamespace(get_cpu_cache_operator=lambda: operator)),
        is_hybrid_att_model=False,
    )
    cache.cpu_cache_client = SimpleNamespace(cpu_kv_cache_tensor=object())

    cache.read_req_from_cpu(req, [0], [256], 0)

    req.alloc_kv_mem.assert_called_once_with(256 - prefix)
    operator.read_req.assert_called_once()
    assert operator.read_req.call_args.kwargs["token_indices"].values == list(range(256))
    assert req.cur_kv_len == req.hold_kv_len == 256


@pytest.mark.parametrize("return_fail_index", [False, True])
@pytest.mark.parametrize("success", [False, True])
def test_native_batch_return_formats(monkeypatch, return_fail_index, success):
    copier = load_methods(
        "lightllm/common/cuda_wrapper.py",
        "PinnedH2DBatchCopier",
        {"_copy_on_stream", "_try_native_batch_h2d"},
    )
    copier._native_batch_available = True
    copier._protect_pinned_sources = Mock()
    error = 0 if success else SimpleNamespace(name="cudaErrorInvalidValue")
    result = (error, 0) if return_fail_index else (error,)
    runtime = SimpleNamespace(
        cudaMemcpyAttributes=SimpleNamespace,
        cudaMemcpySrcAccessOrder=SimpleNamespace(cudaMemcpySrcAccessOrderStream=1),
        cudaError_t=SimpleNamespace(cudaSuccess=0),
        cudaMemcpyBatchAsync=Mock(return_value=result),
    )
    monkeypatch.setitem(sys.modules, "cuda", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "cuda.bindings", SimpleNamespace(runtime=runtime))
    source = SimpleNamespace(data_ptr=lambda: 100, numel=lambda: 256, element_size=lambda: 4)
    target = SimpleNamespace(data_ptr=lambda: 200, copy_=Mock())
    stream = SimpleNamespace(cuda_stream=7)

    copier._copy_on_stream([target], [source], stream)

    runtime.cudaMemcpyBatchAsync.assert_called_once()
    assert runtime.cudaMemcpyBatchAsync.call_args.args[:4] == ([200], [100], [1024], 1)
    assert copier._native_batch_available is success
    if success:
        target.copy_.assert_not_called()
        copier._protect_pinned_sources.assert_called_once_with([source], stream)
    else:
        target.copy_.assert_called_once_with(source, non_blocking=True)
        copier._protect_pinned_sources.assert_not_called()
