"""Model code reaches attention kernels only through lightllm.common.attention.backends."""

import ast
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]

# Code that runs attention for a model: layer infers, infer states, models and their attention groups.
CALLER_GLOBS = (
    "lightllm/models/*/layer_infer/**/*.py",
    "lightllm/models/*/infer_struct.py",
    "lightllm/models/*/model.py",
    "lightllm/common/basemodel/attention_backend_manager.py",
    "lightllm/common/basemodel/layer_infer/**/*.py",
)

# Modules that compute attention over the KV cache.
ATTENTION_KERNEL_MODULES = re.compile(
    r"^("
    r"lightllm\.common\.attention\.kernels(\..*)?"
    r"|lightllm\.models\.[a-z0-9_]+\.triton_kernel\."
    r"(context_flashattention\w*|token_attention\w*|token_flashattention\w*|\w*flash_decoding\w*)"
    r"|flash_attn_interface|sgl_kernel\.flash_attn|flashinfer\.(decode|prefill|mla)|flash_mla|flash_kda|fla\.ops.*"
    r")$"
)

# The vision encoder attends within an image without a KV cache, outside the backend layer.
ALLOWED_CALLERS = {"lightllm/models/vit/layer_infer/transformer_layer_infer.py"}


def _is_type_checking_block(node: ast.AST) -> bool:
    test = getattr(node, "test", None)
    return isinstance(node, ast.If) and (
        (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING")
        or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")
    )


def _runtime_nodes(tree: ast.AST):
    """Walk the tree, skipping `if TYPE_CHECKING:` bodies: annotation-only imports create no runtime dependency."""
    stack = [tree]
    while stack:
        node = stack.pop()
        yield node
        children = list(ast.iter_child_nodes(node))
        if _is_type_checking_block(node):
            children = list(node.orelse)
        stack.extend(children)


def imported_modules(path: pathlib.Path):
    """Modules path imports at runtime, as (line, module) pairs."""
    tree = ast.parse(path.read_text(), filename=str(path))
    package = ".".join(path.relative_to(ROOT).with_suffix("").parts[:-1])
    for node in _runtime_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
                module = f"{base}.{node.module}" if node.module else base
            else:
                module = node.module
            yield node.lineno, module
            # "from package import module" imports a submodule.
            for alias in node.names:
                yield node.lineno, f"{module}.{alias.name}"


def caller_files():
    files = {path for pattern in CALLER_GLOBS for path in ROOT.glob(pattern)}
    return sorted(path for path in files if path.relative_to(ROOT).as_posix() not in ALLOWED_CALLERS)


def test_caller_globs_find_the_attention_callers():
    found = {path.relative_to(ROOT).as_posix() for path in caller_files()}
    for expected in (
        "lightllm/models/llama/layer_infer/transformer_layer_infer.py",
        "lightllm/models/deepseek2/layer_infer/transformer_layer_infer.py",
        "lightllm/models/qwen3next/layer_infer/transformer_layer_infer.py",
        "lightllm/common/basemodel/attention_backend_manager.py",
        "lightllm/common/basemodel/layer_infer/template/transformer_layer_infer_template.py",
    ):
        assert expected in found


@pytest.mark.parametrize(
    "module,is_kernel",
    [
        ("lightllm.common.attention.kernels.gqa.bf16.decode.flash_decoding", True),
        ("lightllm.common.attention.backends.gqa.triton", False),
        ("lightllm.models.llama.triton_kernel.token_attention_nopad_att1", True),
        ("lightllm.models.llama.triton_kernel.rotary_emb", False),
        ("lightllm.common.basemodel.triton_kernel.destindex_copy_kv", False),
        ("flash_attn_interface", True),
        ("flashinfer.mla", True),
    ],
)
def test_kernel_module_pattern(module, is_kernel):
    assert bool(ATTENTION_KERNEL_MODULES.match(module)) == is_kernel


def test_models_do_not_import_attention_kernels():
    offenders = [
        f"{path.relative_to(ROOT)}:{lineno} imports {module}"
        for path in caller_files()
        for lineno, module in imported_modules(path)
        if ATTENTION_KERNEL_MODULES.match(module)
    ]
    assert not offenders, "call attention through lightllm.common.attention.backends:\n" + "\n".join(offenders)


# The backend layer knows nothing about models at runtime: models and lightllm.common.basemodel import it, not
# the reverse. Annotations may name InferStateInfo under TYPE_CHECKING, the documented input of build. Generic
# Triton helpers that happen to live under basemodel/triton_kernel are allowed.
MODEL_SIDE_MODULES = re.compile(r"^lightllm\.(models|common\.basemodel)(\.|$)(?!triton_kernel(\.|$))")


@pytest.mark.parametrize(
    "module,is_model_side",
    [
        ("lightllm.models.llama.model", True),
        ("lightllm.common.basemodel", True),
        ("lightllm.common.basemodel.infer_struct", True),
        ("lightllm.common.basemodel.attention_backend_manager", True),
        ("lightllm.common.basemodel.triton_kernel.fa3_utils", False),
        ("lightllm.common.attention.backend", False),
    ],
)
def test_model_side_module_pattern(module, is_model_side):
    assert bool(MODEL_SIDE_MODULES.match(module)) == is_model_side


def test_backend_layer_does_not_import_model_side_code():
    files = sorted(ROOT.glob("lightllm/common/attention/**/*.py"))
    assert any(path.name == "selector.py" for path in files)
    offenders = [
        f"{path.relative_to(ROOT)}:{lineno} imports {module}"
        for path in files
        for lineno, module in imported_modules(path)
        if MODEL_SIDE_MODULES.match(module)
    ]
    assert not offenders, "lightllm.common.attention must not depend on models:\n" + "\n".join(offenders)


def test_type_checking_imports_are_not_runtime_dependencies():
    source = (
        "import typing\n"
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from lightllm.common.basemodel.infer_struct import InferStateInfo\n"
        "if typing.TYPE_CHECKING:\n"
        "    import lightllm.models.llama.model\n"
        "else:\n"
        "    import lightllm.common.basemodel.basemodel\n"
    )
    modules = set()
    for node in _runtime_nodes(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module)
    assert modules == {"typing", "lightllm.common.basemodel.basemodel"}
