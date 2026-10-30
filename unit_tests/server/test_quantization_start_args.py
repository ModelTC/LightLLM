import sys
from types import SimpleNamespace

import pytest
import yaml

from lightllm.common.quantization import Quantcfg
from lightllm.common.quantization import registry
from lightllm.server.api_cli import make_argument_parser
from lightllm.server.api_server import launch_server
from lightllm.server.core.objs.start_args_type import StartArgs


INVALID_METHOD = "triton-fp8a8w8-pertensor"


@pytest.mark.parametrize("option", ["--quant_type", "--vit_quant_type"])
def test_cli_rejects_unknown_quantization(option, capsys):
    with pytest.raises(SystemExit) as exc:
        make_argument_parser().parse_args([option, INVALID_METHOD])
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert option in error and INVALID_METHOD in error
    assert "fp8w8a8-pt-triton" in error


def test_lazy_method_is_imported_only_when_requested(monkeypatch):
    factory = registry.QuantMethodFactory()
    factory.register(["test", "alias"])("test_module.Method")
    imports = []

    class Method:
        pass

    def import_module(name, package):
        imports.append((name, package))
        return SimpleNamespace(Method=Method)

    monkeypatch.setattr(registry, "import_module", import_module)
    factory.validate("test")
    assert not imports
    assert isinstance(factory.get("test"), Method)
    assert isinstance(factory.get("test"), Method)
    assert isinstance(factory.get("alias"), Method)
    assert imports == [(".test_module", registry.__package__)] * 2


@pytest.mark.parametrize("option", ["quant_type", "vit_quant_type"])
def test_programmatic_launch_rejects_unknown_method_before_loading_startup(monkeypatch, option):
    monkeypatch.setitem(sys.modules, "lightllm.server.api_start", None)
    with pytest.raises(ValueError, match=f"--{option}.*{INVALID_METHOD}.*fp8w8a8-pt-triton"):
        launch_server(StartArgs(**{option: INVALID_METHOD}))


@pytest.mark.parametrize("option", ["quant_cfg", "vit_quant_cfg"])
@pytest.mark.parametrize("mixed", [False, True])
def test_launch_rejects_unknown_yaml_method_before_loading_startup(tmp_path, monkeypatch, option, mixed):
    path = tmp_path / "quant.yaml"
    data = {"quant_type": INVALID_METHOD}
    if mixed:
        data = {"quant_type": "none", "mix_bits": [{"name": "q_proj", "quant_type": INVALID_METHOD}]}
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setitem(sys.modules, "lightllm.server.api_start", None)
    with pytest.raises(ValueError) as exc:
        launch_server(StartArgs(**{option: str(path)}))
    error = str(exc.value)
    assert f"--{option}" in error and str(path) in error and INVALID_METHOD in error
    if mixed:
        assert "mix_bits[0]" in error


@pytest.mark.parametrize("option", ["quant_cfg", "vit_quant_cfg"])
@pytest.mark.parametrize(
    "data, fields",
    [
        ({}, ["quant_type"]),
        ([], ["mapping"]),
        ({"quant_type": "none", "mix_bits": None}, ["mix_bits"]),
        ({"quant_type": "none", "mix_bits": 1}, ["mix_bits"]),
        ({"quant_type": "none", "mix_bits": {}}, ["mix_bits"]),
        ({"quant_type": "none", "mix_bits": [None]}, ["mix_bits[0]", "mapping"]),
        ({"quant_type": "none", "mix_bits": [{"quant_type": "none"}]}, ["mix_bits[0]", "name"]),
        ({"quant_type": "none", "mix_bits": [{"name": "q_proj"}]}, ["mix_bits[0]", "quant_type"]),
        (
            {"quant_type": "none", "mix_bits": [{"name": [], "quant_type": "none"}]},
            ["mix_bits[0]", "name"],
        ),
        *[
            (
                {"quant_type": "none", "mix_bits": [{"name": "q_proj", "quant_type": "none", "layer_nums": nums}]},
                ["mix_bits[0]", "layer_nums"],
            )
            for nums in (None, 1, "0", {0: 1}, [0, "1"], [True], [-1])
        ],
    ],
)
def test_malformed_quant_config(tmp_path, monkeypatch, option, data, fields):
    path = tmp_path / "quant.yaml"
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setitem(sys.modules, "lightllm.server.api_start", None)
    with pytest.raises(ValueError) as exc:
        launch_server(StartArgs(**{option: str(path)}))
    error = str(exc.value)
    assert f"--{option}" in error and str(path) in error
    assert all(field in error for field in fields)


@pytest.mark.parametrize("option", ["--quant_type", "--vit_quant_type"])
@pytest.mark.parametrize("platform_first", [False, True])
def test_musa_rejects_cuda_methods(monkeypatch, option, platform_first):
    cli_args = ["--hardware_platform", "musa", option, "awq"]
    if not platform_first:
        cli_args = cli_args[2:] + cli_args[:2]
    args = StartArgs(**vars(make_argument_parser().parse_args(cli_args)))
    monkeypatch.setitem(sys.modules, "lightllm.server.api_start", None)
    with pytest.raises(ValueError, match=f"{option}.*awq.*platform 'musa'"):
        launch_server(args)


@pytest.mark.parametrize("option", ["quant_cfg", "vit_quant_cfg"])
@pytest.mark.parametrize("mixed", [False, True])
def test_musa_rejects_cuda_config(tmp_path, monkeypatch, option, mixed):
    path = tmp_path / "quant.yaml"
    data = {"quant_type": "awq"}
    if mixed:
        data = {"quant_type": "none", "mix_bits": [{"name": "q_proj", "quant_type": "awq"}]}
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setitem(sys.modules, "lightllm.server.api_start", None)
    with pytest.raises(ValueError) as exc:
        launch_server(StartArgs(hardware_platform="musa", **{option: str(path)}))
    error = str(exc.value)
    assert f"--{option}" in error and str(path) in error and "platform 'musa'" in error
    if mixed:
        assert "mix_bits[0]" in error


@pytest.mark.parametrize("method, cli_method", [("none", "none"), ("musa-only", "musa-only"), ("none", "awq")])
def test_musa_methods_reach_startup(tmp_path, monkeypatch, method, cli_method):
    monkeypatch.setitem(registry.QUANTMETHODS._quant_methods, "musa-only", {"musa": object})
    path = tmp_path / "quant.yaml"
    path.write_text(yaml.safe_dump({"quant_type": method, "mix_bits": [{"name": "q_proj", "quant_type": method}]}))
    cli_args = ["--hardware_platform", "musa", "--quant_type", cli_method, "--vit_quant_type", cli_method]
    cli_args += ["--quant_cfg", str(path), "--vit_quant_cfg", str(path)]
    args = StartArgs(**vars(make_argument_parser().parse_args(cli_args)))
    dispatched = []
    start = SimpleNamespace(
        normal_or_p_d_start=dispatched.append,
        pd_master_start=None,
        pd_master_multi_worker_start=None,
        config_server_start=None,
        visual_only_start=None,
    )
    monkeypatch.setitem(sys.modules, "lightllm.server.api_start", start)
    monkeypatch.setattr("torch.multiprocessing.set_start_method", lambda *args: None)
    launch_server(args)
    assert dispatched == [args]


@pytest.mark.parametrize("option", ["quant_cfg", "vit_quant_cfg"])
@pytest.mark.parametrize("failure", ["missing", "unreadable"])
def test_quant_config_file_errors(tmp_path, monkeypatch, option, failure):
    path = tmp_path / "quant.yaml"
    if failure == "unreadable":
        path.write_text("quant_type: none\n")

        def deny_open(*args, **kwargs):
            raise PermissionError("permission denied")

        monkeypatch.setattr("lightllm.common.quantization.open", deny_open, raising=False)
    monkeypatch.setitem(sys.modules, "lightllm.server.api_start", None)
    with pytest.raises(ValueError) as exc:
        launch_server(StartArgs(**{option: str(path)}))
    assert f"--{option}" in str(exc.value) and str(path) in str(exc.value)
    assert isinstance(exc.value.__cause__.__cause__, OSError)


def test_quantcfg_rejects_unknown_method_before_constructing_layers():
    with pytest.raises(ValueError, match=INVALID_METHOD):
        Quantcfg({"n_layer": 1}, quant_type=INVALID_METHOD)


@pytest.mark.parametrize("expert_dtype", [None, "fp8"])
def test_mixed_config_preserves_quantization_precedence(tmp_path, expert_dtype):
    path = tmp_path / "quant.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "quant_type": "fp8w8a8-pt",
                "mix_bits": [{"name": "fused_moe", "quant_type": "none", "layer_nums": [0]}],
            }
        )
    )
    config = Quantcfg({"n_layer": 2, "expert_dtype": "fp8"}, INVALID_METHOD, str(path), expert_dtype=expert_dtype)
    assert config.get_quant_type(0, "q_proj") == "fp8w8a8-pt"
    assert config.get_quant_type(0, "fused_moe") == ("none" if expert_dtype is None else "fp8w8a8-b128-deepgemm")
    assert config.get_quant_type(1, "fused_moe") == "fp8w8a8-b128-deepgemm"


def test_no_quantization_still_loads_through_registry(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    method = Quantcfg({"n_layer": 1}).get_quant_method(0, "q_proj")
    assert method.method_name == "none"
    assert method.hf_quantization_config is None
