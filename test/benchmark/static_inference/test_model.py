import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

import static_benchmark
from static_benchmark import main


class _FakeDraftModel:
    def __init__(self, kvargs):
        self.kvargs = kvargs


def _mtp_args(model_dir):
    return SimpleNamespace(
        mtp_mode="vanilla_with_att",
        mtp_draft_model_dir=[model_dir],
        disable_chunked_prefill=False,
    )


def _main_kvargs():
    return {
        "load_way": "HF",
        "max_req_num": 4,
        "max_seq_length": 128,
        "data_type": "float16",
        "graph_max_batch_size": 1,
        "graph_max_len_in_batch": 128,
        "disable_cudagraph": True,
        "mem_fraction": 0.9,
        "batch_max_tokens": None,
        "quant_type": None,
        "quant_cfg": None,
        "expert_dtype": None,
    }


def _main_model():
    return SimpleNamespace(mem_manager=SimpleNamespace(size=1024))


def test_static_benchmark_accepts_qwen35_mtp_draft(monkeypatch):
    monkeypatch.setattr(
        static_benchmark.PretrainedConfig,
        "get_config_dict",
        lambda _: ({"model_type": "qwen3_5"}, None),
    )
    monkeypatch.setattr(static_benchmark, "Qwen3_5MTPModel", _FakeDraftModel, raising=False)

    draft_models = static_benchmark.init_mtp_draft_models(_mtp_args("/draft"), _main_kvargs(), _main_model())

    assert len(draft_models) == 1
    assert isinstance(draft_models[0], _FakeDraftModel)
    assert draft_models[0].kvargs["weight_dir"] == "/draft"


def test_static_benchmark_accepts_qwen35_moe_text_mtp_draft(monkeypatch):
    monkeypatch.setattr(
        static_benchmark.PretrainedConfig,
        "get_config_dict",
        lambda _: ({"model_type": "qwen3_5_moe_text"}, None),
    )
    monkeypatch.setattr(static_benchmark, "Qwen3_5MoeMTPModel", _FakeDraftModel, raising=False)

    draft_models = static_benchmark.init_mtp_draft_models(_mtp_args("/draft-moe"), _main_kvargs(), _main_model())

    assert len(draft_models) == 1
    assert isinstance(draft_models[0], _FakeDraftModel)
    assert draft_models[0].kvargs["weight_dir"] == "/draft-moe"


if __name__ == "__main__":
    main()
