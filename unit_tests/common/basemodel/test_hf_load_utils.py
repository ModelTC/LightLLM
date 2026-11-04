import multiprocessing.synchronize

import pytest
import torch
from safetensors.torch import save_file
from tqdm import tqdm

from lightllm.common.basemodel.layer_weights import hf_load_utils


@pytest.mark.parametrize("load_failure", [False, True])
def test_threaded_checkpoint_loading_does_not_create_process_semaphores(tmp_path, monkeypatch, load_failure):
    loaded = {}

    class Weights:
        data_type_ = torch.float16

        def load_hf_weights(self, weights):
            if load_failure and "layer1" in weights:
                raise RuntimeError("invalid checkpoint shard")
            loaded.update(weights)

    def unexpected_semaphore(*args, **kwargs):
        raise AssertionError("threaded checkpoint loading must not create process semaphores")

    for index in range(3):
        save_file(
            {f"layer{index}": torch.tensor([index], dtype=torch.float16)}, tmp_path / f"weights-{index}.safetensors"
        )
    monkeypatch.delattr(tqdm, "_lock", raising=False)
    monkeypatch.setattr(multiprocessing.synchronize.SemLock, "__init__", unexpected_semaphore)
    monkeypatch.setattr(hf_load_utils, "get_current_device", lambda: torch.device("cpu"))
    monkeypatch.setenv("LOADWORKER", "2")
    if load_failure:
        with pytest.raises(RuntimeError, match="invalid checkpoint shard"):
            hf_load_utils.load_hf_weights(torch.float16, str(tmp_path), pre_post_layer=Weights())
        return
    hf_load_utils.load_hf_weights(torch.float16, str(tmp_path), pre_post_layer=Weights())
    assert set(loaded) == {"layer0", "layer1", "layer2"}
    for index in range(3):
        torch.testing.assert_close(loaded[f"layer{index}"], torch.tensor([index], dtype=torch.float16))


@pytest.mark.parametrize("format", ["safetensors", "bin", "dict"])
def test_checkpoint_names_follow_the_loader_selection_before_allocation(tmp_path, monkeypatch, format):
    if format == "dict":
        assert hf_load_utils.checkpoint_weight_names("does-not-exist", {"in_memory": object()}) == {"in_memory"}
        return
    for i in range(2):
        values = {f"layer{i}": torch.ones(2)}
        if format == "safetensors":
            save_file(values, tmp_path / f"model-{i}.safetensors")
        else:
            torch.save(values, tmp_path / f"model-{i}.bin")
    if format == "safetensors":
        torch.save({"ignored_bin": torch.ones(1)}, tmp_path / "old.bin")
        monkeypatch.setattr(hf_load_utils.utils.PetrelHelper, "load", lambda *a, **kw: pytest.fail("unused bin"))
    else:
        load = hf_load_utils.utils.PetrelHelper.load

        def meta_load(*args, **kwargs):
            result = load(*args, **kwargs)
            assert all(tensor.is_meta for tensor in result.values())
            return result

        monkeypatch.setattr(hf_load_utils.utils.PetrelHelper, "load", meta_load)
    assert hf_load_utils.checkpoint_weight_names(str(tmp_path)) == {"layer0", "layer1"}
