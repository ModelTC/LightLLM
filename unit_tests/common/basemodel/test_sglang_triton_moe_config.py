from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl import (
    triton_impl,
)


def test_glm5_decode_uses_measured_h100_moe_config(monkeypatch):
    triton_impl._get_sglang_triton_moe_configs.cache_clear()
    monkeypatch.setattr(triton_impl.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(triton_impl.torch.cuda, "get_device_name", lambda _: "NVIDIA H100 80GB HBM3")

    up_config, down_config = triton_impl._get_sglang_triton_moe_configs(
        (289, 512, 4096),
        (289, 4096, 256),
        9,
        False,
        24,
    )

    assert up_config["BLOCK_SIZE_M"] == 16
    assert up_config["BLOCK_SIZE_N"] == 64
    assert up_config["num_stages"] == 3
    assert down_config["BLOCK_SIZE_M"] == up_config["BLOCK_SIZE_M"]
    assert down_config["BLOCK_SIZE_N"] == 64
    assert down_config["num_stages"] == 2


def test_glm5_draft_uses_measured_h100_down_config(monkeypatch):
    triton_impl._get_sglang_triton_moe_configs.cache_clear()
    monkeypatch.setattr(triton_impl.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(triton_impl.torch.cuda, "get_device_name", lambda _: "NVIDIA H100 80GB HBM3")

    up_config, down_config = triton_impl._get_sglang_triton_moe_configs(
        (289, 512, 4096),
        (289, 4096, 256),
        9,
        False,
        8,
    )

    assert up_config["BLOCK_SIZE_M"] == 16
    assert down_config["GROUP_SIZE_M"] == 8
    assert down_config["num_stages"] == 3


def test_glm5_prefill_uses_measured_h100_moe_config(monkeypatch):
    triton_impl._get_sglang_triton_moe_configs.cache_clear()
    monkeypatch.setattr(triton_impl.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(triton_impl.torch.cuda, "get_device_name", lambda _: "NVIDIA H100 80GB HBM3")

    (up_config, down_config) = triton_impl._get_sglang_triton_moe_configs(
        (289, 512, 4096),
        (289, 4096, 256),
        9,
        True,
        8192,
    )

    assert up_config["BLOCK_SIZE_M"] == 64
    assert up_config["BLOCK_SIZE_N"] == 128
    assert up_config["GROUP_SIZE_M"] == 64
    assert down_config["BLOCK_SIZE_M"] == up_config["BLOCK_SIZE_M"]
    assert down_config["BLOCK_SIZE_N"] == 128
    assert down_config["GROUP_SIZE_M"] == 8


def test_glm5_large_prefill_uses_independent_down_tile(monkeypatch):
    triton_impl._get_sglang_triton_moe_configs.cache_clear()
    monkeypatch.setattr(triton_impl.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(triton_impl.torch.cuda, "get_device_name", lambda _: "NVIDIA H100 80GB HBM3")

    up_config, down_config = triton_impl._get_sglang_triton_moe_configs(
        (289, 512, 4096),
        (289, 4096, 256),
        9,
        True,
        65536,
    )

    assert up_config["BLOCK_SIZE_N"] == 128
    assert up_config["GROUP_SIZE_M"] == 64
    assert down_config["BLOCK_SIZE_M"] == up_config["BLOCK_SIZE_M"]
    assert down_config["BLOCK_SIZE_N"] == 64
    assert down_config["GROUP_SIZE_M"] == 32


def test_non_h100_has_no_sglang_override(monkeypatch):
    triton_impl._get_sglang_triton_moe_configs.cache_clear()
    monkeypatch.setattr(triton_impl.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(triton_impl.torch.cuda, "get_device_name", lambda _: "NVIDIA H20")

    assert (
        triton_impl._get_sglang_triton_moe_configs(
            (289, 512, 4096),
            (289, 4096, 256),
            9,
            False,
            24,
        )
        is None
    )
