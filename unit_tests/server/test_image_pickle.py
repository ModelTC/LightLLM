import asyncio
import base64
import pickle
from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

from lightllm.server import multimodal_params
from lightllm.server.multimodal_params import ImageItem, MultimodalParams


def _image_bytes(image_format="PNG"):
    buffer = BytesIO()
    Image.new("RGB", (64, 48), "green").save(buffer, format=image_format)
    return buffer.getvalue()


@pytest.mark.parametrize("protocol", [4, 5])
@pytest.mark.parametrize("image_format", ["JPEG", "PNG"])
@pytest.mark.parametrize("resize_mode", ["within-budget", "resized", "disabled"])
def test_preloaded_images_are_sent_once_and_survive_retries_and_forwarding(
    monkeypatch, protocol, image_format, resize_mode
):
    async def run():
        monkeypatch.setattr(
            multimodal_params,
            "get_env_start_args",
            lambda: SimpleNamespace(
                max_image_pixels=4096 if resize_mode == "within-budget" else 256,
                disable_image_resize=resize_mode == "disabled",
            ),
        )
        original = base64.b64encode(_image_bytes(image_format)).decode("ascii")
        source = MultimodalParams(images=[{"type": "base64", "data": original}])
        await source.verify_and_preload(None)
        image = source.images[0]
        image.token_id, image.token_num = 2 ** 40, 32
        image.grid_thwd, image.start_idx = [1, 2, 3], 5
        image.extra_params = {"max_slice_nums": 4}
        verified = image._preload_data

        # Default object serialization is the previous wire format and remains readable.
        with monkeypatch.context() as patch:
            patch.delattr(ImageItem, "__getstate__")
            legacy_payload = pickle.dumps(source, protocol=protocol)
        legacy = pickle.loads(legacy_payload)
        assert legacy.images[0]._data == original
        assert legacy.images[0].read() == verified

        packed = pickle.dumps(source, protocol=protocol)
        assert original.encode("ascii") not in packed
        assert len(legacy_payload) - len(packed) >= len(original)
        received = pickle.loads(packed)
        assert received.images[0]._data is None
        assert received.images[0]._preload_data == verified
        assert received.images[0].to_dict() == image.to_dict()
        assert received.images[0].extra_params == image.extra_params
        assert (received.images[0].image_w, received.images[0].image_h) == (image.image_w, image.image_h)

        with monkeypatch.context() as patch:
            verify = Mock(side_effect=AssertionError("already-preloaded image was decoded again"))
            patch.setattr(multimodal_params, "_verify_image_bytes", verify)
            await received.verify_and_preload(None)
            verify.assert_not_called()

        # Multinode TP may deepcopy and pickle the received request again before reading it.
        relayed = pickle.loads(pickle.dumps(deepcopy(received), protocol=protocol))
        await relayed.verify_and_preload(None)
        assert relayed.images[0].read() == verified

        # HTTP forwarding uses the verified image, including any configured resizing.
        forwarded = received.to_origin_dict()
        assert base64.b64decode(forwarded["images"][0]["data"]) == verified
        http_request = MultimodalParams(**forwarded)
        await http_request.verify_and_preload(None)
        assert http_request.images[0].read() == verified
        assert received.images[0]._data is None
        assert received.images[0].read() == verified

        # Serializing or consuming a receiver's copy must preserve the Master's original request.
        assert source.to_origin_dict()["images"][0]["data"] == original
        assert image._preload_data == verified
        retried = pickle.loads(pickle.dumps(source, protocol=protocol))
        assert retried.images[0].read() == verified

    asyncio.run(run())


@pytest.mark.parametrize("image_type", ["base64", "url", "image_size"])
def test_unloaded_images_preserve_their_source_and_can_be_preloaded(monkeypatch, image_type):
    async def run():
        raw = _image_bytes()
        data = {
            "base64": base64.b64encode(raw).decode("ascii"),
            "url": "https://example.com/image.png",
            "image_size": [64, 48],
        }
        monkeypatch.setattr(
            multimodal_params,
            "get_env_start_args",
            lambda: SimpleNamespace(max_image_pixels=4096, disable_image_resize=False),
        )
        monkeypatch.setattr(multimodal_params, "fetch_resource", AsyncMock(return_value=raw))
        source = ImageItem(type=image_type, data=data[image_type])
        received = pickle.loads(pickle.dumps(source))
        assert vars(received) == vars(source)
        await received.preload(None)
        assert (received.image_w, received.image_h) == (64, 48)
        if image_type != "image_size":
            assert received.read() == raw

    asyncio.run(run())


def test_url_images_and_audio_keep_their_existing_serialized_state():
    source = MultimodalParams(
        images=[{"type": "url", "data": "https://example.com/image.png"}],
        audios=[{"type": "base64", "data": "original encoded audio"}],
    )
    source.images[0]._preload_data = b"downloaded image"
    source.audios[0]._preload_data = b"decoded PCM audio"
    received = pickle.loads(pickle.dumps(source))
    assert vars(received.images[0]) == vars(source.images[0])
    assert vars(received.audios[0]) == vars(source.audios[0])


def test_consumed_images_and_text_only_requests_stay_empty():
    image = ImageItem(type="base64", data="source")
    image._preload_data = b"verified image"
    assert image.read() == b"verified image"
    received = pickle.loads(pickle.dumps(image))
    assert received.to_origin_dict() == {"type": "base64", "data": None}
    assert received._preload_data is None
    empty = pickle.loads(pickle.dumps(MultimodalParams()))
    assert empty.images == [] and empty.audios == []
