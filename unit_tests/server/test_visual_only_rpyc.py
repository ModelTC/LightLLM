import os
import tempfile
import threading
import time
import unittest

import rpyc
from rpyc import ThreadedServer, connect

from lightllm.server.visualserver.objs import visual_only_rpyc_config
from lightllm.server.visualserver.rpc_payload import RemoteEvent, pack_image_items, unpack_image_items


class ImageItem:
    def __init__(self, type, data):
        self._type = type
        self._data = data
        self.uuid = None
        self.token_id = None
        self.start_index_in_embed_cache = None
        self.token_num = None
        self.md5 = None
        self.start_idx = None
        self.grid_thwd = None
        self.image_w = 0
        self.image_h = 0
        self.extra_params = {}


class PayloadService(rpyc.Service):
    def exposed_accept(self, payload):
        images = unpack_image_items(payload, ImageItem, max_count=4, require_image_data=True)
        return tuple((image.md5, image.data_bytes) for image in images)

    def exposed_signal(self, event):
        event.set()


class MaliciousPayload:
    def __init__(self, marker_path):
        self.marker_path = marker_path

    def __reduce__(self):
        return (os.mkdir, (self.marker_path,))


def _start_server():
    server = ThreadedServer(
        PayloadService,
        hostname="127.0.0.1",
        port=0,
        protocol_config=visual_only_rpyc_config,
        auto_register=False,
    )
    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()
    for _ in range(100):
        try:
            client = connect("127.0.0.1", server.port, config=visual_only_rpyc_config)
            return server, thread, client
        except OSError:
            time.sleep(0.01)
    server.close()
    raise RuntimeError("local RPyC service did not start")


class VisualOnlyRpcTests(unittest.TestCase):
    def test_image_payload_round_trip_preserves_supported_fields(self):
        image = ImageItem("base64", "encoded source")
        image.uuid = "image-1"
        image.token_id = 17
        image.start_index_in_embed_cache = 23
        image.token_num = 2
        image.md5 = "0123456789abcdef0123456789abcdef"
        image.start_idx = 31
        image.grid_thwd = [1, 2, 3]
        image.image_w = 640
        image.image_h = 480
        image.extra_params = {"crop": [0, 1], "options": {"enabled": True}}
        image.data_bytes = b"image bytes"

        result = unpack_image_items(pack_image_items([image]), ImageItem)

        self.assertEqual(len(result), 1)
        restored = result[0]
        self.assertEqual(restored._type, image._type)
        self.assertEqual(restored._data, image._data)
        self.assertEqual(restored.uuid, image.uuid)
        self.assertEqual(restored.token_id, image.token_id)
        self.assertEqual(restored.start_index_in_embed_cache, image.start_index_in_embed_cache)
        self.assertEqual(restored.token_num, image.token_num)
        self.assertEqual(restored.md5, image.md5)
        self.assertEqual(restored.start_idx, image.start_idx)
        self.assertEqual(restored.grid_thwd, image.grid_thwd)
        self.assertEqual(restored.image_w, image.image_w)
        self.assertEqual(restored.image_h, image.image_h)
        self.assertEqual(restored.extra_params, image.extra_params)
        self.assertEqual(restored.data_bytes, image.data_bytes)

    def test_unpack_rejects_invalid_payload_shapes(self):
        for payload in ([], (2, ()), (1, (("invalid",),)), (1, (("none",),))):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    unpack_image_items(payload, ImageItem)

    def test_unpack_enforces_batch_limit(self):
        image = ImageItem("base64", "source")
        payload = pack_image_items([image, image])

        with self.assertRaisesRegex(ValueError, "configured limit"):
            unpack_image_items(payload, ImageItem, max_count=1)

    def test_secure_rpyc_payload_and_callback_do_not_unpickle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            marker = os.path.join(temp_dir, "unpickled")
            image = ImageItem("base64", "encoded source")
            image.uuid = "image-1"
            image.md5 = "0123456789abcdef0123456789abcdef"
            image.data_bytes = b"image bytes"
            server, thread, client = _start_server()
            try:
                self.assertEqual(
                    client.root.accept(pack_image_items([image])),
                    ((image.md5, image.data_bytes),),
                )

                event = threading.Event()
                client.root.signal(RemoteEvent(event))
                self.assertTrue(event.wait(1))

                attacker_config = dict(visual_only_rpyc_config, allow_pickle=True, allow_all_attrs=True)
                attacker = connect("127.0.0.1", server.port, config=attacker_config)
                try:
                    with self.assertRaises(Exception):
                        attacker.root.accept(MaliciousPayload(marker))
                finally:
                    attacker.close()

                self.assertFalse(os.path.exists(marker))
            finally:
                client.close()
                server.close()
                thread.join(1)

    def test_visual_only_rpyc_config_disables_unsafe_protocol_features(self):
        self.assertFalse(visual_only_rpyc_config["allow_pickle"])
        self.assertFalse(visual_only_rpyc_config["allow_all_attrs"])
        self.assertFalse(visual_only_rpyc_config["allow_setattr"])
        self.assertFalse(visual_only_rpyc_config["allow_delattr"])
