import os
import socket
import tempfile
import threading
import time
import unittest
import zlib
from unittest.mock import patch

import rpyc
from rpyc import connect
from rpyc.core.channel import Channel

from lightllm.server.visualserver.objs import VISUAL_ONLY_RPC_HOST, visual_only_rpyc_config
from lightllm.server.visualserver.rpc_payload import (
    MAX_RPC_FRAME_BYTES,
    RemoteEvent,
    notify_remote_event_async,
    pack_image_items,
    unpack_image_items,
)
from lightllm.server.visualserver.rpyc_server import VisualOnlyThreadedServer


class ImageItem:
    def __init__(self, type, data):
        """Create a lightweight image stand-in for payload tests."""
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
    """Expose the visual payload helpers through a local test server."""

    def exposed_accept(self, payload):
        """Validate a remote image payload and return its stable fields."""
        images = unpack_image_items(payload, ImageItem, max_count=4, require_image_data=True)
        return tuple((image.md5, image.data_bytes) for image in images)

    def exposed_signal(self, event):
        """Signal a client-owned event through the restricted protocol."""
        event.set()


class MaliciousPayload:
    def __init__(self, marker_path):
        """Store the path that an unsafe unpickler would create."""
        self.marker_path = marker_path

    def __reduce__(self):
        """Return a pickle reduction that would create the marker directory."""
        return (os.mkdir, (self.marker_path,))


def _start_server(max_message_size=None):
    """Start a loopback visual RPC server for transport tests."""
    protocol_config = dict(visual_only_rpyc_config)
    if max_message_size is not None:
        protocol_config["max_message_size"] = max_message_size
    server = VisualOnlyThreadedServer(
        PayloadService,
        hostname=VISUAL_ONLY_RPC_HOST,
        port=0,
        protocol_config=protocol_config,
        auto_register=False,
    )
    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()
    for _ in range(100):
        try:
            client = connect(VISUAL_ONLY_RPC_HOST, server.port, config=protocol_config)
            return server, thread, client
        except OSError:
            time.sleep(0.01)
    server.close()
    raise RuntimeError("local RPyC service did not start")


class VisualOnlyRpcTests(unittest.TestCase):
    """Exercise the serialized visual RPC contract and its transport limits."""

    def test_image_payload_round_trip_preserves_supported_fields(self):
        """Preserve supported image fields through pack and unpack."""
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
        """Reject unknown versions, malformed records, and invalid tags."""
        for payload in ([], (2, ()), (1, (("invalid",),)), (1, (("none",),))):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    unpack_image_items(payload, ImageItem)

    def test_unpack_enforces_batch_limit(self):
        """Enforce the configured image-count limit while decoding."""
        image = ImageItem("base64", "source")
        payload = pack_image_items([image, image])

        with self.assertRaisesRegex(ValueError, "configured limit"):
            unpack_image_items(payload, ImageItem, max_count=1)

    def test_pack_and_unpack_enforce_image_byte_limits(self):
        """Enforce per-image and aggregate byte limits on both RPC boundaries."""
        oversized = ImageItem("base64", "source")
        oversized.data_bytes = b"12345"
        oversized_payload = pack_image_items([oversized])
        with patch("lightllm.server.visualserver.rpc_payload.MAX_IMAGE_BYTES", 4):
            with self.assertRaisesRegex(ValueError, "per-image"):
                pack_image_items([oversized])
            with self.assertRaisesRegex(ValueError, "per-image"):
                unpack_image_items(oversized_payload, ImageItem)

        image_a = ImageItem("base64", "source")
        image_a.data_bytes = b"1234"
        image_b = ImageItem("base64", "source")
        image_b.data_bytes = b"5678"
        batch_payload = pack_image_items([image_a, image_b])
        with patch("lightllm.server.visualserver.rpc_payload.MAX_BATCH_IMAGE_BYTES", 7):
            with self.assertRaisesRegex(ValueError, "aggregate"):
                pack_image_items([image_a, image_b])
            with self.assertRaisesRegex(ValueError, "aggregate"):
                unpack_image_items(batch_payload, ImageItem)

    def test_remote_event_notification_does_not_block_the_caller(self):
        """Run a slow client callback on a daemon worker thread."""
        callback_started = threading.Event()
        callback_release = threading.Event()
        caller_finished = threading.Event()

        class SlowRemoteEvent:
            """Block until released to model a slow client connection."""

            def set(self):
                """Mark that notification began, then wait for the test to release it."""
                callback_started.set()
                callback_release.wait(2)

        def notify():
            """Dispatch the callback and report when dispatch has returned."""
            notify_remote_event_async(SlowRemoteEvent())
            caller_finished.set()

        threading.Thread(target=notify, daemon=True).start()
        try:
            self.assertTrue(callback_started.wait(1))
            self.assertTrue(caller_finished.wait(0.5))
        finally:
            callback_release.set()

    def test_server_binds_loopback_and_rejects_oversized_frame_headers(self):
        """Bind only to loopback and reject an oversized frame before its body."""
        server, thread, client = _start_server()
        try:
            self.assertEqual(server.listener.getsockname()[0], VISUAL_ONLY_RPC_HOST)
            with socket.create_connection((VISUAL_ONLY_RPC_HOST, server.port), timeout=2) as raw_client:
                raw_client.settimeout(2)
                raw_client.sendall(Channel.FRAME_HEADER.pack(MAX_RPC_FRAME_BYTES + 1, 0))
                self.assertEqual(raw_client.recv(1), b"")
        finally:
            client.close()
            server.close()
            thread.join(1)

    def test_server_caps_compressed_frame_expansion(self):
        """Reject a compressed frame whose expanded data exceeds the cap."""
        max_message_size = 32
        server, thread, client = _start_server(max_message_size=max_message_size)
        try:
            compressed = zlib.compress(b"x" * (max_message_size + 1))
            frame = Channel.FRAME_HEADER.pack(len(compressed), 1) + compressed + Channel.FLUSHER
            with socket.create_connection((VISUAL_ONLY_RPC_HOST, server.port), timeout=2) as raw_client:
                raw_client.settimeout(2)
                raw_client.sendall(frame)
                self.assertEqual(raw_client.recv(1), b"")
        finally:
            client.close()
            server.close()
            thread.join(1)

    def test_secure_rpyc_payload_and_callback_do_not_unpickle(self):
        """Round-trip RPC data and ensure restricted transport rejects pickle."""
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
                attacker = connect(VISUAL_ONLY_RPC_HOST, server.port, config=attacker_config)
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
        """Keep pickle and unrestricted attribute access disabled."""
        self.assertFalse(visual_only_rpyc_config["allow_pickle"])
        self.assertFalse(visual_only_rpyc_config["allow_all_attrs"])
        self.assertFalse(visual_only_rpyc_config["allow_setattr"])
        self.assertFalse(visual_only_rpyc_config["allow_delattr"])
        self.assertEqual(visual_only_rpyc_config["max_message_size"], MAX_RPC_FRAME_BYTES)
