import logging
from math import isfinite
from threading import Event, Thread
from typing import Any, Callable, List, Optional, Sequence, Tuple


_IMAGE_FIELDS = (
    "_type",
    "_data",
    "uuid",
    "token_id",
    "start_index_in_embed_cache",
    "token_num",
    "md5",
    "start_idx",
    "grid_thwd",
    "image_w",
    "image_h",
    "extra_params",
    "data_bytes",
)
_PAYLOAD_VERSION = 1
_MAX_IMAGE_COUNT = 4096
_MAX_CONTAINER_ITEMS = 4096
_MAX_NESTING = 16
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_BATCH_IMAGE_BYTES = 64 * 1024 * 1024
MAX_RPC_FRAME_BYTES = MAX_BATCH_IMAGE_BYTES + 1024 * 1024
logger = logging.getLogger(__name__)


def _validate_image_data_size(data_bytes: Any, total_bytes: int) -> int:
    """Validate one image payload and update the batch byte total."""
    if data_bytes is None:
        return total_bytes
    if type(data_bytes) is not bytes:
        raise ValueError("visual RPC image data must be bytes")
    if len(data_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("visual RPC image exceeds the per-image byte limit")
    total_bytes += len(data_bytes)
    if total_bytes > MAX_BATCH_IMAGE_BYTES:
        raise ValueError("visual RPC image batch exceeds the aggregate byte limit")
    return total_bytes


def _pack_value(value: Any, depth: int = 0) -> Tuple[Any, ...]:
    """Encode a supported value as a tagged tuple of built-in values."""
    if depth > _MAX_NESTING:
        raise ValueError("visual RPC value is nested too deeply")
    if value is None:
        return ("none",)
    if type(value) is bool:
        return ("bool", value)
    if type(value) is int:
        if value < -(2 ** 63) or value >= 2 ** 63:
            raise ValueError("visual RPC integer is out of range")
        return ("int", value)
    if type(value) is float:
        if not isfinite(value):
            raise ValueError("visual RPC float must be finite")
        return ("float", value)
    if type(value) is str:
        return ("str", value)
    if type(value) is bytes:
        return ("bytes", value)
    if type(value) in (list, tuple):
        if len(value) > _MAX_CONTAINER_ITEMS:
            raise ValueError("visual RPC container is too large")
        tag = "list" if type(value) is list else "tuple"
        return (tag, tuple(_pack_value(item, depth + 1) for item in value))
    if type(value) is dict:
        if len(value) > _MAX_CONTAINER_ITEMS:
            raise ValueError("visual RPC mapping is too large")
        if any(type(key) is not str for key in value):
            raise ValueError("visual RPC mapping keys must be strings")
        return (
            "dict",
            tuple((key, _pack_value(item, depth + 1)) for key, item in value.items()),
        )
    raise ValueError("visual RPC values must use supported built-in types")


def _unpack_value(value: Any, depth: int = 0) -> Any:
    """Decode one tagged value while validating its type and nesting."""
    if depth > _MAX_NESTING or type(value) is not tuple or not value:
        raise ValueError("invalid visual RPC value")

    tag = value[0]
    if tag == "none" and len(value) == 1:
        return None
    if tag in ("bool", "int", "float", "str", "bytes") and len(value) == 2:
        item = value[1]
        expected = {
            "bool": bool,
            "int": int,
            "float": float,
            "str": str,
            "bytes": bytes,
        }[tag]
        if type(item) is not expected:
            raise ValueError("invalid visual RPC scalar")
        if tag == "int" and (item < -(2 ** 63) or item >= 2 ** 63):
            raise ValueError("visual RPC integer is out of range")
        if tag == "float" and not isfinite(item):
            raise ValueError("visual RPC float must be finite")
        return item
    if tag in ("list", "tuple") and len(value) == 2 and type(value[1]) is tuple:
        items = value[1]
        if len(items) > _MAX_CONTAINER_ITEMS:
            raise ValueError("visual RPC container is too large")
        unpacked = [_unpack_value(item, depth + 1) for item in items]
        return unpacked if tag == "list" else tuple(unpacked)
    if tag == "dict" and len(value) == 2 and type(value[1]) is tuple:
        entries = value[1]
        if len(entries) > _MAX_CONTAINER_ITEMS:
            raise ValueError("visual RPC mapping is too large")
        result = {}
        for entry in entries:
            if type(entry) is not tuple or len(entry) != 2 or type(entry[0]) is not str:
                raise ValueError("invalid visual RPC mapping entry")
            if entry[0] in result:
                raise ValueError("duplicate visual RPC mapping key")
            result[entry[0]] = _unpack_value(entry[1], depth + 1)
        return result
    raise ValueError("invalid visual RPC value tag")


def pack_image_items(images: Sequence[Any]) -> Tuple[Any, ...]:
    """Serialize image items and enforce their per-image and batch byte budgets."""
    if type(images) not in (list, tuple) or len(images) > _MAX_IMAGE_COUNT:
        raise ValueError("invalid visual RPC image batch")

    records = []
    total_bytes = 0
    for image in images:
        total_bytes = _validate_image_data_size(getattr(image, "data_bytes", None), total_bytes)
        records.append(tuple(_pack_value(getattr(image, field, None)) for field in _IMAGE_FIELDS))
    return (_PAYLOAD_VERSION, tuple(records))


def unpack_image_items(
    payload: Any,
    image_item_factory: Callable[..., Any],
    max_count: Optional[int] = None,
    require_image_data: bool = False,
) -> List[Any]:
    """Validate a versioned image payload and reconstruct its image items."""
    if type(payload) is not tuple or len(payload) != 2:
        raise ValueError("visual RPC payload must be a versioned tuple")
    if type(payload[0]) is not int or payload[0] != _PAYLOAD_VERSION:
        raise ValueError("unsupported visual RPC payload version")
    records = payload[1]
    if type(records) is not tuple or len(records) > _MAX_IMAGE_COUNT:
        raise ValueError("invalid visual RPC image records")
    if max_count is not None and len(records) > max_count:
        raise ValueError("visual RPC image batch exceeds the configured limit")

    images = []
    total_bytes = 0
    for record in records:
        if type(record) is not tuple or len(record) != len(_IMAGE_FIELDS):
            raise ValueError("invalid visual RPC image record")
        values = [_unpack_value(value) for value in record]
        total_bytes = _validate_image_data_size(values[-1], total_bytes)
        image = image_item_factory(type=values[0], data=values[1])
        for field, value in zip(_IMAGE_FIELDS[2:], values[2:]):
            if field == "data_bytes" and value is None:
                continue
            setattr(image, field, value)
        if image._type not in ("url", "base64", "image_size"):
            raise ValueError("invalid visual RPC image type")
        data_bytes = getattr(image, "data_bytes", None)
        if data_bytes is not None and type(data_bytes) is not bytes:
            raise ValueError("visual RPC image data must be bytes")
        if require_image_data and (type(data_bytes) is not bytes or not data_bytes):
            raise ValueError("visual RPC image data is required")
        for field in ("uuid", "md5"):
            value = getattr(image, field)
            if value is not None and type(value) is not str:
                raise ValueError("invalid visual RPC image metadata")
        if require_image_data and (not image.uuid or not image.md5):
            raise ValueError("visual RPC image identifiers are required")
        for field in ("token_id", "start_index_in_embed_cache", "token_num", "start_idx", "image_w", "image_h"):
            value = getattr(image, field)
            if value is not None and type(value) is not int:
                raise ValueError("invalid visual RPC image dimensions or token metadata")
        if image.grid_thwd is not None and (
            type(image.grid_thwd) not in (tuple, list) or any(type(value) is not int for value in image.grid_thwd)
        ):
            raise ValueError("invalid visual RPC image grid")
        if type(image.extra_params) is not dict:
            raise ValueError("invalid visual RPC image options")
        images.append(image)
    return images


class RemoteEvent:
    """Expose a narrow callback for signaling an event owned by the client."""

    def __init__(self, event: Event):
        """Store the event that the remote peer may signal."""
        self._event = event

    def exposed_set(self) -> None:
        """Signal the stored event."""
        self._event.set()


def notify_remote_event_async(remote_event: Any) -> None:
    """Signal a client-owned event on a daemon thread without blocking inference."""

    def _notify() -> None:
        try:
            remote_event.set()
        except Exception:
            logger.exception("Failed to notify the visual inference client")

    worker = Thread(target=_notify, name="visual-rpc-callback", daemon=True)
    try:
        worker.start()
    except RuntimeError:
        logger.exception("Failed to start visual RPC callback thread")
