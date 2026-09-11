"""Image transport from the HTTP worker to GenerationServer."""

from lightllm.server.generationserver.client import GenerationClient
from lightllm.server.generationserver.image.protocol import ImageCancel, ImageResult


class ImageDispatchNotSent(RuntimeError):
    """The send failed before ownership reached GenerationServer."""


class ImageGenerationClient(GenerationClient):
    result_type = ImageResult
    dispatch_error = ImageDispatchNotSent
    label = "image generation"

    def _make_cancel(self, correlation_id: str, reason: str):
        return ImageCancel(correlation_id=correlation_id, reason=reason)
