from .fp import TritonAttBackend


class NeoTritonAttBackend(TritonAttBackend):
    """Triton prefill for Neo models, whose layers attend bidirectionally inside image spans.

    The layers ask for it with AttControl.image_spans; the Triton GQA backend supports it.
    """
