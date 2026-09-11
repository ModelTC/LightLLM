"""Autonomous image/video generation server, backend, and worker."""


def configure_defaults(args) -> None:
    """Enable the image branch only when explicitly requested."""

    if args.generation_backend is None:
        return
    args.generation_backend = "x2v"
    args.generation_model_dir = args.generation_model_dir or args.model_dir
    # Bidirectional image prefixes cannot be partially cached or chunked.
    args.disable_dynamic_prompt_cache = True
    args.disable_chunked_prefill = True


def create_client(args):
    from .client import ImageGenerationClient

    return ImageGenerationClient(args)


def get_start_process():
    from .server import start_image_generation_process

    return start_image_generation_process
