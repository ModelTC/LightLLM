import argparse

from lightllm.server.api_cli import add_cli_args
from lightllm.server.core.objs.start_args_type import StartArgs


def test_multimodal_chunk_flag():
    parser = add_cli_args(argparse.ArgumentParser())
    assert parser.parse_args(["--disable_chunked_mm_input"]).disable_chunked_mm_input is True


def test_multimodal_chunking_keeps_existing_defaults():
    parser = add_cli_args(argparse.ArgumentParser())
    assert parser.parse_args([]).disable_chunked_mm_input is False
    assert StartArgs().disable_chunked_mm_input is False
