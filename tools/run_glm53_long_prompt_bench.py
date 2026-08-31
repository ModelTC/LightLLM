#!/usr/bin/env python3
"""Run SGLang bench_serving with one cached long prompt reused per request."""

import copy
import os
import pickle
import runpy
import sys
import types
from pathlib import Path

from sglang.benchmark.datasets.random import RandomDataset


def _install_prompt_cache() -> None:
    cache_path = Path(os.environ["GLM53_LONG_PROMPT_CACHE"])
    original_load = RandomDataset.load

    def cached_load(self, tokenizer, model_id=None):
        expected = {
            "input_len": self.input_len,
            "output_len": self.output_len,
            "range_ratio": self.range_ratio,
        }
        if cache_path.exists():
            with cache_path.open("rb") as cache_file:
                cached = pickle.load(cache_file)
            if cached["metadata"] != expected:
                raise ValueError(f"Long-prompt cache metadata mismatch: {cached['metadata']} != {expected}")
            row = cached["row"]
            print(f"Loaded long prompt from {cache_path}", flush=True)
        else:
            num_requests = self.num_requests
            self.num_requests = 1
            try:
                row = original_load(self, tokenizer, model_id)[0]
            finally:
                self.num_requests = num_requests
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as cache_file:
                pickle.dump({"metadata": expected, "row": row}, cache_file)
            print(f"Cached long prompt at {cache_path}", flush=True)

        return [copy.copy(row) for _ in range(self.num_requests)]

    RandomDataset.load = cached_load


def main() -> None:
    _install_prompt_cache()

    disaggregation_utils = types.ModuleType("sglang.srt.disaggregation.utils")
    disaggregation_utils.FAKE_BOOTSTRAP_HOST = "fake"
    sys.modules[disaggregation_utils.__name__] = disaggregation_utils

    network_utils = types.ModuleType("sglang.srt.utils.network")
    network_utils.NetworkAddress = object
    sys.modules[network_utils.__name__] = network_utils

    sys.argv = ["sglang.bench_serving", *sys.argv[1:]]
    runpy.run_module("sglang.bench_serving", run_name="__main__")


if __name__ == "__main__":
    main()
