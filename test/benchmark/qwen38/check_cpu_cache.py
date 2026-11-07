import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

from check_generation import REVISION, verified_checkpoint
from check_prefix_eviction import cache_hit, eviction_cases, validate_eviction
from check_scheduler import generate


def validate_cpu_restore(events, ranks, target_token, initial, restored, minimum_prefix=2048):
    if type(ranks) is not int or ranks < 1 or not initial < restored[0] < restored[1]:
        raise ValueError("Require valid rank count and ordered restore interval")
    fields = {"qsa_raw_keys", "qsa_pooled_keys", "qsa_key_cos", "qsa_key_sin"}
    proof = []
    for rank in range(ranks):
        local = [event for event in events if event.get("rank") == rank and event.get("first_token") == target_token]
        offloads = [
            event
            for event in local
            if event["event"] == "cpu_offload_queued"
            and initial <= event["monotonic"] < restored[0]
            and type(event.get("pages")) is int
            and event["pages"] > 0
        ]
        loads = [
            event
            for event in local
            if event["event"] == "cpu_load_queued"
            and restored[0] <= event["monotonic"] <= restored[1]
            and type(event.get("tokens")) is int
            and event["tokens"] >= 128
            and type(event.get("kv_length")) is int
            and event["kv_length"] >= minimum_prefix
            and type(event.get("request_id")) is int
            and event["request_id"] >= 0
            and event.get("pinned") is True
        ]
        complete = []
        for load in loads:
            extra = [
                event
                for event in local
                if event["event"] == "cpu_extra_load_queued"
                and event.get("request_id") == load.get("request_id")
                and restored[0] <= event["monotonic"] <= load["monotonic"]
                and fields <= set(event.get("slot_fields", []))
                and event.get("lexical_layers")
            ]
            if extra:
                complete.append(load)
        if not offloads or not complete:
            raise ValueError(f"No complete target CPU offload/restore evidence on TP rank {rank}")
        proof.append(dict(rank=rank, offloads=len(offloads), restores=len(complete)))
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--service-metadata", type=Path, required=True)
    parser.add_argument("--cache-events", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if not os.environ.get("EXP_ACTIVE"):
        parser.error("Run through exp")
    verified_checkpoint(args.model_directory)
    service = json.loads(args.service_metadata.read_text())
    if (
        service.get("framework") != "lightllm"
        or any(not service.get(key) for key in ("version", "command", "hardware", "configuration"))
        or service["configuration"].get("cpu_cache") is not True
    ):
        raise ValueError("Require complete actual native CPU-cache service provenance")
    cases = eviction_cases(args.model_directory)
    reference = json.loads(args.reference.read_text())
    if (
        reference.get("status") not in ("recorded", "passed")
        or reference["revision"] != REVISION
        or reference["cases"] != cases
    ):
        raise ValueError("Require a passing reference with identical checkpoint and pressure inputs")
    run = Path(os.environ["EXP_RUN_DIR"])
    report = dict(status="running", scope=__doc__, service=service, revision=REVISION, cases=cases, requests={})
    path = run / "cpu-cache.json"

    def request(label, case):
        result = generate(args.base_url, case, [], args.timeout, run / f"{label}.json")
        if result["token_ids"] != reference["requests"][label]["token_ids"]:
            raise ValueError(f"CPU-cache output differs from qualified reference for {label}")
        cache_hit(result)
        report["requests"][label] = result
        path.write_text(json.dumps(report, indent=2))
        return result

    try:
        target = cases["target"]
        initial = request("initial", target)
        warm = request("warm", target)
        if cache_hit(initial) != 0 or cache_hit(warm) < 2048:
            raise ValueError("Require fresh target followed by actual warm-prefix reuse")
        for name, case in cases.items():
            if name.startswith("small_pressure_"):
                request(name, case)
        request("after_small_pressure", target)
        for name, case in cases.items():
            if name.startswith("pressure_"):
                request(name, case)
        after = request("after_pressure", target)
        if cache_hit(after) < 2048:
            raise ValueError("Evicted target did not restore a long cached prefix")
        request("rewarm", target)

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(
                executor.map(
                    lambda index: generate(args.base_url, target, [], args.timeout, run / f"concurrent_{index}.json"),
                    range(4),
                )
            )
        for index, result in enumerate(results):
            label = f"concurrent_{index}"
            if result["token_ids"] != reference["requests"][label]["token_ids"]:
                raise ValueError("Concurrent restored outputs differ from qualified reference")
            report["requests"][label] = result
        events = [json.loads(line) for line in args.cache_events.read_text().splitlines() if line]
        ranks = service["configuration"]["tp"]
        report["eviction_proof"] = validate_eviction(
            events, ranks, target["inputs"][0], warm["ended"], after["started"]
        )
        report["cpu_restore_proof"] = validate_cpu_restore(
            events, ranks, target["inputs"][0], initial["started"], (after["started"], after["ended"])
        )
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
