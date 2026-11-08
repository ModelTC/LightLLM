import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

from check_generation import REVISION, cases_for, verified_checkpoint
from check_scheduler import generate


def eviction_cases(root, pressure_count=12, prompt_length=2369):
    if pressure_count < 2 or not 2049 <= prompt_length <= 4063:
        raise ValueError("Eviction workload requires distinct long prefixes")
    seed = cases_for(root)["long_4063"]
    cases = {
        "target"
        if index == 0
        else f"pressure_{index - 1}": dict(
            inputs=[1000 + index, *seed[: prompt_length - 1]],
            max_new_tokens=32 if index == 0 else 8,
            ignore_eos=True,
        )
        for index in range(pressure_count + 1)
    }
    cases.update(
        {
            f"small_pressure_{index}": dict(inputs=[2000 + index, *seed[:64]], max_new_tokens=8, ignore_eos=True)
            for index in range(20)
        }
    )
    return cases


def cache_hit(result):
    value = result["response"]["tokens"][0]["prompt_cache_len"]
    if type(value) is not int or not 0 <= value < result["response"]["prompt_tokens"]:
        raise ValueError("Invalid actual prompt-cache hit length")
    return value


def validate_eviction(events, ranks, target_token, started, ended):
    if type(ranks) is not int or ranks < 1 or not started < ended:
        raise ValueError("Require valid TP ranks and pressure interval")
    during = [event for event in events if started <= event["monotonic"] <= ended]
    proof = []
    for rank in range(ranks):
        local = [event for event in during if event.get("rank") == rank]
        nodes = [
            event
            for event in local
            if event["event"] == "node_evicted"
            and event["first_token"] == target_token
            and event.get("big_state") is not None
        ]
        small = [
            event
            for event in local
            if event["event"] == "small_state_evicted"
            and event["first_token"] == target_token
            and event.get("small_state") is not None
        ]
        freed = [event for event in local if event["event"] == "state_freed"]

        def observed_free(event, kind):
            return any(
                item["pool"] == event[f"{kind}_pool"]
                and event[f"{kind}_state"] in item["indexes"]
                and item["monotonic"] <= event["monotonic"]
                for item in freed
            )

        if not any(observed_free(event, "big") for event in nodes):
            raise ValueError(f"No observed target KV/big-state eviction on TP rank {rank}")
        if not any(observed_free(event, "small") for event in small):
            raise ValueError(f"No observed target small-state eviction on TP rank {rank}")
        proof.append(dict(rank=rank, target_nodes=len(nodes), target_small_states=len(small)))
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--service-metadata", type=Path, required=True)
    parser.add_argument("--cache-events", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if not os.environ.get("EXP_ACTIVE"):
        parser.error("Run through exp")
    verified_checkpoint(args.model_directory)
    service = json.loads(args.service_metadata.read_text())
    if service.get("framework") != "lightllm" or any(
        not service.get(key) for key in ("version", "command", "hardware", "configuration")
    ):
        raise ValueError("Record complete actual native service provenance")
    cases = eviction_cases(args.model_directory)
    reference = json.loads(args.reference.read_text()) if args.reference else None
    if reference and (
        reference.get("status") not in ("recorded", "passed")
        or reference["revision"] != REVISION
        or reference["cases"] != cases
    ):
        raise ValueError("Require a passing reference with identical checkpoint and eviction inputs")
    run = Path(os.environ["EXP_RUN_DIR"])
    report = dict(
        status="running",
        scope=__doc__,
        revision=REVISION,
        service=service,
        cases=cases,
        reference=str(args.reference) if args.reference else None,
        cache_events=str(args.cache_events),
        requests={},
    )
    path = run / "prefix-eviction.json"

    def request(label, case):
        result = generate(args.base_url, case, [], args.timeout, run / f"{label}.json")
        if reference and result["token_ids"] != reference["requests"][label]["token_ids"]:
            raise ValueError(f"Output differs from ordinary reference for {label}")
        return result

    def save(label, result):
        report["requests"][label] = result
        path.write_text(json.dumps(report, indent=2))

    try:
        target = cases["target"]
        save("initial", request("initial", target))
        save("warm", request("warm", target))
        initial = report["requests"]["initial"]
        warm = report["requests"]["warm"]
        if cache_hit(initial) != 0 or cache_hit(warm) < 2048 or initial["token_ids"] != warm["token_ids"]:
            raise ValueError("Require a fresh target and exact warm-prefix reuse before pressure")
        for name, case in cases.items():
            if name.startswith("small_pressure_"):
                save(name, request(name, case))
        save("after_small_pressure", request("after_small_pressure", target))
        small = report["requests"]["after_small_pressure"]
        if not 0 < cache_hit(small) < cache_hit(warm) or small["token_ids"] != initial["token_ids"]:
            raise ValueError("Small-state pressure must fall back to a shorter big-state prefix with exact output")
        for name, case in cases.items():
            if name.startswith("pressure_"):
                save(name, request(name, case))
        save("after_pressure", request("after_pressure", target))
        after = report["requests"]["after_pressure"]
        if cache_hit(after) >= cache_hit(warm) or after["token_ids"] != initial["token_ids"]:
            raise ValueError("Target must lose its warm prefix and preserve exact output after pressure")
        save("rewarm", request("rewarm", target))
        if cache_hit(report["requests"]["rewarm"]) < 2048:
            raise ValueError("Reallocated target prefix did not become reusable")
        with ThreadPoolExecutor(max_workers=4) as executor:
            pending = {
                f"concurrent_{index}": executor.submit(request, f"concurrent_{index}", target) for index in range(4)
            }
            for label, future in pending.items():
                save(label, future.result())
        if any(report["requests"][label]["token_ids"] != initial["token_ids"] for label in ["rewarm", *pending]):
            raise ValueError("Reallocated target outputs differ under reuse/concurrency")
        events = [json.loads(line) for line in args.cache_events.read_text().splitlines() if line]
        report["eviction_proof"] = validate_eviction(
            events, service["configuration"]["tp"], target["inputs"][0], warm["ended"], after["started"]
        )
        report["status"] = "passed" if reference else "recorded"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
