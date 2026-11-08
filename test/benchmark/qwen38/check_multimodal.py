import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

from PIL import Image
import requests

from check_generation import REVISION, verified_checkpoint
from check_scheduler import validate_generation


def image_fixtures(directory, pressure_count):
    directory.mkdir(parents=True, exist_ok=True)
    definitions = {"red": ((255, 0, 0), (256, 256)), "blue": ((0, 0, 255), (256, 256))}
    definitions["rect_blue"] = ((0, 0, 255), (384, 128))
    definitions.update({f"pressure_{i}": ((255, i + 1, 0), (256, 256)) for i in range(pressure_count)})
    payloads, metadata = {}, {}
    for name, (color, size) in definitions.items():
        buffer = BytesIO()
        Image.new("RGB", size, color).save(buffer, format="PNG")
        data = buffer.getvalue()
        (directory / f"{name}.png").write_bytes(data)
        payloads[name] = {"type": "base64", "data": base64.b64encode(data).decode("ascii")}
        metadata[name] = dict(sha256=hashlib.sha256(data).hexdigest(), width=size[0], height=size[1])
    if len({item["sha256"] for item in metadata.values()}) != len(metadata):
        raise ValueError("Cache pressure requires distinct image bytes")
    return payloads, metadata


def prepare_cases(tokenizer, images):
    definitions = {
        "red": (["red"], ["red"]),
        "blue": (["blue"], ["blue"]),
        "rect_blue": (["rect_blue"], ["blue"]),
        "pair": (["red", "blue"], ["red", "blue"]),
        "reversed_pair": (["blue", "red"], ["blue", "red"]),
    }
    definitions.update({name: ([name], []) for name in images if name.startswith("pressure_")})
    cases = {}
    for name, (names, expected) in definitions.items():
        question = (
            "Name the colors of these images in order. Reply with only the two English color words, "
            "separated by a comma."
            if len(names) > 1
            else "What color is the image? Reply with one English word."
        )
        prompt = tokenizer.tokenizer.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"} for _ in names] + [{"type": "text", "text": question}]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        items = []
        for index, image in enumerate(names):
            item = SimpleNamespace(
                image_w=images[image]["width"], image_h=images[image]["height"], token_id=100000000 + index * 1024
            )
            item.token_num = tokenizer.get_image_token_length(item)
            items.append(item)
        expanded = tokenizer.encode(prompt, SimpleNamespace(images=items, audios=[]))
        cases[name] = dict(
            inputs=prompt,
            images=names,
            expected_words=expected,
            prompt_tokens=len(expanded),
            image_tokens=sum(item.token_num for item in items),
            max_new_tokens=1 if not expected else 64,
            ignore_eos=not expected,
        )
    return cases


def validate_image_result(body, case, eos_ids, tokenizer):
    ids = validate_generation(body, case["prompt_tokens"], case["max_new_tokens"], case["ignore_eos"], eos_ids)
    usage = body["input_usage"]
    if (
        usage["input_image_tokens"] != case["image_tokens"]
        or usage["input_audio_tokens"] != 0
        or usage["input_text_tokens"] != case["prompt_tokens"] - case["image_tokens"]
    ):
        raise ValueError("Image/text accounting differs from actual expanded input IDs")
    answer = tokenizer.decode([token for token in ids if token not in eos_ids], skip_special_tokens=True).strip()
    if case["expected_words"] and re.findall(r"[a-z]+", answer.lower()) != case["expected_words"]:
        raise ValueError(f"Image color/order response differs from the fixture: {answer!r}")
    return dict(token_ids=ids, answer=answer, response=body)


def verify_reallocation(events, image_md5):
    allocations = [
        item
        for event in events
        if event["event"] == "alloc"
        for item in event["records"]
        if item["md5"].startswith(image_md5 + "_")
    ]
    if not allocations:
        raise ValueError("No actual image-cache allocation was observed")
    first, last = allocations[0], allocations[-1]
    evicted = {event["record"]["id"] for event in events if event["event"] == "evicted"}
    embedded = {item["id"] for event in events if event["event"] == "embedded" for item in event["records"]}
    if first["id"] == last["id"] or first["token_id"] == last["token_id"] or first["id"] not in evicted:
        raise ValueError("The original image was not demonstrably evicted and assigned new virtual IDs")
    if not {first["id"], last["id"]}.issubset(embedded):
        raise ValueError("Both original and reallocated images require actual embedding-completion evidence")
    return dict(original=first, reallocated=last, original_evicted=True, both_embedded=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--service-metadata", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--cache-events", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if not os.environ.get("EXP_ACTIVE"):
        parser.error("Run through exp")
    verified_checkpoint(args.model_directory)
    service = json.loads(args.service_metadata.read_text())
    if service.get("framework") != "lightllm" or any(
        not service.get(key) for key in ("version", "command", "hardware", "configuration")
    ):
        raise ValueError("Require actual native service provenance")
    command = service["command"]
    capacity = int(command[command.index("--cache_capacity") + 1])
    if not 3 <= capacity <= 16:
        raise ValueError("This bounded two-image workload requires explicit cache capacity between 3 and 16")
    run = Path(os.environ["EXP_RUN_DIR"])
    payloads, images = image_fixtures(run / "images", capacity + 1)
    from lightllm.server.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(str(args.model_directory))
    cases = prepare_cases(tokenizer, images)
    source = {
        name: hashlib.sha256((args.model_directory / name).read_bytes()).hexdigest()
        for name in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "preprocessor_config.json",
            "generation_config.json",
        )
    }
    generation_eos = json.loads((args.model_directory / "generation_config.json").read_text())["eos_token_id"]
    eos_ids = sorted(
        set(generation_eos if isinstance(generation_eos, list) else [generation_eos]) | {tokenizer.eos_token_id}
    )
    reference = json.loads(args.reference.read_text()) if args.reference else None
    if reference and (
        reference["status"] not in ("passed", "recorded")
        or reference["revision"] != REVISION
        or reference["source"] != source
        or reference["cases"] != cases
        or reference["images"] != images
        or reference["eos_ids"] != eos_ids
    ):
        raise ValueError("Require a passing reference with identical images, prompt template, processor and EOS")
    report = dict(
        status="running",
        scope=__doc__,
        revision=REVISION,
        service=service,
        source=source,
        reference=str(args.reference) if args.reference else None,
        images=images,
        cases=cases,
        eos_ids=eos_ids,
        cache_capacity=capacity,
        serial={},
        repeats=[],
        pressure=[],
        waves=[],
    )
    path = run / "multimodal.json"
    active = None

    def save():
        path.write_text(json.dumps(report, indent=2))

    if args.prepare_only:
        report["status"] = "prepared"
        save()
        return

    def request(name, label, cold=False):
        case = cases[name]
        params = dict(
            do_sample=False,
            return_details=True,
            print_eos_token=True,
            disable_prompt_cache=cold,
            max_new_tokens=case["max_new_tokens"],
            ignore_eos=case["ignore_eos"],
        )
        with requests.Session() as client:
            client.trust_env = False
            response = client.post(
                args.base_url.rstrip("/") + "/generate",
                json=dict(
                    inputs=case["inputs"],
                    parameters=params,
                    multimodal_params=dict(images=[payloads[n] for n in case["images"]]),
                ),
                timeout=args.timeout,
            )
        (run / f"{label}.txt").write_text(response.text)
        response.raise_for_status()
        return validate_image_result(response.json(), case, eos_ids, tokenizer.tokenizer)

    def compare(name, result):
        if result["token_ids"] != report["serial"][name]["token_ids"]:
            raise ValueError(f"Repeated/concurrent image IDs differ: {name}")

    try:
        save()
        for name in ("red", "blue", "rect_blue", "pair", "reversed_pair"):
            active = {"phase": "serial", "case": name}
            case = cases[name]
            with requests.Session() as client:
                client.trust_env = False
                response = client.post(
                    args.base_url.rstrip("/") + "/tokens",
                    json=dict(
                        text=case["inputs"], multimodal_params=dict(images=[payloads[n] for n in case["images"]])
                    ),
                    timeout=args.timeout,
                )
            (run / f"count-{name}.txt").write_text(response.text)
            response.raise_for_status()
            if response.json()["ntokens"] != case["prompt_tokens"]:
                raise ValueError("Public image token count differs from actual expansion")
            result = request(name, f"serial-{name}")
            report["serial"][name] = result
            save()
            if reference and result["token_ids"] != reference["serial"][name]["token_ids"]:
                raise ValueError(f"Ordinary/MTP image output IDs differ: {name}")
        for cold in (False, True):
            for name in ("red", "rect_blue", "pair"):
                active = dict(phase="repeat", case=name, cold=cold)
                result = request(name, f"repeat-{int(cold)}-{name}", cold)
                report["repeats"].append(dict(case=name, cold=cold, result=result))
                save()
                compare(name, result)
        for index in range(capacity + 1):
            name = f"pressure_{index}"
            active = dict(phase="cache_pressure", case=name)
            report["pressure"].append(request(name, name))
            save()
        active = dict(phase="after_cache_pressure", case="red")
        report["after_cache_pressure"] = request("red", "after-cache-pressure")
        save()
        compare("red", report["after_cache_pressure"])
        for concurrency in (2, 4):
            active = dict(phase="concurrent", concurrency=concurrency)
            wave = dict(concurrency=concurrency, results={})
            report["waves"].append(wave)
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                pending = {
                    executor.submit(request, name, f"wave-{concurrency}-{index}"): (index, name)
                    for index, name in enumerate(["red", "blue", "pair", "reversed_pair"] * 2)
                }
                for future in as_completed(pending):
                    index, name = pending[future]
                    result = future.result()
                    wave["results"][str(index)] = dict(case=name, result=result)
                    save()
                    compare(name, result)
        if args.cache_events:
            events = [json.loads(line) for line in args.cache_events.read_text().splitlines()]
            digest = hashlib.md5((run / "images/red.png").read_bytes()).hexdigest()
            report["cache_reallocation"] = verify_reallocation(events, digest)
            report["cache_events"] = str(args.cache_events)
        report["status"] = "passed" if reference else "recorded"
    except Exception as error:
        report.update(status="failed", failed_request=active, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
