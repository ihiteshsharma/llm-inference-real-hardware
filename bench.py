#!/usr/bin/env python3
"""Small OpenAI-compatible streaming benchmark with explicit SLO verdicts."""

import argparse
import concurrent.futures
import json
import math
import time
import urllib.request
from pathlib import Path


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return round(ordered[index], 3)


def request_once(base_url, model, prompt, max_tokens, submitted_ns, request_id):
    started_ns = time.monotonic_ns()
    if submitted_ns is None:
        submitted_ns = started_ns
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    first_chunk_ns = None
    completion_tokens = 0
    chunks = 0
    text_parts = []
    error = None
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                event = json.loads(line[5:].strip())
                choices = event.get("choices") or []
                has_content = any(
                    (choice.get("delta") or {}).get("content") for choice in choices
                )
                if first_chunk_ns is None and has_content:
                    first_chunk_ns = time.monotonic_ns()
                text_parts.extend(
                    content for choice in choices
                    if isinstance(
                        content := (choice.get("delta") or {}).get("content"), str
                    )
                )
                chunks += 1
                usage = event.get("usage") or {}
                completion_tokens = max(
                    completion_tokens, int(usage.get("completion_tokens", 0))
                )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    ended_ns = time.monotonic_ns()
    ttft_ms = (
        (first_chunk_ns - submitted_ns) / 1_000_000
        if first_chunk_ns is not None else None
    )
    generation_ms = (
        (ended_ns - first_chunk_ns) / 1_000_000
        if first_chunk_ns is not None else None
    )
    tpot_ms = (
        generation_ms / max(completion_tokens - 1, 1)
        if completion_tokens and generation_ms is not None else None
    )
    return {
        "request_id": request_id,
        "ok": error is None,
        "error": error,
        "client_queue_ms": round((started_ns - submitted_ns) / 1_000_000, 3),
        "ttft_ms": round(ttft_ms, 3) if ttft_ms is not None else None,
        "tpot_ms": round(tpot_ms, 3) if tpot_ms is not None else None,
        "e2e_ms": round((ended_ns - submitted_ns) / 1_000_000, 3),
        "completion_tokens": completion_tokens,
        "stream_events": chunks,
        "response_text": "".join(text_parts),
    }


def run(args):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as pool:
        futures = [
            pool.submit(
                request_once, args.base_url, args.model, args.prompt,
                args.max_tokens, None, f"r{i + 1:04d}"
            )
            for i in range(args.requests)
        ]
        records = [
            {**future.result(), "offered_concurrency": args.concurrency}
            for future in futures
        ]
    output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    print(json.dumps(summarize(records, args.max_ttft_ms, args.max_tpot_ms), indent=2))


def summarize(records, max_ttft_ms, max_tpot_ms):
    successful = [item for item in records if item["ok"]]
    known_ttft = [x["ttft_ms"] for x in successful if x["ttft_ms"] is not None]
    known_tpot = [x["tpot_ms"] for x in successful if x["tpot_ms"] is not None]
    compliant = [
        item for item in successful
        if item["ttft_ms"] is not None and item["tpot_ms"] is not None
        and item["ttft_ms"] <= max_ttft_ms
        and item["tpot_ms"] <= max_tpot_ms
    ]
    checks = {
        "all_requests_completed": len(successful) == len(records),
        "all_timings_known": (
            len(known_ttft) == len(successful)
            and len(known_tpot) == len(successful)
        ),
        "all_completed_requests_meet_slo": len(compliant) == len(successful),
    }
    verdict = "passed" if records and all(checks.values()) else "failed"
    return {
        "submitted": len(records),
        "completed": len(successful),
        "failed": len(records) - len(successful),
        "goodput_requests": len(compliant),
        "ttft_ms_p50": percentile(known_ttft, 0.50),
        "ttft_ms_p95": percentile(known_ttft, 0.95),
        "tpot_ms_p50": percentile(known_tpot, 0.50),
        "tpot_ms_p95": percentile(known_tpot, 0.95),
        "e2e_ms_p95": percentile([x["e2e_ms"] for x in successful], 0.95),
        "thresholds": {
            "max_ttft_ms": max_ttft_ms,
            "max_tpot_ms": max_tpot_ms,
        },
        "checks": checks,
        "verdict": verdict,
        "conclusion": (
            "Every request met the declared TTFT and TPOT limits."
            if verdict == "passed"
            else "This load is outside the demonstrated SLO-compliant envelope."
        ),
    }


def load_records(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def compare(paths, max_ttft_ms, max_tpot_ms):
    runs = []
    for path in paths:
        records = load_records(path)
        offered = {item.get("offered_concurrency") for item in records}
        if len(offered) != 1 or None in offered:
            raise ValueError(f"{path} must contain one offered_concurrency value")
        runs.append({
            "source": Path(path).name,
            "offered_concurrency": offered.pop(),
            **summarize(records, max_ttft_ms, max_tpot_ms),
        })
    runs.sort(key=lambda item: item["offered_concurrency"])
    passing = [item for item in runs if item["verdict"] == "passed"]
    highest = max(
        (item["offered_concurrency"] for item in passing), default=None
    )
    checks = {
        "compliant_load_found": highest is not None,
        "overload_observed": any(item["verdict"] == "failed" for item in runs),
    }
    return {
        "verdict": "passed" if highest is not None else "failed",
        "checks": checks,
        "highest_slo_compliant_concurrency": highest,
        "runs": runs,
        "conclusion": (
            f"The highest observed SLO-compliant concurrency is {highest}."
            if highest is not None
            else "No tested concurrency demonstrated SLO compliance."
        ),
    }


def self_check():
    rows = [
        {"ok": True, "ttft_ms": 100.0, "tpot_ms": 20.0, "e2e_ms": 200.0},
        {"ok": True, "ttft_ms": 300.0, "tpot_ms": 40.0, "e2e_ms": 500.0},
        {"ok": False, "ttft_ms": None, "tpot_ms": None, "e2e_ms": 10.0},
    ]
    result = summarize(rows, 200, 30)
    assert result["submitted"] == 3
    assert result["completed"] == 2
    assert result["goodput_requests"] == 1
    print("self-check passed")


def parser():
    root = argparse.ArgumentParser()
    root.add_argument("--self-check", action="store_true")
    sub = root.add_subparsers(dest="command")
    execute = sub.add_parser("run")
    execute.add_argument("--base-url", required=True)
    execute.add_argument("--model", required=True)
    execute.add_argument("--prompt", default="Explain why idempotency matters in two sentences.")
    execute.add_argument("--max-tokens", type=int, default=128)
    execute.add_argument("--requests", type=int, default=8)
    execute.add_argument("--concurrency", type=int, default=1)
    execute.add_argument("--max-ttft-ms", type=float, default=1500)
    execute.add_argument("--max-tpot-ms", type=float, default=120)
    execute.add_argument("--output", required=True)
    report = sub.add_parser("summarize")
    report.add_argument("path")
    report.add_argument("--max-ttft-ms", type=float, required=True)
    report.add_argument("--max-tpot-ms", type=float, required=True)
    comparison = sub.add_parser("compare")
    comparison.add_argument("paths", nargs="+")
    comparison.add_argument("--max-ttft-ms", type=float, required=True)
    comparison.add_argument("--max-tpot-ms", type=float, required=True)
    comparison.add_argument("--output", required=True)
    return root


def main():
    args = parser().parse_args()
    if args.self_check:
        self_check()
    elif args.command == "run":
        run(args)
    elif args.command == "summarize":
        print(json.dumps(summarize(
            load_records(args.path), args.max_ttft_ms, args.max_tpot_ms
        ), indent=2))
    elif args.command == "compare":
        result = compare(args.paths, args.max_ttft_ms, args.max_tpot_ms)
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    else:
        parser().print_help()


if __name__ == "__main__":
    main()
