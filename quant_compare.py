#!/usr/bin/env python3
"""Select the smallest quant meeting predeclared quality and latency gates."""

import argparse
import json
from pathlib import Path


def select(manifest, observations):
    thresholds = manifest["thresholds"]
    by_name = {item["name"]: item for item in observations}
    expected = [item["name"] for item in manifest["variants"]]
    observed_names = [name for name in expected if name in by_name]
    missing = [name for name in expected if name not in by_name]
    coverage_complete = not missing
    candidates = []
    for variant in manifest["variants"]:
        observed = by_name.get(variant["name"])
        if not observed:
            continue
        passed = (
            observed["quality_pass_rate"] >= thresholds["min_quality_pass_rate"]
            and observed["ttft_ms_p95"] <= thresholds["max_ttft_ms_p95"]
            and observed["tpot_ms_p95"] <= thresholds["max_tpot_ms_p95"]
        )
        candidates.append({**variant, **observed, "passed": passed})
    winners = sorted(
        (item for item in candidates if item["passed"]),
        key=lambda item: item["size_bytes"],
    )
    selected = winners[0]["name"] if coverage_complete and winners else None
    if not coverage_complete:
        verdict = "incomplete"
        conclusion = (
            "Observed variants were evaluated individually, but cannot select "
            "a quantization until every manifest variant is observed."
        )
    elif selected:
        verdict = "passed"
        conclusion = (
            f"{selected} is the smallest fully observed variant passing every gate."
        )
    else:
        verdict = "failed"
        conclusion = "No fully observed variant passes every declared gate."
    return {
        "verdict": verdict,
        "checks": {
            "coverage_complete": coverage_complete,
            "passing_variant_found": bool(winners),
        },
        "observed_variants": observed_names,
        "missing_variants": missing,
        "coverage_complete": coverage_complete,
        "selected": selected,
        "variants": candidates,
        "conclusion": conclusion,
    }


def self_check():
    manifest = {
        "thresholds": {
            "min_quality_pass_rate": 0.8,
            "max_ttft_ms_p95": 1000,
            "max_tpot_ms_p95": 100,
        },
        "variants": [
            {"name": "Q4_K_M", "size_bytes": 100},
            {"name": "Q5_K_M", "size_bytes": 120},
        ],
    }
    observations = [
        {"name": "Q4_K_M", "quality_pass_rate": 0.7, "ttft_ms_p95": 500, "tpot_ms_p95": 50},
        {"name": "Q5_K_M", "quality_pass_rate": 0.9, "ttft_ms_p95": 600, "tpot_ms_p95": 60},
    ]
    result = select(manifest, observations)
    assert result["selected"] == "Q5_K_M"
    print("self-check passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--manifest")
    parser.add_argument("--observations")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not args.manifest or not args.observations:
        parser.error("--manifest and --observations are required")
    manifest = json.loads(Path(args.manifest).read_text())
    observations = json.loads(Path(args.observations).read_text())
    print(json.dumps(select(manifest, observations), indent=2))


if __name__ == "__main__":
    main()
