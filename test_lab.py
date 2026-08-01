import unittest
from unittest.mock import patch

import bench
import quant_compare


class FakeResponse:
    def __init__(self, lines, state):
        self.lines = lines
        self.state = state

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def __iter__(self):
        for index, line in enumerate(self.lines):
            self.state["line"] = index
            yield line


class BenchmarkTests(unittest.TestCase):
    def test_ttft_starts_at_first_non_empty_content_delta(self):
        state = {"line": -1}
        lines = [
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
            b'data: {"choices":[{"delta":{"content":"hello"}}]}\n',
            b'data: {"choices":[],"usage":{"completion_tokens":2}}\n',
            b'data: [DONE]\n',
        ]

        def clock():
            return {
                -1: 1_000_000_000,
                0: 1_100_000_000,
                1: 1_200_000_000,
                2: 1_300_000_000,
                3: 1_400_000_000,
            }[state["line"]]

        with patch.object(bench.urllib.request, "urlopen", return_value=FakeResponse(lines, state)), patch.object(bench.time, "monotonic_ns", side_effect=clock):
            record = bench.request_once("http://local", "model", "prompt", 8, 900_000_000, "r1")

        self.assertEqual(record["ttft_ms"], 200.0)
        self.assertEqual(record["completion_tokens"], 2)
        self.assertEqual(record.get("response_text"), "hello")

    def test_summary_tolerates_success_with_unknown_token_timing(self):
        rows = [
            {"ok": True, "ttft_ms": 100.0, "tpot_ms": 20.0, "e2e_ms": 220.0},
            {"ok": True, "ttft_ms": None, "tpot_ms": None, "e2e_ms": 250.0},
        ]
        try:
            result = bench.summarize(rows, 200, 30)
        except TypeError as exc:
            self.fail(f"unknown metrics crashed summary: {exc}")
        self.assertEqual(result["completed"], 2)
        self.assertEqual(result["goodput_requests"], 1)
        self.assertEqual(result["ttft_ms_p50"], 100.0)


class QuantizationTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {
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

    def test_selects_smallest_variant_that_passes_every_gate(self):
        result = quant_compare.select(self.manifest, [
            {"name": "Q4_K_M", "quality_pass_rate": 0.9, "ttft_ms_p95": 700, "tpot_ms_p95": 70},
            {"name": "Q5_K_M", "quality_pass_rate": 1.0, "ttft_ms_p95": 600, "tpot_ms_p95": 60},
        ])
        self.assertEqual(result["selected"], "Q4_K_M")

    def test_returns_no_selection_when_every_variant_fails(self):
        result = quant_compare.select(self.manifest, [
            {"name": "Q4_K_M", "quality_pass_rate": 0.7, "ttft_ms_p95": 700, "tpot_ms_p95": 70},
            {"name": "Q5_K_M", "quality_pass_rate": 0.9, "ttft_ms_p95": 1200, "tpot_ms_p95": 60},
        ])
        self.assertIsNone(result["selected"])


if __name__ == "__main__":
    unittest.main()
