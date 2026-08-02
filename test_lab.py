import json
import io
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
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
    def test_ttft_runs_from_submission_to_first_non_empty_content(self):
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

        self.assertEqual(record["ttft_ms"], 300.0)
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

    def test_summary_exposes_the_slo_decision(self):
        rows = [
            {"ok": True, "ttft_ms": 100.0, "tpot_ms": 20.0, "e2e_ms": 220.0},
            {"ok": True, "ttft_ms": 120.0, "tpot_ms": 25.0, "e2e_ms": 250.0},
        ]
        result = bench.summarize(rows, 200, 30)
        self.assertEqual(result["verdict"], "passed")
        self.assertTrue(all(result["checks"].values()))
        self.assertIn("every request met", result["conclusion"].lower())

        failed = bench.summarize([
            rows[0],
            {"ok": True, "ttft_ms": None, "tpot_ms": None, "e2e_ms": 250.0},
        ], 200, 30)
        self.assertEqual(failed["verdict"], "failed")
        self.assertFalse(failed["checks"]["all_timings_known"])

    def test_compare_selects_only_observed_passing_concurrency(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for concurrency, ttft in ((1, 100.0), (2, 150.0), (4, 250.0)):
                path = Path(directory) / f"c{concurrency}.jsonl"
                path.write_text(json.dumps({
                    "ok": True,
                    "offered_concurrency": concurrency,
                    "ttft_ms": ttft,
                    "tpot_ms": 20.0,
                    "e2e_ms": 300.0,
                }) + "\n")
                paths.append(path)
            result = bench.compare(paths, 200, 30)
        self.assertEqual(result["highest_slo_compliant_concurrency"], 2)
        self.assertTrue(result["checks"]["overload_observed"])

    def test_run_timestamps_requests_when_workers_begin(self):
        submitted_values = []

        def fake_request(*arguments):
            submitted_values.append(arguments[4])
            return {
                "ok": True,
                "ttft_ms": 100.0,
                "tpot_ms": 20.0,
                "e2e_ms": 220.0,
            }

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                output=str(Path(directory) / "rows.jsonl"),
                requests=2,
                concurrency=1,
                base_url="http://local",
                model="model",
                prompt="prompt",
                max_tokens=8,
                max_ttft_ms=200,
                max_tpot_ms=30,
            )
            with patch.object(bench, "request_once", side_effect=fake_request):
                with redirect_stdout(io.StringIO()):
                    bench.run(args)

        self.assertEqual(submitted_values, [None, None])


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
        self.assertEqual(result["verdict"], "passed")

    def test_returns_no_selection_when_every_variant_fails(self):
        result = quant_compare.select(self.manifest, [
            {"name": "Q4_K_M", "quality_pass_rate": 0.7, "ttft_ms_p95": 700, "tpot_ms_p95": 70},
            {"name": "Q5_K_M", "quality_pass_rate": 0.9, "ttft_ms_p95": 1200, "tpot_ms_p95": 60},
        ])
        self.assertIsNone(result["selected"])
        self.assertEqual(result["verdict"], "failed")

    def test_one_observed_quant_is_explicitly_incomplete(self):
        result = quant_compare.select(self.manifest, [{
            "name": "Q4_K_M",
            "quality_pass_rate": 0.9,
            "ttft_ms_p95": 700,
            "tpot_ms_p95": 70,
        }])
        self.assertEqual(result["verdict"], "incomplete")
        self.assertIsNone(result["selected"])
        self.assertEqual(result["missing_variants"], ["Q5_K_M"])
        self.assertIn("cannot select", result["conclusion"])


class LauncherTests(unittest.TestCase):
    def test_launcher_defaults_to_zero_gpu_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            fake_server = Path(directory) / "fake-server"
            captured = Path(directory) / "arguments.txt"
            fake_server.write_text(
                '#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE"\n'
            )
            fake_server.chmod(0o755)
            environment = {
                **os.environ,
                "LLAMA_SERVER": str(fake_server),
                "MODEL": str(Path(directory) / "model.gguf"),
                "CAPTURE": str(captured),
            }
            subprocess.run(
                ["sh", "serve-local.sh"],
                check=True,
                env=environment,
                capture_output=True,
                text=True,
            )
            arguments = captured.read_text().splitlines()
        position = arguments.index("--n-gpu-layers")
        self.assertEqual(arguments[position + 1], "0")


if __name__ == "__main__":
    unittest.main()
