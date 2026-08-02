# LLM Inference on Real Hardware

One repeatable lab for three related engineering questions:

1. How should an OpenAI-compatible LLM server be benchmarked without hiding queueing or failed requests?
2. What workload envelope can a 3B model sustain on one CPU host?
3. Which GGUF quantization is the smallest artifact that still passes declared quality and latency gates?

The harness uses only the Python standard library. Inference is delegated to a pinned `llama-server` build. A complete Q4_K_M run is recorded under [`evidence/q4-first-run`](evidence/q4-first-run); it is evidence for one machine and workload, not a universal hardware claim.

## Reproducibility pins

- Python 3.11 or newer.
- llama.cpp release `b10217`, commit `ddd4ec1`.
- `Qwen/Qwen2.5-3B-Instruct-GGUF` revision `7dabda4d13d513e3e842b20f0d435c732f172cbe`.
- Q4_K_M, Q5_K_M, Q6_K, and Q8_0 artifact hashes and byte sizes are recorded in `quant-manifest.example.json`.

The repository code is MIT licensed. The model is separately distributed under its upstream Qwen license; review that license before downloading or redistributing weights.

## 1. Check the harness

```bash
python3 -m unittest -v test_lab.py
python3 bench.py --self-check
python3 quant_compare.py --self-check
```

Expected: nine unit tests and both self-checks pass. These checks do not download a model or start a listener.

## 2. Obtain the pinned runtime and model

Download or build llama.cpp release `b10217` from the official release, then locate its `llama-server` binary. Download the Q4_K_M file from the pinned model revision:

```text
https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/7dabda4d13d513e3e842b20f0d435c732f172cbe/qwen2.5-3b-instruct-q4_k_m.gguf
```

Verify it before execution:

```bash
shasum -a 256 /absolute/path/to/qwen2.5-3b-instruct-q4_k_m.gguf
```

Expected SHA-256: `626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d`.

## 3. Start a localhost-only server

```bash
LLAMA_SERVER=/absolute/path/to/llama-server \
MODEL=/absolute/path/to/qwen2.5-3b-instruct-q4_k_m.gguf \
CTX_SIZE=4096 PARALLEL=1 THREADS=4 GPU_LAYERS=0 sh ./serve-local.sh
```

`serve-local.sh` refuses a non-loopback bind. The lab has no TLS, authentication, authorization, or prompt redaction and must not be exposed to a network.

## Article 1 lab: benchmark the workload, not one number

Keep the model, prompt, output length, context size, warmup policy, and machine fixed. Vary only offered concurrency:

```bash
python3 bench.py run \
  --base-url http://127.0.0.1:8080 \
  --model qwen2.5-3b-instruct \
  --prompt "Explain why idempotency matters in two sentences." \
  --max-tokens 128 --requests 8 --concurrency 1 \
  --max-ttft-ms 1500 --max-tpot-ms 120 \
  --output results/q4-k-m-c1.jsonl
```

Repeat at concurrency 2, 4, and 8. Save warmup requests separately. Each JSONL row preserves client queue time, TTFT, TPOT, end-to-end latency, completion-token count, errors, and response text. The printed summary reports completion count, failed requests, tail latency, and **goodput**: completed requests that passed both latency limits.

Failure exercise: offer more concurrency than `PARALLEL=1` can serve within the declared limits. Recovery: return to the highest concurrency whose failures are zero and whose goodput equals completed requests. A faster p50 does not compensate for failed or non-compliant work.

### Executed Q4 result

The recorded run used TTFT <= 1500 ms and TPOT <= 120 ms as gates. With one 4096-token slot, concurrency 2 was the highest passing point. Concurrency 4 and 8 completed every request but failed the TTFT gate; returning to concurrency 2 passed again.

| Offered concurrency | Goodput / submitted | TTFT p95 | TPOT p95 | Verdict |
|---:|---:|---:|---:|---|
| 1 | 8 / 8 | 123.939 ms | 27.666 ms | Pass |
| 2 | 8 / 8 | 1131.052 ms | 27.057 ms | Pass |
| 4 | 2 / 8 | 3066.273 ms | 26.951 ms | Fail |
| 8 | 2 / 8 | 7036.714 ms | 25.900 ms | Fail |

This separates queue saturation from token generation: TPOT stayed stable while TTFT crossed the gate. Run `python3 bench.py compare results/q4-control-c*.jsonl` to derive the same bounded-load verdict from raw JSONL files.

## Article 2 lab: find the CPU serving envelope

Record the machine, `llama-server --version`, logical CPU count, memory, and chosen thread count. Run the same request at two controlled envelopes:

```bash
CTX_SIZE=4096 PARALLEL=1 THREADS=4 GPU_LAYERS=0 sh ./serve-local.sh
CTX_SIZE=16384 PARALLEL=4 THREADS=4 GPU_LAYERS=0 sh ./serve-local.sh
```

Restart the server between configurations and repeat the exact workload. `llama-server` divides total `--ctx-size` among parallel slots, so 16384 with four slots preserves the control's 4096 tokens per slot. Using 8192 would reduce each slot to 2048 and change two variables. Compare success rate, TTFT, TPOT, end-to-end latency, and process footprint. The larger envelope reserves more KV-cache capacity; it is not free throughput.

Failure exercise: deliberately request a context or concurrency envelope that either fails allocation, rejects work, or violates the latency gate. Preserve the error record. Recovery: reduce context, parallel slots, or offered concurrency one variable at a time and rerun until the gate passes.

### Executed Q4 envelope result

On an Apple M2 Pro with 16 GiB unified memory, four CPU threads, and GPU offload disabled, the control used one 4096-token slot and a 2.0G physical footprint after warmup. The expanded server used four 4096-token slots and a 2.5G footprint. Its highest passing offered concurrency was 4; concurrency 8 failed TTFT, and returning to 4 passed.

| Envelope | Slots x context | Physical footprint | Highest passing concurrency | First failing concurrency |
|---|---:|---:|---:|---:|
| Control | 1 x 4096 | 2.0G | 2 | 4 |
| Expanded | 4 x 4096 | 2.5G | 4 | 8 |

The result supports a bounded operating envelope on this machine. It does not imply that four slots provide four times the throughput: at concurrency 4, TPOT p95 rose to 62.793 ms and end-to-end p95 to 3719.724 ms.

## Article 3 lab: choose a quantization under constraints

Copy the pinned manifest and download only variants the machine can hold:

```bash
cp quant-manifest.example.json quant-manifest.json
```

Run the identical benchmark matrix and the fixed cases in `quality-cases.json` for each variant. Grade every case with the written criterion before looking at aggregate performance. Create `observations.json` with one object per variant:

```json
[
  {
    "name": "Q4_K_M",
    "quality_pass_rate": 0.0,
    "ttft_ms_p95": 0.0,
    "tpot_ms_p95": 0.0
  }
]
```

Replace zeros only with observed values. Then run:

```bash
python3 quant_compare.py \
  --manifest quant-manifest.json \
  --observations observations.json
```

The selector returns the smallest byte-size variant that passes all declared gates, or `null` when none qualifies. It also refuses to choose when a manifest variant is missing. Failure exercise: include a variant that fits memory but misses the quality floor. Recovery: evaluate the next artifact; do not weaken the threshold after seeing the result.

### Executed Q4 quality result

Q4_K_M passed 4 of 6 fixed cases (`0.666667`) and failed the predeclared `0.8` quality floor. It failed natural version ordering and did not identify queueing/prefill as the dominant cause in a latency diagnosis. Because Q5_K_M, Q6_K, and Q8_0 have not been run, `quant_compare.py` reports `verdict: incomplete` and `selected: null`.

This negative result is intentional: Q4 met the expanded latency gates but did not meet the quality gate. One tested quantization cannot establish a winner.

## Recorded evidence

- [`environment.json`](evidence/q4-first-run/environment.json): machine, runtime, model hash, and exact envelopes.
- [`control-summary.json`](evidence/q4-first-run/control-summary.json): one-slot load and recovery verdict.
- [`expanded-summary.json`](evidence/q4-first-run/expanded-summary.json): four-slot load and recovery verdict.
- [`conclusion.json`](evidence/q4-first-run/conclusion.json): supported conclusion and incomplete quantization status.

Raw JSONL and model files stay ignored because response text can contain prompts and the model is a multi-gigabyte upstream artifact.

## Interpreting the evidence

- Sourced fact: runtime/model revisions, artifact hashes, CLI behavior, and upstream model metadata.
- Observation: JSONL rows, process memory, errors, output text, and quality grades produced on the test machine.
- Analysis: the chosen operating envelope or quantization, justified from the predeclared gates.

## What the lab supports

On the recorded machine and workload, Q4_K_M has bounded CPU serving envelopes, TTFT exposes overload before TPOT does, and reducing offered load restores compliance. The Q4 artifact fails the fixed quality floor.

## What the lab does not support

It does not establish a generally optimal concurrency, prove performance on other hardware, compare CPU with GPU offload, or select a quantization before the remaining manifest variants run. A production system still needs authenticated ingress, admission control, load shedding, request-size limits, cancellation, telemetry retention, capacity models, rolling upgrades, model provenance controls, and representative multi-tenant traces.

## Cleanup

Stop `llama-server`, then remove generated evidence and optional model files:

```bash
rm -r results
rm quant-manifest.json observations.json
```

Delete GGUF files only if they are no longer needed. Never commit model weights, prompts containing sensitive data, or raw production traces.

## Primary references

- [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases)
- [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
- [Qwen2.5 3B GGUF model card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF)
