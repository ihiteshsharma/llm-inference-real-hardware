# LLM Inference on Real Hardware

One repeatable lab for three related engineering questions:

1. How should an OpenAI-compatible LLM server be benchmarked without hiding queueing or failed requests?
2. What workload envelope can a 3B model sustain on one CPU host?
3. Which GGUF quantization is the smallest artifact that still passes declared quality and latency gates?

The harness uses only the Python standard library. Inference is delegated to a pinned `llama-server` build. The code checks have been executed; the hardware-dependent model matrix is intentionally not represented as a universal result because it changes with CPU, memory, build flags, thermal state, and workload.

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

Expected: four unit tests and both self-checks pass. These checks do not download a model or start a listener.

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
CTX_SIZE=4096 PARALLEL=1 THREADS=4 sh ./serve-local.sh
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

## Article 2 lab: find the CPU serving envelope

Record the machine, `llama-server --version`, logical CPU count, memory, and chosen thread count. Run the same request at two controlled envelopes:

```bash
CTX_SIZE=4096 PARALLEL=1 THREADS=4 sh ./serve-local.sh
CTX_SIZE=8192 PARALLEL=4 THREADS=4 sh ./serve-local.sh
```

Restart the server between configurations and repeat the exact workload. Compare success rate, TTFT, TPOT, end-to-end latency, and resident memory from the operating system. The larger context/parallel envelope reserves more KV-cache capacity; it is not free throughput.

Failure exercise: deliberately request a context or concurrency envelope that either fails allocation, rejects work, or violates the latency gate. Preserve the error record. Recovery: reduce context, parallel slots, or offered concurrency one variable at a time and rerun until the gate passes.

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

The selector returns the smallest byte-size variant that passes all declared gates, or `null` when none qualifies. Failure exercise: include a variant that fits memory but misses the quality floor. Recovery: select the next-smallest passing artifact; do not weaken the threshold after seeing the result.

## Interpreting the evidence

- Sourced fact: runtime/model revisions, artifact hashes, CLI behavior, and upstream model metadata.
- Observation: JSONL rows, process memory, errors, output text, and quality grades produced on the test machine.
- Analysis: the chosen operating envelope or quantization, justified from the predeclared gates.

This lab is deliberately small. A production system still needs authenticated ingress, admission control, load shedding, request-size limits, cancellation, telemetry retention, capacity models, rolling upgrades, model provenance controls, and representative multi-tenant traces.

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
