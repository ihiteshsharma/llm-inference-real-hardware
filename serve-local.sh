#!/bin/sh
set -eu

: "${LLAMA_SERVER:?set LLAMA_SERVER to the pinned llama-server binary}"
: "${MODEL:?set MODEL to a checksummed GGUF file}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
CTX_SIZE="${CTX_SIZE:-4096}"
PARALLEL="${PARALLEL:-1}"
THREADS="${THREADS:-4}"
GPU_LAYERS="${GPU_LAYERS:-0}"
ALIAS="${ALIAS:-qwen2.5-3b-instruct}"

if [ "$HOST" != "127.0.0.1" ]; then
  echo "refusing non-localhost bind: $HOST" >&2
  exit 2
fi

exec "$LLAMA_SERVER" \
  --model "$MODEL" \
  --alias "$ALIAS" \
  --host "$HOST" \
  --port "$PORT" \
  --ctx-size "$CTX_SIZE" \
  --parallel "$PARALLEL" \
  --threads "$THREADS" \
  --n-gpu-layers "$GPU_LAYERS" \
  --metrics \
  --perf
