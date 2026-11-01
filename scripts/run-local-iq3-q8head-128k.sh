#!/bin/sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
runtime_dir="${LLAMA_Q8_RUNTIME:-$HOME/.local/lib/llama-local/current}"
model="${LLAMA_Q8_MODEL:-$repo_dir/models/local/qwen3.8-27b-iq3-mtp-q8head.gguf}"

if [ ! -x "$runtime_dir/llama-server" ] || [ ! -f "$model" ]; then
    echo "Missing installed runtime or experimental IQ3 Q8-head model." >&2
    exit 1
fi

export DYLD_LIBRARY_PATH="$runtime_dir"
case "${LLAMA_LOCAL_SPECULATIVE:-1}" in
    1) set -- --spec-type draft-mtp --spec-draft-n-max 5 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 "$@" ;;
    0) ;;
    *) echo "LLAMA_LOCAL_SPECULATIVE must be 0 or 1" >&2; exit 2 ;;
esac
exec "$runtime_dir/llama-server" \
    -m "$model" --offline --host 127.0.0.1 --port 8000 \
    --fit off -np 1 -c 131072 -ngl 99 -t 5 -tb 5 \
    -b 64 -ub 64 -fa on -ctk q8_0 -ctv q8_0 \
    --reasoning on --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 \
    --presence-penalty 0 --repeat-penalty 1.0 "$@"
