#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/checkpoint-N" >&2
  exit 2
fi

swift infer \
  --adapters "$1" \
  --use_hf true \
  --stream true \
  --temperature 0 \
  --max_new_tokens 4096

